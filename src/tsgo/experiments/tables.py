"""Per-container results tables, emitted as LaTeX for the manuscript to input.

The tables in the paper were maintained by hand, and had drifted: the H-box
table reported a baseline of exactly zero with an infinite improvement in six
rows, from a solver that was replaced two commits earlier, against a column of
our own radii that no longer matched the sweep. Generating them removes the
class of error rather than the instance.

Each table compares our best feasible radius per cell against a reference. The
reference is a published value where one exists (data/reference_D.json), and
otherwise the multi-start NLP baseline. For baseline comparisons the achieved
time budget is reported alongside, as a multiple of our own mean run on the
same cell: the nominal budget is ten times ours, but a per-run ceiling binds
wherever our runs are slow, and a comparison is only as good as the budget the
loser actually received.

    tsgo-tables --sweep results/sweep.csv --out paper/generated
"""

import argparse
import os
from typing import Dict, List, Optional

import pandas as pd

from tsgo.experiments.plots import LABEL, load_nlp_baseline, load_reference

METRIC_TEX = {"EUCLIDEAN": r"\ell_2", "TAXICAB": r"\ell_1", "SUP_NORM": r"\ell_\infty"}


def build(sweep: pd.DataFrame, container: str, metric: str,
          nlp: Optional[pd.DataFrame] = None,
          nominal_scale: float = 10.0) -> Dict[str, object]:
    """Rows for one container/metric, plus what the reference was."""
    ours_runs = sweep[(sweep["container"] == container)
                      & (sweep["metric"] == metric)]
    ours = ours_runs[ours_runs["feasible"] == True].groupby("p")["D"].max()   # noqa: E712
    our_time = ours_runs.groupby("p")["wall_time_s"].mean()

    ref = (load_reference().get(container, {}) or {}).get(metric)
    published = {int(k): float(v) for k, v in (ref or {}).get("values", {}).items()}

    if nlp is None:
        nlp = load_nlp_baseline()
    b = nlp[(nlp["container"] == container) & (nlp["metric"] == metric)
            & nlp["feasible"]] if not nlp.empty else nlp
    base = b.groupby("p")["D"].max() if len(b) else pd.Series(dtype=float)
    base_budget = b.groupby("p")["time_limit_s"].max() if len(b) else pd.Series(dtype=float)
    base_runs = b.groupby("p").size() if len(b) else pd.Series(dtype=int)

    rows: List[dict] = []
    for p in sorted(ours.index):
        # Truncated, not rounded: the paper reports radii truncated to eight
        # decimals, and rounding can print a radius above what the
        # configuration supports -- 1/6 rounds up to 0.16666667.
        r = {"p": int(p), "ours": truncate(float(ours[p]))}
        if p in published:
            r["ref"] = published[p]
            r["kind"] = "published"
        elif p in base.index:
            r["ref"] = float(base[p])
            r["kind"] = "baseline"
            t = float(our_time.get(p, 0.0))
            r["budget"] = (float(base_budget[p]) / t) if t > 0 else float("nan")
            r["runs"] = int(base_runs[p])
        else:
            r["ref"] = None
            r["kind"] = "none"
        if r["ref"] is not None:
            r["abs"] = r["ours"] - r["ref"]
            # Relative to the reference, as the manuscript defines it
            # (R_err = A_err / D_Best); this used to divide by our own radius.
            r["rel"] = 100.0 * r["abs"] / r["ref"] if r["ref"] else float("nan")
        rows.append(r)

    return {"rows": rows, "reference": ref, "container": container, "metric": metric,
            "nominal_scale": nominal_scale}


DECIMALS = 8


def truncate(x: float, nd: int = DECIMALS) -> float:
    """x cut, not rounded, to nd decimals -- never above x."""
    import math
    scale = 10.0 ** nd
    t = math.floor(x * scale) / scale
    return t if t <= x else t - 1.0 / scale    # guard the float product


_TEX_ESCAPES = {"%": r"\%", "&": r"\&", "#": r"\#", "_": r"\_"}


def _tex_escape(text: str) -> str:
    """Escape the characters that break a caption, leaving math alone."""
    for ch, rep in _TEX_ESCAPES.items():
        text = text.replace(ch, rep)
    return text


def _fmt(x, nd=8):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "--"
    return f"{x:.{nd}f}"


def _signed(x, nd=8):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "--"
    return (f"$-${abs(x):.{nd}f}" if x < 0 else f"+{x:.{nd}f}")


def _plain_latex(spec: Dict[str, object], label: str) -> str:
    """Best radius per p, for a container with no published reference."""
    container, metric = spec["container"], spec["metric"]
    cap = (f"{LABEL.get(container, container)}, ${METRIC_TEX.get(metric, metric)}$ metric: "
           r"best radius found ($D_{\mathrm{Ours}}$), truncated to eight decimals. "
           "No published values exist for this container.")
    out = [r"\begin{table}[h!]", r"  \centering", f"  \\caption{{{cap}}}", f"  \\label{{{label}}}",
           r"  \begin{tabular}{c|c}", r"    \hline", r"    $p$ & $D_{\mathrm{Ours}}$ \\", r"    \hline"]
    out += [f"    {r['p']} & {_fmt(r['ours'])} \\\\" for r in spec["rows"]]
    out += [r"    \hline", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def to_latex(spec: Dict[str, object], label: str) -> str:
    rows = spec["rows"]
    if all(r["ref"] is None for r in rows):
        return _plain_latex(spec, label)
    ref = spec["reference"]
    container, metric = spec["container"], spec["metric"]
    kinds = {r["kind"] for r in rows if r["ref"] is not None}
    has_budget = any(r["kind"] == "baseline" for r in rows)

    ref_name = (ref["label"] if ref else "multi-start NLP baseline")
    if kinds == {"published"}:
        ref_head = r"$D_{\mathrm{Ref}}$"
    elif kinds == {"baseline"}:
        ref_head = r"$D_{\mathrm{Base}}$"
    else:
        ref_head = r"$D_{\mathrm{Ref}}$"

    cols = ["$p$", r"$D_{\mathrm{Ours}}$", ref_head, r"$A_{\text{err}}$",
            r"$R_{\text{err}}(\%)$"]
    if has_budget:
        cols.append("budget")

    out = [r"\begin{table}[h!]", r"  \centering"]
    cap = (f"{LABEL.get(container, container)}, "
           f"${METRIC_TEX.get(metric, metric)}$ metric: best radius found "
           f"($D_{{\\mathrm{{Ours}}}}$) against {ref_name}. "
           r"$A_{\text{err}}$ and $R_{\text{err}}$ are the absolute and relative "
           r"differences; negative values are cells where the reference is ahead.")
    if has_budget:
        cap += (r" ``budget'' is the wall time the baseline actually received on "
                r"that cell, as a multiple of our own mean run; the nominal "
                f"grant is ${spec['nominal_scale']:.0f}\\times$, and a value "
                r"below it means the per-run ceiling bound.")
    if ref and ref.get("note"):
        # The note is prose from a data file, so it has to be escaped: a bare
        # "%" comments out the rest of the caption and the table stops building.
        cap += " " + _tex_escape(ref["note"])
    out += [f"  \\caption{{{cap}}}", f"  \\label{{{label}}}",
            r"  \begin{tabular}{" + "c|" * (len(cols) - 1) + "c}", r"    \hline",
            "    " + " & ".join(cols) + r" \\", r"    \hline"]

    for r in rows:
        cells = [str(r["p"]), _fmt(r["ours"]), _fmt(r["ref"]),
                 _signed(r.get("abs")), _signed(r.get("rel"), 3)]
        if has_budget:
            b = r.get("budget")
            cells.append("--" if b is None or pd.isna(b) else f"{b:.1f}$\\times$")
        out.append("    " + " & ".join(cells) + r" \\")

    out += [r"    \hline", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


METRICS = ("EUCLIDEAN", "TAXICAB", "SUP_NORM")


def all_metrics_latex(sweep: pd.DataFrame, container: str, label: str,
                      extended_ps=()) -> str:
    """One table per container with all three metrics side by side.

    Where the container has a published Euclidean reference (cube: Packomania;
    tetrahedron: Kazakov et al.), its value and our relative difference sit
    beside the l2 column, and the table spans both columns of the page."""
    ok = sweep[(sweep["container"] == container) & (sweep["feasible"] == True)]  # noqa: E712
    best = ok.groupby(["metric", "p"])["D"].max()
    ps = sorted(ok["p"].unique())
    ref = (load_reference().get(container, {}) or {}).get("EUCLIDEAN")
    pub = {int(k): float(v) for k, v in (ref or {}).get("values", {}).items()}
    name = LABEL.get(container, container)
    sym = {m: METRIC_TEX[m] for m in METRICS}

    def cell(m, p):
        return _fmt(truncate(float(best[(m, p)]))) if (m, p) in best.index else "--"

    if pub:
        head = (["$p$", f"$D$, ${sym['EUCLIDEAN']}$", f"{ref['label']}", r"$R_{\text{err}}$ (\%)",
                 f"$D$, ${sym['TAXICAB']}$", f"$D$, ${sym['SUP_NORM']}$"])
        rows = []
        for p in ps:
            if p in pub and ("EUCLIDEAN", p) in best.index:
                ours = truncate(float(best[("EUCLIDEAN", p)]))
                r = [_fmt(pub[p]), _signed(100.0 * (ours - pub[p]) / pub[p], 4)]
            else:
                r = ["--", "--"]
            rows.append([str(p), cell("EUCLIDEAN", p)] + r + [cell("TAXICAB", p), cell("SUP_NORM", p)])
        cap = (f"{name}: best radius found in each metric, truncated to eight decimals. "
               f"For ${sym['EUCLIDEAN']}$ the published value ({_tex_escape(ref['label'])}) and the "
               r"relative difference $R_{\text{err}} = (D - D_{\mathrm{ref}})/D_{\mathrm{ref}}$ "
               "are given; negative values are cells where the published value is ahead.")
        if ref.get("note"):
            cap += " " + _tex_escape(ref["note"])
        env, colspec = "table*", "c|c|c|c|c|c"
    else:
        head = ["$p$"] + [f"$D$, ${sym[m]}$" for m in METRICS]
        rows = [[str(p)] + [cell(m, p) for m in METRICS] for p in ps]
        cap = (f"{name}: best radius found in each metric, truncated to eight decimals. "
               "No published values exist for this container.")
        env, colspec = "table", "c|c|c|c"
    if extended_ps:
        ps_txt = ", ".join(str(int(x)) for x in sorted(extended_ps))
        cap += (f" Rows for $p = {ps_txt}$ come from additional runs beyond the main grid, "
                "with the same configuration and the same 10 runs per cell.")
    out = [f"\\begin{{{env}}}[t]", "  \\centering", f"  \\caption{{{cap}}}", f"  \\label{{{label}}}",
           f"  \\begin{{tabular}}{{{colspec}}}", "    \\hline", "    " + " & ".join(head) + r" \\",
           "    \\hline"]
    out += ["    " + " & ".join(r) + r" \\" for r in rows]
    out += ["    \\hline", "  \\end{tabular}", f"\\end{{{env}}}"]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Emit the paper's results tables.")
    ap.add_argument("--sweep", default="results/sweep.csv")
    ap.add_argument("--nlp", default="results/nlp_parts")
    ap.add_argument("--out", default="paper/generated")
    ap.add_argument("--metric", default="EUCLIDEAN")
    ap.add_argument("--containers", nargs="+",
                    default=["cube", "tetrahedron", "h-box", "pointy-star", "cage"])
    args = ap.parse_args()

    sweep = pd.read_csv(args.sweep, skipinitialspace=True)
    nlp = load_nlp_baseline(args.nlp)
    os.makedirs(args.out, exist_ok=True)

    sweep["feasible"] = sweep["feasible"].astype(str).str.lower().eq("true")
    for c in args.containers:
        slug = c.replace("-", "_")
        path = os.path.join(args.out, f"results_{slug}.tex")
        with open(path, "w") as fh:
            fh.write(all_metrics_latex(sweep, c, f"tab:{slug}"))
        print(f"[tables] {path}  (all three metrics)")
    for c in args.containers:
        if sweep[(sweep["container"] == c) & (sweep["metric"] == args.metric)].empty:
            print(f"[tables] skip {c}/{args.metric}: no runs")
            continue
        spec = build(sweep, c, args.metric, nlp=nlp)
        slug = c.replace("-", "_")
        label = f"tab:{slug}_{args.metric.lower()}"
        path = os.path.join(args.out, f"results_{slug}_{args.metric.lower()}.tex")
        with open(path, "w") as fh:
            fh.write(to_latex(spec, label))
        n_ref = sum(1 for r in spec["rows"] if r["ref"] is not None)
        print(f"[tables] {path}  ({len(spec['rows'])} rows, {n_ref} with a reference)")


if __name__ == "__main__":
    main()
