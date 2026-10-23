#!/usr/bin/env python
"""Build results/sweep.csv -- the gold file -- and refuse to write a bad one.

One row per (container, metric, p, seed), read from the run files of the single
results tree, output/<container>/p<p>/<metric>/seed<k>.npz: each carries its
own table row (field "row"), so the table is a pure function of those files
and nothing else has to be kept beside them.

The checks exist because every one of them has already been violated at least
once in this project's history. Rows have been merged from four directories at
nine different commits; cells have silently carried 20 seeds where the rest of
the table had 50; and solution files have disagreed with the rows naming them.
A gold file that quietly holds any of that is worse than no gold file, because
the tables and figures are built from it without further question.

    python finalize_sweep.py                 # check and write
    python finalize_sweep.py --check-only    # report, write nothing
"""
import argparse
import csv
import glob
import io
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

KEY = ("container", "metric", "p", "seed")

# The code a sweep run executes: the solver, its geometry and the per-run
# driver. Rows from commits that differ only outside these (a status page, the
# tables, the plots) come from one algorithm.
SOLVER_PATHS = ("src/tsgo/algorithm", "src/tsgo/container", "src/tsgo/utils",
                "src/tsgo/configuration.py", "src/tsgo/experiments/batch.py",
                "src/tsgo/experiments/cell_run.py")


def load(root, containers=None):
    """(header, rows) as CSV text lines, from every run file under root -- or
    only those of `containers`, so each sweep's table holds its own runs."""
    files = sorted(glob.glob(os.path.join(root, "*", "p*", "*", "seed*.npz")))
    if containers:
        files = [f for f in files if os.path.relpath(f, root).split(os.sep)[0] in containers]
    if not files:
        sys.exit(f"FAIL: no run files under {root}")
    dicts = []
    for f in files:
        with np.load(f) as z:
            if "row" not in z.files:
                sys.exit(f"FAIL: {f} has no table row (written before rows were stored in the run file)")
            dicts.append(json.loads(str(z["row"])))
    fields = list(dicts[0])
    for f, d in zip(files, dicts):
        if list(d) != fields:
            sys.exit(f"FAIL: {f} has different fields from the rest")
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(fields)
    for d in dicts:
        w.writerow([d[k] for k in fields])
    lines = buf.getvalue().splitlines()
    return lines[0], lines[1:]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-root", dest="output_root", default="output")
    ap.add_argument("--out", default="results/sweep.csv")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--containers", nargs="*", default=None,
                    help="only these containers (default: every container under the root)")
    ap.add_argument("--allow-mixed-solver", dest="allow_mixed", action="store_true",
                    help="accept rows from commits whose solver code differs, as when "
                         "selected runs were rerun after a fix; the commits are listed")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="report short cells without failing (use while a sweep is still running)")
    args = ap.parse_args()

    header, rows = load(args.output_root, args.containers)
    cols = header.split(",")
    idx = {c: cols.index(c) for c in KEY}
    commit_i = cols.index("git_commit") if "git_commit" in cols else None

    seen, dupes = {}, []
    cells = defaultdict(set)
    commits = Counter()
    for r in rows:
        c = r.split(",")
        k = tuple(c[idx[x]] for x in KEY)
        if k in seen:
            dupes.append(k)
        seen[k] = r
        cells[(c[idx["container"]], c[idx["metric"]], c[idx["p"]])].add(c[idx["seed"]])
        if commit_i is not None:
            commits[c[commit_i]] += 1

    problems = []
    if dupes:
        problems.append(f"{len(dupes)} duplicated (container, metric, p, seed) keys")
    short = {k: len(v) for k, v in cells.items() if len(v) != args.seeds}
    if short and not args.allow_partial:
        problems.append(f"{len(short)} of {len(cells)} cells do not have {args.seeds} seeds")
    if len(commits) > 1:
        # Several commits is not itself a defect. What matters is whether the
        # solver changed, and a sweep long enough to span days will pick up
        # commits that only touch a driver script or a p-grid. Compare the
        # source tree instead of the label: if src/ is identical across every
        # commit present, the rows came from one algorithm and the file is
        # sound. This distinction is not academic -- the first complete sweep
        # spanned four commits whose src/ trees were byte-identical.
        import subprocess
        cs = sorted(commits)
        differing = []
        for a, b in zip(cs, cs[1:]):
            r = subprocess.run(["git", "diff", "--quiet", a, b, "--", *SOLVER_PATHS],
                               capture_output=True)
            if r.returncode == 1:
                differing.append((a, b))
            elif r.returncode != 0:
                problems.append(f"cannot compare {a}..{b}; is this a git checkout?")
                break
        if differing and args.allow_mixed:
            print("note       : solver code differs between "
                  + ", ".join(f"{a}..{b}" for a, b in differing)
                  + "; accepted (--allow-mixed-solver): rows from reruns after a fix")
        elif differing:
            problems.append("the solver changed mid-sweep, between "
                            + ", ".join(f"{a}..{b}" for a, b in differing))
        else:
            print(f"note       : {len(commits)} commits present, but the solver code is "
                  f"identical across all of them -- one solver, one sweep")

    print(f"runs from  : {args.output_root}/<container>/p<p>/<metric>/seed<k>.npz")
    print(f"rows       : {len(rows)}   unique keys: {len(seen)}")
    print(f"cells      : {len(cells)}   complete at {args.seeds} seeds: {len(cells) - len(short)}")
    if commit_i is not None:
        print(f"commits    : {', '.join(sorted(commits))}")
    if short:
        sample = sorted(short.items())[:8]
        print(f"short cells: {len(short)}" + (f"  e.g. " + ", ".join(
            f"{c}/{m}/p={p}({n})" for (c, m, p), n in sample) if sample else ""))

    if problems:
        print("\n" + "\n".join(f"  PROBLEM: {p}" for p in problems))
        if not args.allow_partial:
            sys.exit("refusing to write a gold file with the problems above")

    if args.check_only:
        print("\ncheck-only: nothing written")
        return

    with open(args.out, "w") as fh:
        fh.write("\n".join([header] + list(seen.values())) + "\n")
    print(f"\nwrote {args.out}  ({len(seen)} rows)")


if __name__ == "__main__":
    main()
