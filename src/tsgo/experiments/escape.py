"""Continue finished runs with the escape phase (tsgo_algorithm.escape_restarts).

    python -m tsgo.experiments.escape --containers stoyan-yaskov-6 --p 30-34 --seeds 10 \\
        --out results/stoyan_yaskov_escape.csv --jobs 10

Each run file under --src (the results tree) is loaded and its saved best
configuration goes through the escape phase: restarts whose tabu target is
--factor times the best radius, stopping after the run's restart patience in a
row without a gain or --time-limit seconds. This is the post-run phase used for
the earlier escape diagnosis (Stoyan & Yaskov example 6, h-box l-inf p = 56,
cage l2 p = 100). It is not what `--escape-factor` does now: that option retries
each unproductive restart inside the run (tsgo_algorithm._tsgo_algorithm), so
a fresh run with it differs from a continuation made here.

The source files are never touched. Results go to their own tree,
--dest/<container>/p<p>/<metric>/seed<k>.npz (default output/variants/escape),
and their own CSV: the sweep's columns, with D, energy, feasibility, wall time,
commit and config hash those of the continued run, plus

    escape_factor, escape_patience, escape_time_limit   the phase's options
    escape_starts, escape_wall_s                        what it did
    D_before, source_git_commit, source_config_hash     the run it continued

so no escape run can be read as a default one.
"""
import argparse
import csv
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np

from tsgo.algorithm import energy_functions as ef
from tsgo.algorithm.polish import polish as polish_configuration
from tsgo.algorithm.tsgo_algorithm import (ESCAPE_FACTOR, default_restart_patience, escape_restarts,
                                           escape_seeds)
from tsgo.container.sample_polyhedron import CONTAINERS
from tsgo.experiments.batch import FIELDS, git_commit, load_done, parse_p, run_dir

ESCAPE_FIELDS = ["escape_factor", "escape_patience", "escape_time_limit", "escape_starts",
                 "escape_wall_s", "D_before", "source_git_commit", "source_config_hash"]


def continue_run(src: str, dest_root: str, factor: float, time_limit: float, commit: str):
    with np.load(src, allow_pickle=True) as z:
        f = {k: z[k] for k in z.files}
    row = json.loads(str(f["row"]))
    c, metric, p, seed = str(f["container"]), str(f["metric"]), int(f["p"]), int(f["seed"])
    if int(row["processes"]) != 1:
        raise ValueError(f"{src}: processes={row['processes']}; the winning worker's seed is not recorded")
    if row.get("escape_factor"):
        raise ValueError(f"{src} already had an escape phase")
    iterations = int(row["iterations"])
    patience = int(row["restart_patience"]) if row["restart_patience"] else default_restart_patience(p)
    polish = bool(row["polish"])

    # The worker seed tsgo_algorithm_parallel gives a one-process run.
    worker = int(np.random.SeedSequence(seed).spawn(1)[0].generate_state(1)[0])
    seeds, rays = escape_seeds(worker, iterations, iterations)
    container = CONTAINERS[c]()
    container.set_ray_directions(np.random.default_rng(rays))

    X0, D0 = np.asarray(f["X"], dtype=float), float(f["D"])
    base = float(f["elapsed_time"])
    # Runs from before the progress trace existed have none; the escape phase's
    # starts are still recorded.
    progress = [tuple(r) for r in np.asarray(f.get("progress", np.zeros((0, 2)))).reshape(-1, 2)]
    last = {"X": X0, "D": D0}
    t0 = time.perf_counter()

    def record(X, D):
        if X is not last["X"]:
            last["X"] = X
            last["D"] = (float(polish_configuration(X.copy(), container, metric)[1]) if polish
                         else float(ef.max_feasible_D(X, container, metric)))
        progress.append((base + time.perf_counter() - t0, last["D"]))

    X, D, starts = escape_restarts(
        X0, D0, p, container, metric, seeds, factor, patience, time_limit,
        int(row["Q"]), int(row["beta_max"]), float(row["eps"]), "L-BFGS-B", on_start=record)
    if polish and D > D0:
        X, D = polish_configuration(X, container, metric)
    wall_escape = time.perf_counter() - t0

    # Checked as batch checks a run: a fresh container, caches cleared.
    ef.clear_caches()
    E = ef.E(X, D, CONTAINERS[c](), metric=metric)
    eps = float(row["eps"])
    opts = {"escape_factor": factor, "escape_patience": patience, "escape_time_limit": time_limit}
    out = dict(row)
    out.update({
        "D": f"{D:.12f}", "energy_final": f"{E:.6e}", "feasible": bool(np.isfinite(E) and E < eps),
        "wall_time_s": f"{base + wall_escape:.3f}", "git_commit": commit,
        "config_hash": hashlib.sha256(json.dumps([row["config_hash"], opts]).encode()).hexdigest()[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **opts, "escape_starts": starts, "escape_wall_s": f"{wall_escape:.3f}",
        "D_before": f"{D0:.12f}", "source_git_commit": row["git_commit"],
        "source_config_hash": row["config_hash"],
    })
    dest = run_dir(dest_root, c, p, metric)
    os.makedirs(dest, exist_ok=True)
    npz = os.path.join(dest, f"seed{seed}.npz")
    tmp = os.path.join(dest, f".seed{seed}.npz.tmp")
    f.update(X=X, D=D, energy=E, elapsed_time=base + wall_escape,
             progress=np.array(progress, dtype=float).reshape(-1, 2), git_commit=commit,
             config_hash=out["config_hash"], timestamp=datetime.now(timezone.utc).isoformat(),
             row=json.dumps(out), source=src)
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **f)
    os.replace(tmp, npz)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--containers", nargs="+", required=True, choices=list(CONTAINERS))
    ap.add_argument("--p", required=True)
    ap.add_argument("--metrics", nargs="+", default=["EUCLIDEAN"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--src", default="output", help="Results tree holding the runs to continue.")
    ap.add_argument("--dest", default="output/variants/escape")
    ap.add_argument("--out", required=True, help="Append-only CSV for the continued runs.")
    ap.add_argument("--factor", type=float, default=ESCAPE_FACTOR)
    ap.add_argument("--time-limit", dest="time_limit", type=float, default=900.0)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()

    commit = git_commit()
    done = load_done(args.out)
    todo = []
    for c in args.containers:
        for m in args.metrics:
            for p in parse_p(args.p):
                for s in range(args.seeds):
                    src = os.path.join(run_dir(args.src, c, p, m), f"seed{s}.npz")
                    if (c, m, p, s) in done:
                        continue
                    if not os.path.exists(src):
                        print(f"[escape] missing {src}, skipped")
                        continue
                    todo.append(src)
    print(f"[escape] {len(todo)} runs to continue, factor {args.factor}, "
          f"{args.time_limit:.0f} s each, {args.jobs} jobs, commit {commit}", flush=True)
    if not todo:
        return
    write_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    t_start = time.perf_counter()
    with open(args.out, "a", newline="") as fh, ProcessPoolExecutor(args.jobs) as ex:
        w = csv.DictWriter(fh, fieldnames=FIELDS + ESCAPE_FIELDS)
        if write_header:
            w.writeheader(); fh.flush()
        futs = {ex.submit(continue_run, s, args.dest, args.factor, args.time_limit, commit): s
                for s in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            w.writerow(r); fh.flush()
            print(f"[{i}/{len(todo)}] {r['container']} p={r['p']} seed={r['seed']}  "
                  f"D {float(r['D_before']):.5f} -> {float(r['D']):.5f}  "
                  f"starts={r['escape_starts']} {float(r['escape_wall_s']):.0f}s  "
                  f"feasible={r['feasible']}  [{(time.perf_counter() - t_start) / 60:.1f}m]", flush=True)


if __name__ == "__main__":
    main()
