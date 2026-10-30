"""One run of one cell, exactly as the sweep runs it.

The result is a single file, output/<container>/p<p>/<metric>/seed<k>.npz,
holding the solution and every field of the run's table row; run_sweep.sh calls
this once per (container, metric, p, seed) and finalize_sweep.py builds
results/sweep.csv from the files. A run whose file already exists is skipped,
so an interrupted sweep resumes where it stopped.

Settings are the solver's defaults: ladder start at density 0.25, restarts at
the best radius until ceil(3 ln p) in a row fail to raise it, capped at 41
starts and 30 minutes, Q = max(3, round(ln p)), final LP tightening.
--escape-factor turns on escape: a restart that ends feasible but no better
retries from its configuration with tabu aimed that much higher, inside the
same time limit; the run is then tagged sweep-escape.

    python -m tsgo.experiments.cell_run CONTAINER METRIC P SEED [--output-root output]
        [--escape-factor 1.02]
"""
import argparse
import os
import subprocess
import types

import tsgo.experiments.batch as batch


def main() -> None:
    ap = argparse.ArgumentParser(description="One sweep run.")
    ap.add_argument("container")
    ap.add_argument("metric", choices=["EUCLIDEAN", "TAXICAB", "SUP_NORM"])
    ap.add_argument("p", type=int)
    ap.add_argument("seed", type=int)
    ap.add_argument("--output-root", dest="output_root", default="output")
    ap.add_argument("--starts", type=int, default=41, help="cap on starts per run")
    ap.add_argument("--restart-patience", dest="restart_patience", type=int, default=None,
                    help="default ceil(3 ln p); 0 turns it off")
    ap.add_argument("--time-limit", dest="time_limit", type=float, default=1800.0,
                    help="seconds per run; 0 turns it off")
    ap.add_argument("--escape-factor", dest="escape_factor", type=float, default=None,
                    help="escape: retry a restart that gains nothing with tabu aimed at this "
                         "multiple of the best radius; off by default")
    a = ap.parse_args()

    npz = os.path.join(batch.run_dir(a.output_root, a.container, a.p, a.metric), f"seed{a.seed}.npz")
    tag = f"{a.container} {a.metric} p={a.p} s={a.seed}"
    if os.path.exists(npz):
        print(f"{tag} already done", flush=True)
        return

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip() or "unknown"
    args = types.SimpleNamespace(d_init=None, packing_density=0.25, output_root=a.output_root,
                                 eps=1e-8, iterations=a.starts, processes=1, Q=None, beta_max=10,
                                 method="L-BFGS-B", restart_patience=a.restart_patience,
                                 time_limit=a.time_limit, escape_factor=a.escape_factor)
    tag_hash = "sweep-escape" if a.escape_factor else "sweep"
    row, failed = batch.run_cell(a.container, a.metric, a.p, a.seed, args, commit, tag_hash)
    print(f"{tag} D={row['D']} feasible={row['feasible']} wall={row['wall_time_s']}s"
          + (f" FAILED {failed}" if failed else ""), flush=True)


if __name__ == "__main__":      # spawned solver workers re-import this module
    main()
