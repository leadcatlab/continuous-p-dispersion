"""Gather every solver's run files into one tree per solver, and count them.

    output/<container>/p<p>/<metric>/seed<k>.npz              TSGO, sweep configuration
    output/<container>/p<p>/<metric>/pinned/...               TSGO, fixed starting radius
    output/<container>/p<p>/<metric>/comparison/seed<k>.json  TSGO, equal-budget comparison
    ipopt_output/<container>/p<p>/<metric>/seed<k>.json       IPOPT, equal-budget comparison
    slsqp_output/<container>/p<p>/<metric>/seed<k>.json       SLSQP, equal-budget comparison

Comparison trials are copied from the trial directories (results/baseline,
results/baseline_ext), which their driver and analysis still read. A copy is
skipped when an identical file is already in place and refused when a
different one is, so rerunning picks up new trials and never overwrites.
Writes results/inventory.csv: completed runs per solver and cell.

    python -m tsgo.experiments.consolidate
"""
import filecmp
import glob
import os
import re
import shutil
import sys

import pandas as pd

TRIAL_DIRS = ["results/baseline", "results/baseline_ext"]
DEST = {"tsgo": "output/{c}/p{p}/{m}/comparison", "ipopt": "ipopt_output/{c}/p{p}/{m}",
        "slsqp": "slsqp_output/{c}/p{p}/{m}"}
TRIAL = re.compile(r"(?P<c>.+)_(?P<m>EUCLIDEAN|TAXICAB|SUP_NORM)_p(?P<p>\d+)_s(?P<s>\d+)\.json$")
SWEEP = re.compile(r"output/(?P<c>[^/_][^/]*)/p(?P<p>\d+)/(?P<m>[a-z_]+)/seed(?P<s>\d+)\.npz$")


def gather() -> tuple:
    copied = same = 0
    for root in TRIAL_DIRS:
        for solver, dest in DEST.items():
            for f in glob.glob(os.path.join(root, solver, "*.json")):
                g = TRIAL.search(os.path.basename(f))
                if not g:
                    continue
                d = dest.format(c=g["c"], p=g["p"], m=g["m"].lower())
                target = os.path.join(d, f"seed{g['s']}.json")
                if os.path.exists(target):
                    if not filecmp.cmp(f, target, shallow=False):
                        sys.exit(f"REFUSED: {target} exists and differs from {f}")
                    same += 1
                    continue
                os.makedirs(d, exist_ok=True)
                shutil.copy2(f, target)
                copied += 1
    return copied, same


def inventory() -> pd.DataFrame:
    rows = []
    for f in glob.glob("output/*/p*/*/seed*.npz"):
        g = SWEEP.search(f)
        if g:
            rows.append(("tsgo", g["c"], g["m"].upper(), int(g["p"]), int(g["s"])))
    for solver, dest in DEST.items():
        for f in glob.glob(dest.format(c="*", p="*", m="*") + "/seed*.json"):
            parts = f.split("/")
            # .../<c>/p<p>/<m>/[comparison/]seed<k>.json
            c, p, m = parts[-5:-2] if solver == "tsgo" else parts[-4:-1]
            rows.append((f"{solver}_comparison" if solver == "tsgo" else solver,
                         c, m.upper(), int(p[1:]), int(parts[-1][4:-5])))
    df = pd.DataFrame(rows, columns=["solver", "container", "metric", "p", "seed"])
    return (df.groupby(["solver", "container", "metric", "p"]).seed.nunique()
              .rename("runs").reset_index())


def main() -> None:
    copied, same = gather()
    inv = inventory()
    inv.to_csv("results/inventory.csv", index=False)
    print(f"[consolidate] copied {copied}, already in place {same}")
    print(inv.groupby("solver").runs.agg(["count", "sum"]).rename(
        columns={"count": "cells", "sum": "runs"}).to_string())


if __name__ == "__main__":
    main()
