"""Snapshot of a running sweep, for the progress page.

Reads the results tree (output/<container>/p<p>/<metric>/seed<k>.npz), the
sweep log (results/sweep.log) and the process table, and writes one JSON
document: per cell, a status character per seed -- D done, R running,
F failed, . queued -- with the best radius so far, its gap to the published
reference where there is one, and the median run time; plus totals and an
estimated finish.

The grid mirrors run_sweep.sh's defaults; pass the same overrides if the
sweep was started with different ones.

    python -m tsgo.experiments.sweep_status --out /tmp/status.json
"""
import argparse
import glob
import json
import os
import re
import statistics
import subprocess
import time
from datetime import datetime, timezone

import numpy as np

CONTAINERS = ["cube", "tetrahedron", "h-box", "pointy-star", "cage"]
METRICS = ["EUCLIDEAN", "TAXICAB", "SUP_NORM"]
GRID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100]
EXTRA = {"tetrahedron": [35, 56, 84], "cube": [27, 64]}

# Expected minutes per run by p, from the 2026-10-05 restart-stopping study
# (mean over cube, tetrahedron, h-box and star; l2 and l1), for the estimate of
# remaining work. Only ratios matter: the estimate is rescaled by the
# measured speed once runs have finished.
_COST = [(1, 0.1), (9, 0.6), (10, 0.8), (20, 6.3), (40, 18.8), (60, 27.1), (100, 27.4)]


def expected_minutes(p: int) -> float:
    for (p0, c0), (p1, c1) in zip(_COST, _COST[1:]):
        if p <= p1:
            return c0 + (c1 - c0) * (p - p0) / (p1 - p0) if p >= p0 else c0
    return _COST[-1][1]


def running_runs() -> set:
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    found = set()
    for line in out.splitlines():
        m = re.search(r"tsgo\.experiments\.cell_run (\S+) (\S+) (\d+) (\d+)", line)
        if m and "sh -c" not in line:
            found.add((m[1], m[2], int(m[3]), int(m[4])))
    return found


def failed_runs(log: str) -> set:
    bad = set()
    if not os.path.exists(log):
        return bad
    for line in open(log, errors="ignore"):
        m = re.match(r"(\S+) (EUCLIDEAN|TAXICAB|SUP_NORM) p=(\d+) s=(\d+) .*(FAILED|feasible=False)", line)
        if m:
            bad.add((m[1], m[2], int(m[3]), int(m[4])))
    return bad


def sweep_times(log: str):
    """(start time, commit, finished?) from the sweep log's header lines."""
    start = commit = None
    finished = False
    if os.path.exists(log):
        for line in open(log, errors="ignore"):
            m = re.match(r"\[sweep\] commit (\S+), \d+ runs, .* started (.+)$", line)
            if m:
                commit, start = m[1], m[2].strip()
            if line.startswith("[sweep] done"):
                finished = True
            elif line.startswith("[sweep] commit"):
                finished = False
    return start, commit, finished


def references(path: str) -> dict:
    try:
        d = json.load(open(path))
    except OSError:
        return {}
    out = {}
    for c, by_metric in d.items():
        if not isinstance(by_metric, dict):
            continue
        for m, r in by_metric.items():
            if isinstance(r, dict) and r.get("values"):
                out[(c, m)] = (r.get("label", ""), {int(k): float(v) for k, v in r["values"].items()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--log", default="results/sweep.log")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=11)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    refs = references(os.path.join(os.path.dirname(__file__), "data", "reference_D.json"))
    running, failed = running_runs(), failed_runs(a.log)
    start, commit, finished = sweep_times(a.log)

    containers, totals = [], dict(done=0, running=0, failed=0, queued=0)
    cost_done = cost_left = 0.0
    for c in CONTAINERS:
        metrics = []
        for m in METRICS:
            cells = []
            for p in sorted(GRID + EXTRA.get(c, [])):
                d = os.path.join(a.output_root, c, f"p{p}", m.lower())
                status, Ds, walls = [], [], []
                for s in range(a.seeds):
                    f = os.path.join(d, f"seed{s}.npz")
                    if os.path.exists(f):
                        status.append("D")
                        try:
                            with np.load(f) as z:
                                row = json.loads(str(z["row"]))
                            if str(row.get("feasible")).lower() == "true":
                                Ds.append(float(row["D"]))
                            walls.append(float(row["wall_time_s"]))
                        except Exception:
                            pass
                    elif (c, m, p, s) in running:
                        status.append("R")
                    elif (c, m, p, s) in failed:
                        status.append("F")
                    else:
                        status.append(".")
                st = "".join(status)
                for k, key in (("D", "done"), ("R", "running"), ("F", "failed"), (".", "queued")):
                    totals[key] += st.count(k)
                e = expected_minutes(p)
                cost_done += e * st.count("D")
                cost_left += e * (st.count(".") + 0.5 * st.count("R"))
                cell = {"p": p, "s": st}
                if Ds:
                    cell["best"] = max(Ds)
                if walls:
                    cell["t"] = round(statistics.median(walls), 1)
                ref = refs.get((c, m))
                if ref and p in ref[1]:
                    cell["ref"] = ref[1][p]
                    cell["refLabel"] = ref[0]
                cells.append(cell)
            metrics.append({"metric": m, "cells": cells})
        containers.append({"name": c, "metrics": metrics})

    # Estimated finish: remaining expected work at the pace measured so far.
    eta_hours = None
    elapsed_h = None
    if start:
        try:
            t0 = datetime.strptime(start, "%Y-%m-%d %H:%M:%S %z").timestamp()
        except ValueError:
            t0 = None
        if t0:
            elapsed_h = (time.time() - t0) / 3600
            # The measured pace only means something once a fair sample of
            # runs has finished; before that (a cage run at its D = 0.5 floor
            # takes 30 s where 27 min were expected) use the expected cost.
            if totals["done"] >= 100 and elapsed_h > 0:
                eta_hours = cost_left / (cost_done / elapsed_h)
            else:
                eta_hours = cost_left / 60 / a.jobs

    doc = {
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "started": start, "commit": commit, "finished": finished,
        "seeds": a.seeds, "totals": totals, "total": sum(totals.values()),
        "elapsedHours": None if elapsed_h is None else round(elapsed_h, 2),
        "etaHours": None if eta_hours is None else round(eta_hours, 2),
        "containers": containers,
    }
    with open(a.out, "w") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    print(f"{totals['done']}/{doc['total']} done, {totals['running']} running, "
          f"{totals['failed']} failed; eta {doc['etaHours']} h -> {a.out}")


if __name__ == "__main__":
    main()
