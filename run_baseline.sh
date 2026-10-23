#!/usr/bin/env bash
# The equal-budget baseline comparison, paper/BASELINE_PROTOCOL.md sections 2-4:
# TSGO, IPOPT and SLSQP, 10 trials each on 45 cells, plus Gurobi on the cells it
# can model (l2, convex containers), every trial of a cell under the same
# wall-time budget. Trials land in results/baseline/<solver>/*.json and
# a rerun skips those already written, so the run can be stopped and resumed.
#
#   screen -dmS baseline bash -c './run_baseline.sh >> results/baseline/run.log 2>&1'
set -uo pipefail
cd "$(dirname "$0")"
export PATH="$PWD/.venv/bin:$PATH"
# One core per trial: no solver may quietly use more through a threaded BLAS.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1

OUT="${OUT:-results/baseline}"
JOBS="${JOBS:-10}"
# The pre-registered panel is p = 10, 20, 40; an extension passes its own PS
# (and its own OUT, so the panel's trials and analysis stay as reported).
PS="${PS:-10 20 40}"
# Budgets come from the final sweep alone (the gold file), never from later
# extensions, so a cell's budget does not depend on what was run after it.
SWEEP="${SWEEP:-results/sweep.csv}"
mkdir -p "$OUT"
git rev-parse --short HEAD > "$OUT/COMMIT"

# Budget per cell: max(60 s, 2 x TSGO's median run time in the final sweep).
# Slowest cells first, solvers interleaved so none systematically gets the
# quieter cores.
python - "$OUT/.jobs" "$SWEEP" $PS <<'PY'
import sys, pandas as pd
from tsgo.experiments.stats import load
t = (load(sys.argv[2]).astype({"p": int}).groupby(["container", "metric", "p"])
       .wall_time_s.median().rename("t_median").to_frame())
ps = [int(p) for p in sys.argv[3:]]
jobs = []
for c in ["cube", "tetrahedron", "h-box", "pointy-star", "cage"]:
    for m in ["EUCLIDEAN", "TAXICAB", "SUP_NORM"]:
        for p in ps:
            H = max(60.0, 2.0 * float(t.loc[(c, m, p), "t_median"]))
            for s in range(10):
                for solver in ("tsgo", "ipopt", "slsqp", "gurobi"):
                    if solver == "gurobi" and (m != "EUCLIDEAN" or c not in ("cube", "tetrahedron")):
                        continue
                    jobs.append((H, s, solver, c, m, p))
jobs.sort(key=lambda j: (-j[0], j[1], j[2]))
with open(sys.argv[1], "w") as fh:
    for H, s, solver, c, m, p in jobs:
        fh.write(f"{solver} {c} {m} {p} {s} {H:.1f}\n")
PY
echo "[baseline] $(wc -l < "$OUT/.jobs") trials, $JOBS at a time, commit $(cat "$OUT/COMMIT"), started $(date)"
while read solver c m p s H; do
  [ -f "$OUT/$solver/${c}_${m}_p${p}_s${s}.json" ] || echo "$solver $c $m $p $s $H"
done < "$OUT/.jobs" | OUT="$OUT" xargs -P "$JOBS" -n 6 sh -c '
  out=$(python -m tsgo.experiments.budget_trial "$0" "$1" "$2" "$3" "$4" "$5" "$OUT" 2>>"$OUT/errors.log") \
    || echo "[baseline] FAILED $0 $1 $2 p=$3 s=$4"
  printf "%s\n" "$out" | grep -v "^\[TSGO" || true'
echo "[baseline] finished $(date)"
# One figure per cell: best certified D against time, plots/<container>/p<p>/<metric>/.
python -m tsgo.experiments.baseline_cell_plots --trials "$OUT" --out plots > /dev/null \
  && echo "[baseline] plots -> plots/" || echo "[baseline] FAILED plots"
echo "[baseline] done $(date)"
