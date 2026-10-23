#!/usr/bin/env bash
# The sweep: every TSGO result in the paper, one run per (container, metric, p, seed).
#
# Grid: the five containers x three metrics at p = 1-10, 20, 30, 40, 50, 100, plus
# the tetrahedron at its tetrahedral numbers 35, 56, 84 and the cube at its
# perfect cubes 27, 64; 20 seeds per cell. Every run uses the solver's defaults
# (tsgo.experiments.cell_run): restarts at the best radius until ceil(3 ln p) in
# a row fail to raise it, capped at 41 starts and 30 minutes.
#
# Results, in one place:
#   output/<container>/p<p>/<metric>/seed<k>.npz   one file per run: the solution and its full table row
#   results/sweep.log                              one line per finished run
#   results/sweep.csv                              the table, built from the run files at the end
#
# Resumable: a run whose file exists is skipped, so rerunning picks up where it
# stopped. Slowest cells start first, so the cores stay busy to the end.
#
#   screen -dmS sweep caffeinate -i ./run_sweep.sh
#
# Overrides (for a partial or test sweep): SEEDS, JOBS, CONTAINERS, METRICS, GRID,
# EXTRA, OUTPUT_ROOT, LOG, TABLE; FINALIZE_FLAGS is passed to finalize_sweep.py
# (--allow-mixed-solver after rerunning selected runs with a fixed solver).
set -uo pipefail
cd "$(dirname "$0")"
export PATH="$PWD/.venv/bin:$PATH"
# One core per run: a threaded BLAS would make runs compete and their times incomparable.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1

SEEDS="${SEEDS:-20}"
JOBS="${JOBS:-11}"
CONTAINERS="${CONTAINERS:-cube tetrahedron h-box pointy-star cage}"
METRICS="${METRICS:-EUCLIDEAN TAXICAB SUP_NORM}"
GRID="${GRID:-1 2 3 4 5 6 7 8 9 10 20 30 40 50 100}"
EXTRA="${EXTRA:-tetrahedron:35,56,84 cube:27,64}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output}"
LOG="${LOG:-results/sweep.log}"
TABLE="${TABLE:-results/sweep.csv}"
mkdir -p "$OUTPUT_ROOT" "$(dirname "$LOG")" "$(dirname "$TABLE")"

# "weight container metric p seed", one line per run. p is a fair proxy for cost
# within a container; the cage and the star are the slow containers at any p.
jobs() {
  for c in $CONTAINERS; do
    ps="$GRID"
    for e in $EXTRA; do [ "${e%%:*}" = "$c" ] && ps="$ps ${e#*:}"; done
    ps="${ps//,/ }"
    w=1; [ "$c" = cage ] && w=3; [ "$c" = pointy-star ] && w=2
    for m in $METRICS; do
      for p in $ps; do
        for ((s = 0; s < SEEDS; s++)); do echo "$((p * w)) $c $m $p $s"; done
      done
    done
  done
}

total=$(jobs | wc -l | tr -d ' ')
echo "[sweep] commit $(git rev-parse --short HEAD), $total runs, $JOBS at a time, started $(date "+%Y-%m-%d %H:%M:%S %z")" >> "$LOG"

jobs | sort -k1,1nr -k5,5n | while read -r w c m p s; do
  [ -f "$OUTPUT_ROOT/$c/p$p/$(echo "$m" | tr 'A-Z' 'a-z')/seed$s.npz" ] || echo "$c $m $p $s"
done | xargs -P "$JOBS" -n 4 sh -c \
  'python -m tsgo.experiments.cell_run "$0" "$1" "$2" "$3" --output-root "'"$OUTPUT_ROOT"'" 2>&1 | grep -v "^\[TSGO"' \
  >> "$LOG"

echo "[sweep] runs finished $(date)" >> "$LOG"
python finalize_sweep.py --output-root "$OUTPUT_ROOT" --out "$TABLE" --seeds "$SEEDS" \
  --containers $CONTAINERS ${FINALIZE_FLAGS:-} >> "$LOG" 2>&1
echo "[sweep] done $(date)" >> "$LOG"
