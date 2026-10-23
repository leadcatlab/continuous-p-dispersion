#!/usr/bin/env bash
# The Stoyan & Yaskov (2013) comparison: their eight containers
# (src/tsgo/container/stoyan_yaskov.py) at their published sphere counts and
# the four counts above, 10 seeds each, every run with the solver's defaults
# through run_sweep.sh. A count is fitted when a seed reaches radius >= 1.
#
#   l2         the published comparison (400 runs)
#   extension  examples 1 and 8 past count + 4, where every queued count fitted (90 runs)
#   l1-linf    the same grid under l1 and l-inf; nothing published to compare with (800 runs)
#
# Results go to the one results tree, output/stoyan-yaskov-<k>/p<p>/<metric>/seed<s>.npz;
# each stage logs to its own file under results/.
#
#   screen -dmS stoyan caffeinate -s -i -m ./run_stoyan_yaskov.sh l2 extension l1-linf
set -uo pipefail
cd "$(dirname "$0")"

CONTAINERS="stoyan-yaskov-1 stoyan-yaskov-2 stoyan-yaskov-3 stoyan-yaskov-4 stoyan-yaskov-5 stoyan-yaskov-6 stoyan-yaskov-7 stoyan-yaskov-8"
# Published count, then the four counts above it.
GRID="stoyan-yaskov-1:93,94,95,96,97 stoyan-yaskov-2:91,92,93,94,95 stoyan-yaskov-3:86,87,88,89,90 stoyan-yaskov-4:81,82,83,84,85 stoyan-yaskov-5:79,80,81,82,83 stoyan-yaskov-6:30,31,32,33,34 stoyan-yaskov-7:58,59,60,61,62 stoyan-yaskov-8:73,74,75,76,77"

for stage in "${@:-l2}"; do
  case "$stage" in
    l2)
      CONTAINERS="$CONTAINERS" GRID=" " EXTRA="$GRID" SEEDS=10 METRICS=EUCLIDEAN \
        LOG=results/stoyan_yaskov.log TABLE=results/stoyan_yaskov.csv \
        FINALIZE_FLAGS=--allow-mixed-solver ./run_sweep.sh ;;
    extension)
      CONTAINERS="stoyan-yaskov-1 stoyan-yaskov-8" GRID=" " \
        EXTRA="stoyan-yaskov-1:98,99,100 stoyan-yaskov-8:78,79,80,81,82,83" SEEDS=10 METRICS=EUCLIDEAN \
        LOG=results/stoyan_yaskov_ext.log TABLE=results/stoyan_yaskov_ext.csv \
        FINALIZE_FLAGS=--allow-mixed-solver ./run_sweep.sh ;;
    l1-linf)
      CONTAINERS="$CONTAINERS" GRID=" " EXTRA="$GRID" SEEDS=10 METRICS="TAXICAB SUP_NORM" \
        LOG=results/stoyan_yaskov.log TABLE=results/stoyan_yaskov.csv \
        FINALIZE_FLAGS=--allow-mixed-solver ./run_sweep.sh ;;
    *) echo "unknown stage: $stage (l2, extension, l1-linf)" >&2; exit 2 ;;
  esac
done
