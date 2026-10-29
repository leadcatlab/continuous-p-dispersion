#!/usr/bin/env bash
# The escape phase (tsgo.experiments.escape) over every reported run, on SLURM.
#
# One array task per cell: it continues that cell's saved runs, one core per
# seed, all seeds at once. Studies (pick any; default all three):
#
#   sweep          the main grid of run_sweep.sh, 240 cells x 20 seeds
#   stoyan-yaskov  run_stoyan_yaskov.sh's l2, extension and l1-linf grids, 129 cells x 10 seeds
#   h-box-56       the H-box under l-inf at p = 56, seeds 0-19
#
# Needs the source runs: copy the results tree output/<container>/p<p>/<metric>/seed<k>.npz
# from the machine that ran them (it is not in git). They are read, never written.
#
# Results, kept apart from the default runs and from the earlier escape cells:
#   output/variants/escape-sweep/<container>/p<p>/<metric>/seed<k>.npz
#   results/escape-sweep/<study>/<container>_<metric>_p<p>.csv     one per cell
#   results/escape_<study>.csv                                     merged once every task has ended
#   results/escape-sweep/logs/                                     SLURM logs
#
# Resumable: a run already in its cell's CSV is skipped, so submitting again
# after a timeout or a failure redoes only what is missing.
#
#   ./slurm/submit_escape.sh                          # everything
#   ./slurm/submit_escape.sh sweep                    # one study
#   SEEDS_SWEEP=1 SEEDS_SY=1 ./slurm/submit_escape.sh # pilot: seed 0 of every cell
#
# Overrides: SEEDS_SWEEP (20), SEEDS_SY (10), TIME per task (06:00:00),
# MAX_CONCURRENT array tasks (50), SBATCH_FLAGS (e.g. "--partition=cpu --account=abc").
# Cluster setup (modules, the venv) goes in slurm/env.sh; see slurm/env.sh.example.
set -euo pipefail
cd "$(dirname "$0")/.."

STAGES="${*:-sweep stoyan-yaskov h-box-56}"
SEEDS_SWEEP="${SEEDS_SWEEP:-20}"
SEEDS_SY="${SEEDS_SY:-10}"
TIME="${TIME:-06:00:00}"
MAX_CONCURRENT="${MAX_CONCURRENT:-50}"
SBATCH_FLAGS="${SBATCH_FLAGS:-}"
OUT=results/escape-sweep
mkdir -p "$OUT/logs"

# "weight container metric p", one line per cell, as run_sweep.sh weighs them.
cells() {
  case "$1" in
    sweep)
      for c in cube tetrahedron h-box pointy-star cage; do
        ps="1 2 3 4 5 6 7 8 9 10 20 30 40 50 100"
        [ "$c" = tetrahedron ] && ps="$ps 35 56 84"
        [ "$c" = cube ] && ps="$ps 27 64"
        w=1; [ "$c" = cage ] && w=3; [ "$c" = pointy-star ] && w=2
        for m in EUCLIDEAN TAXICAB SUP_NORM; do
          for p in $ps; do echo "$((p * w)) $c $m $p"; done
        done
      done ;;
    stoyan-yaskov)
      # Published count and the four above it, under all three metrics; then
      # the l2 extension of examples 1 and 8.
      for e in 1:93,94,95,96,97 2:91,92,93,94,95 3:86,87,88,89,90 4:81,82,83,84,85 \
               5:79,80,81,82,83 6:30,31,32,33,34 7:58,59,60,61,62 8:73,74,75,76,77; do
        for p in $(echo "${e#*:}" | tr , ' '); do
          for m in EUCLIDEAN TAXICAB SUP_NORM; do echo "$p stoyan-yaskov-${e%%:*} $m $p"; done
        done
      done
      for e in 1:98,99,100 8:78,79,80,81,82,83; do
        for p in $(echo "${e#*:}" | tr , ' '); do echo "$p stoyan-yaskov-${e%%:*} EUCLIDEAN $p"; done
      done ;;
    h-box-56)
      echo "56 h-box SUP_NORM 56" ;;
    *) echo "unknown study: $1 (sweep, stoyan-yaskov, h-box-56)" >&2; exit 2 ;;
  esac
}

deps=""
for study in $STAGES; do
  seeds="$SEEDS_SWEEP"; [ "$study" = stoyan-yaskov ] && seeds="$SEEDS_SY"
  tasks="$OUT/$study.tasks"
  mkdir -p "$OUT/$study"
  cells "$study" | sort -k1,1nr | cut -d' ' -f2- > "$tasks"
  n=$(wc -l < "$tasks" | tr -d ' ')
  # shellcheck disable=SC2086
  id=$(sbatch --parsable $SBATCH_FLAGS --job-name="escape-$study" --array="1-$n%$MAX_CONCURRENT" \
         --cpus-per-task="$seeds" --time="$TIME" --output="$OUT/logs/%x_%A_%a.out" \
         --export=ALL,TASKS="$tasks",STUDY="$study",SEEDS="$seeds" slurm/escape_cell.sbatch)
  echo "[escape] $study: $n cells x $seeds seeds, array job $id"
  deps="$deps:$id"
done

# shellcheck disable=SC2086
id=$(sbatch --parsable $SBATCH_FLAGS --job-name=escape-merge --dependency="afterany$deps" \
       --output="$OUT/logs/%x_%j.out" --export=ALL,STUDIES="${STAGES// /:}" slurm/escape_merge.sbatch)
echo "[escape] merge job $id runs once every task has ended"
