#!/usr/bin/env bash
# One cell of slurm/submit_escape.sh: line $SLURM_ARRAY_TASK_ID of $TASKS
# ("container metric p"), its seeds 0..$SEEDS-1, one core each.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f slurm/env.sh ] && source slurm/env.sh
# One core per run: a threaded BLAS would make runs compete and their times incomparable.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

read -r c m p < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKS")
mkdir -p "results/escape-sweep/$STUDY"
echo "[escape] $STUDY: $c $m p=$p, seeds 0-$((SEEDS - 1)) on $SLURM_CPUS_PER_TASK cores, $(hostname)"
python -m tsgo.experiments.escape --containers "$c" --metrics "$m" --p "$p" --seeds "$SEEDS" \
  --src output --dest output/variants/escape-sweep \
  --out "results/escape-sweep/$STUDY/${c}_$(echo "$m" | tr 'A-Z' 'a-z')_p$p.csv" \
  --factor 1.02 --time-limit 900 --jobs "$SLURM_CPUS_PER_TASK" 2>&1 | grep -v "^\[TSGO"
