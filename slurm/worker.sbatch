#!/usr/bin/env bash
# One worker of slurm/submit_all.sh: $SLURM_CPUS_PER_TASK single-core runs at a
# time, taken from the shared queue $QUEUE (one command per line, slowest
# first). Workers start whenever SLURM grants them; each claims a line by
# creating $QUEUE.claims/<line> (mkdir is atomic on the shared filesystem), so
# no line runs twice. A finished line leaves $QUEUE.claims/<line>/done.
#
# A worker stops claiming once less than $RESERVE seconds of its time limit
# remain, longer than any one run takes, so no run is cut off by the job's end.
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
# One core per run: a threaded BLAS would make runs compete and their times incomparable.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

end=$(( $(date +%s) + $(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F'[-:]' '
  NF==4 {print ($1*86400)+($2*3600)+($3*60)+$4; next}
  NF==3 {print ($1*3600)+($2*60)+$3; next}
  NF==2 {print ($1*60)+$2}') ))
echo "[worker $SLURM_JOB_ID] $(hostname), $SLURM_CPUS_PER_TASK slots, queue $QUEUE, until $(date -d @"$end")"

mapfile -t LINES < "$QUEUE"

# Each slot walks the queue once, from the top: a line it finds claimed was
# taken by another slot, so it never has to look back.
slot() {
  local n cmd
  for ((n = 1; n <= ${#LINES[@]}; n++)); do
    [ "$(date +%s)" -lt $((end - RESERVE)) ] || break
    [ -e "$QUEUE.claims/$n" ] && continue
    mkdir "$QUEUE.claims/$n" 2>/dev/null || continue
    cmd=${LINES[n - 1]}
    echo "$SLURM_JOB_ID" > "$QUEUE.claims/$n/job"
    if bash -c "$cmd" 2>&1 | grep -v "^\[TSGO"; [ "${PIPESTATUS[0]}" -eq 0 ]; then
      touch "$QUEUE.claims/$n/done"
    else
      echo "[worker $SLURM_JOB_ID] FAILED line $n: $cmd"
    fi
  done
}

mkdir -p "$QUEUE.claims"
for ((i = 0; i < SLURM_CPUS_PER_TASK; i++)); do slot & done
wait
echo "[worker $SLURM_JOB_ID] done $(date)"
