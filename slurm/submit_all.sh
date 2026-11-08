#!/usr/bin/env bash
# Every result, from scratch, on SLURM: every cell with the escape phase on,
# then the equal-budget benchmark and the Gurobi certificates, then the tables,
# figures and report. One command submits the whole chain:
#
#   ./slurm/submit_all.sh            # everything
#   ./slurm/submit_all.sh bench      # from the benchmark on (the sweep is finalized)
#   ./slurm/submit_all.sh report     # only rebuild the report
#   ./slurm/submit_all.sh all        # sweep, extra seeds and benchmark in one queue
#                                    # (benchmark on the previous suite's budgets);
#                                    # needs the previous tables in jobs/previous
#
# Stages, each waiting for the one before:
#
#   sweep   $WORKERS jobs of $CPUS cores (one run per core) sharing one queue:
#           the main grid (240 cells x 20 seeds), Stoyan & Yaskov (129 cells x
#           10 seeds) and the H-box under l-inf at p = 56 (20 seeds); 6,110 runs,
#           each tsgo.experiments.cell_run --escape-factor 1.02, slowest first.
#   finalize  supplement/tables/sweep.csv, hbox_sup_56.csv, stoyan_yaskov.csv,
#           stoyan_yaskov_ext.csv (finalize_sweep.py's checks), and the benchmark queue.
#   bench   $WORKERS jobs on the benchmark queue: TSGO (escape on), IPOPT and SLSQP,
#           10 trials on 45 cells, plus Gurobi on its 6 cells, budget
#           max(60 s, 2 x the new sweep's median run time) per cell.
#   certify one core: Gurobi optimality certificates (cube, tetrahedron, l2), alongside
#           bench; refreshes supplement/REPORT.md when it ends.
#   report  analysis, tables, figures, per-cell plots, configurations and
#           REPORT.md into supplement/.
#
# Workers start as SLURM grants them; nothing waits for all of them. Resumable:
# a finished run (its output file) or trial (its JSON) is never redone, so
# after a timeout or failure, submit the same stage again; the claims of
# unfinished lines are cleared first.
#
# Overrides: WORKERS (16), CPUS per worker (16), MEM per worker (32G), TIME per
# worker (24:00:00), ESCAPE (1.02), SBATCH_FLAGS.
set -euo pipefail
cd "$(dirname "$0")/.."

FROM="${1:-sweep}"
WORKERS="${WORKERS:-16}"
CPUS="${CPUS:-16}"
MEM="${MEM:-32G}"
TIME="${TIME:-24:00:00}"
ESCAPE="${ESCAPE:-1.02}"
SBATCH_FLAGS="${SBATCH_FLAGS:---partition=IllinoisComputes --account=mornik-ic}"
Q=jobs/queues
LOGS=jobs/logs
mkdir -p "$Q" "$LOGS"

if squeue -h -u "$USER" -o %j | grep -q '^cpd-'; then
  echo "cpd- jobs are still queued or running (squeue -u $USER); cancel them or wait." >&2
  exit 1
fi

# "weight command", one line per sweep run; weights as run_sweep.sh gives them.
sweep_lines() {
  run() { echo "$1 python -m tsgo.experiments.cell_run $2 $3 $4 $5 --escape-factor $ESCAPE"; }
  for c in cube tetrahedron h-box pointy-star cage; do
    ps="1 2 3 4 5 6 7 8 9 10 20 30 40 50 100"
    [ "$c" = tetrahedron ] && ps="$ps 35 56 84"
    [ "$c" = cube ] && ps="$ps 27 64"
    w=1; [ "$c" = cage ] && w=3; [ "$c" = pointy-star ] && w=2
    for m in EUCLIDEAN TAXICAB SUP_NORM; do
      for p in $ps; do for ((s = 0; s < 20; s++)); do run $((p * w)) $c $m $p $s; done; done
    done
  done
  for ((s = 0; s < 20; s++)); do run 112 h-box SUP_NORM 56 $s; done
  # Stoyan & Yaskov: the published count and the four above, all three metrics;
  # then the l2 extension of examples 1 and 8.
  for e in 1:93,94,95,96,97 2:91,92,93,94,95 3:86,87,88,89,90 4:81,82,83,84,85 \
           5:79,80,81,82,83 6:30,31,32,33,34 7:58,59,60,61,62 8:73,74,75,76,77; do
    for p in $(echo "${e#*:}" | tr , ' '); do
      for m in EUCLIDEAN TAXICAB SUP_NORM; do
        for ((s = 0; s < 10; s++)); do run $((p * 2)) "stoyan-yaskov-${e%%:*}" $m $p $s; done
      done
    done
  done
  for e in 1:98,99,100 8:78,79,80,81,82,83; do
    for p in $(echo "${e#*:}" | tr , ' '); do
      for ((s = 0; s < 10; s++)); do run $((p * 2)) "stoyan-yaskov-${e%%:*}" EUCLIDEAN $p $s; done
    done
  done
}

# Clears the claims of lines that never finished (a worker was cut off), so
# they are taken again.
reclaim() {
  [ -d "$1.claims" ] || return 0
  local n=0
  for d in "$1.claims"/*/; do
    [ -e "$d/done" ] || { rm -rf "$d"; n=$((n + 1)); }
  done
  [ "$n" -eq 0 ] || echo "[submit] $1: $n unfinished claims cleared"
}

workers() {  # name queue reserve-seconds -> job ids, colon-separated, after $dep
  local ids="" id
  for ((i = 1; i <= WORKERS; i++)); do
    # shellcheck disable=SC2086
    id=$(sbatch --parsable $SBATCH_FLAGS ${dep:+--dependency=afterok:$dep} \
           --job-name="cpd-$1" --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" \
           --output="$LOGS/$1_%j.out" --export=ALL,QUEUE="$2",RESERVE="$3" slurm/worker.sbatch)
    ids="$ids${ids:+:}$id"
  done
  echo "$ids"
}

dep=""
case "$FROM" in
  sweep)
    git rev-parse --short HEAD > supplement/COMMIT
    sweep_lines | sort -k1,1nr -s | cut -d' ' -f2- > "$Q/sweep.queue"
    reclaim "$Q/sweep.queue"
    # The longest run: 30 min, the restart under way, a 15-min escape phase and
    # the start under way: 90 min is held back.
    ids=$(workers sweep "$Q/sweep.queue" 5400)
    echo "[submit] sweep: $(wc -l < "$Q/sweep.queue") runs, $WORKERS workers x $CPUS cores: $ids"
    # shellcheck disable=SC2086
    dep=$(sbatch --parsable $SBATCH_FLAGS --dependency="afterany:$ids" --job-name=cpd-finalize \
            --output="$LOGS/finalize_%j.out" slurm/finalize.sbatch)
    echo "[submit] finalize: $dep"
    ;&
  bench)
    if [ -z "$dep" ]; then   # resubmitted from here: the queue already exists
      reclaim "$Q/bench.queue"
    fi
    ids=$(workers bench "$Q/bench.queue" 600)
    echo "[submit] bench: $WORKERS workers x $CPUS cores: $ids"
    # shellcheck disable=SC2086
    cert=$(sbatch --parsable $SBATCH_FLAGS ${dep:+--dependency=afterok:$dep} --job-name=cpd-certify \
             --cpus-per-task=1 --mem=8G --time=1-12:00:00 --output="$LOGS/certify_%j.out" \
             --wrap="source slurm/env.sh && python -m tsgo.experiments.certify_gurobi && python slurm/make_report.py")
    echo "[submit] certify: $cert"
    # The report waits for the benchmark only; the certificates can take
    # longer, and refresh supplement/REPORT.md themselves when they end.
    dep="$ids"
    ;&
  report)
    # shellcheck disable=SC2086
    id=$(sbatch --parsable $SBATCH_FLAGS ${dep:+--dependency=afterany:$dep} --job-name=cpd-report \
           --cpus-per-task=4 --mem=16G --time=04:00:00 --output="$LOGS/report_%j.out" \
           slurm/report.sbatch)
    echo "[submit] report: $id"
    ;;
  all)
    # Sweep, extra seeds and benchmark in one queue on one pool of workers,
    # longest line first (slurm/all_queue.py), the benchmark on the previous
    # suite's budgets; then finalize, one Gurobi certificate per cell side by
    # side, the merge and the report. The previous suite's tables and budgets
    # must be in $PREVIOUS, and supplement/runs must hold no run files, or every
    # run would be skipped as already done.
    PREVIOUS="${PREVIOUS:-jobs/previous}"
    [ -f "$PREVIOUS/benchmark/budgets.csv" ] && [ -f "$PREVIOUS/tables/sweep.csv" ] || {
      echo "all: $PREVIOUS/tables and $PREVIOUS/benchmark/budgets.csv are needed" >&2; exit 1; }
    if find supplement/runs -name 'seed*.npz' -print -quit 2>/dev/null | grep -q .; then
      echo "all: supplement/runs still holds run files; move them aside first" >&2; exit 1
    fi
    git rev-parse --short HEAD > supplement/COMMIT
    mkdir -p jobs/baseline supplement/benchmark
    git rev-parse --short HEAD > supplement/benchmark/COMMIT
    sweep_lines | python slurm/all_queue.py "$PREVIOUS" "$ESCAPE"
    reclaim "$Q/all.queue"
    ids=$(workers all "$Q/all.queue" 6000)
    echo "[submit] all: $(wc -l < "$Q/all.queue") lines, $WORKERS workers x $CPUS cores: $ids"
    # shellcheck disable=SC2086
    fin=$(sbatch --parsable $SBATCH_FLAGS --dependency="afterany:$ids" --job-name=cpd-finalize \
            --export=ALL,BENCH_QUEUE=prebuilt --output="$LOGS/finalize_%j.out" slurm/finalize.sbatch)
    echo "[submit] finalize: $fin"
    certs=""
    for c in cube tetrahedron; do
      for p in 1 2 3 4 5 6 7 8 9 10 20; do
        # shellcheck disable=SC2086
        id=$(sbatch --parsable $SBATCH_FLAGS --dependency="afterok:$fin" --job-name=cpd-certify \
               --cpus-per-task=1 --mem=8G --time=02:00:00 --output="$LOGS/certify_${c}_p${p}_%j.out" \
               --wrap="source slurm/env.sh && python -m tsgo.experiments.certify_gurobi --cell $c $p")
        certs="$certs${certs:+:}$id"
      done
    done
    echo "[submit] certify: 22 cells, $certs"
    # shellcheck disable=SC2086
    merge=$(sbatch --parsable $SBATCH_FLAGS --dependency="afterany:$certs" --job-name=cpd-merge \
              --cpus-per-task=1 --mem=2G --time=00:10:00 --output="$LOGS/merge_%j.out" \
              --wrap="source slurm/env.sh && python -m tsgo.experiments.certify_gurobi --merge")
    # shellcheck disable=SC2086
    id=$(sbatch --parsable $SBATCH_FLAGS --dependency="afterany:$fin:$merge" --job-name=cpd-report \
           --cpus-per-task=4 --mem=16G --time=04:00:00 --output="$LOGS/report_%j.out" \
           slurm/report.sbatch)
    echo "[submit] merge: $merge  report: $id"
    ;;
  *) echo "unknown stage: $FROM (sweep, bench, report, all)" >&2; exit 2 ;;
esac
echo "[submit] watch: squeue -u $USER -n cpd-sweep,cpd-finalize,cpd-bench,cpd-certify,cpd-report"
echo "[submit] progress: ./slurm/progress.sh"
