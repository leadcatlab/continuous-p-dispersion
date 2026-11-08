#!/usr/bin/env bash
# Where slurm/submit_all.sh has got to: jobs, and lines done / running / left per queue.
#
#   ./slurm/progress.sh            # once
#   watch -n 300 ./slurm/progress.sh
cd "$(dirname "$0")/.."
squeue -u "$USER" -h -o "%j %T" | grep '^cpd-' | sort | uniq -c | sed 's/^/  job /'
for q in jobs/queues/*.queue; do
  [ -f "$q" ] || continue
  total=$(wc -l < "$q")
  claimed=$(find "$q.claims" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  done=$(find "$q.claims" -mindepth 2 -maxdepth 2 -name done 2>/dev/null | wc -l)
  printf "  %-6s %5d / %d done, %d running, %d left\n" "$(basename "$q" .queue)" \
    "$done" "$total" "$((claimed - done))" "$((total - claimed))"
done
grep -h "FAILED" jobs/logs/*.out 2>/dev/null | tail -5
