#!/usr/bin/env bash
# After the sweep (slurm/submit_all.sh): the tables, through finalize_sweep.py's
# checks, and the benchmark queue, whose budgets come from the new sweep.
#
#   supplement/tables/sweep.csv             main grid, 240 cells x 20 seeds (the gold file)
#   supplement/tables/hbox_sup_56.csv       the H-box under l-inf at p = 56, 20 seeds
#   supplement/tables/stoyan_yaskov.csv     Stoyan & Yaskov, 120 cells x 10 seeds (l2, l1, l-inf)
#   supplement/tables/stoyan_yaskov_ext.csv examples 1 and 8 past count + 4, l2, 9 cells x 10 seeds
#   jobs/queues/bench.queue    one benchmark trial per line, longest budget first
#
# Fails, and so holds the benchmark back, if any cell is short of seeds: submit
# the sweep again to finish it (./slurm/submit_all.sh).
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p jobs/queues
source slurm/env.sh
ESCAPE="${ESCAPE:-1.02}"

python finalize_sweep.py --seeds 20 --out jobs/.main.csv \
  --containers cube tetrahedron h-box pointy-star cage
python finalize_sweep.py --seeds 10 --out jobs/.sy.csv \
  --containers stoyan-yaskov-{1,2,3,4,5,6,7,8}
# The declared extension: further seeds of Stoyan & Yaskov cells under l2, same
# solver: example 2 p = 95, example 3 p = 88 and example 8 p = 82 (seeds 10-41),
# example 8 p = 81 (seeds 10-73).
if [ -d supplement/runs/variants/extra-seeds ]; then
  python finalize_sweep.py --seeds 32 --output-root supplement/runs/variants/extra-seeds \
    --cell-seeds stoyan-yaskov-8 EUCLIDEAN 81 64 \
    --out supplement/tables/stoyan_yaskov_extra.csv
fi

python - "$ESCAPE" <<'EOF'
import sys
import pandas as pd

def split(src, rest, parts):
    df = pd.read_csv(src)
    for out, mask in parts:
        df[mask(df)].to_csv(out, index=False)
        print(f"[finalize] {out}: {int(mask(df).sum())} rows")
        df = df[~mask(df)]
    df.to_csv(rest, index=False)
    print(f"[finalize] {rest}: {len(df)} rows")

split("jobs/.main.csv", "supplement/tables/sweep.csv", [
    ("supplement/tables/hbox_sup_56.csv",
     lambda d: (d.container == "h-box") & (d.metric == "SUP_NORM") & (d.p == 56))])
ext = {"stoyan-yaskov-1": range(98, 101), "stoyan-yaskov-8": range(78, 84)}
split("jobs/.sy.csv", "supplement/tables/stoyan_yaskov.csv", [
    ("supplement/tables/stoyan_yaskov_ext.csv",
     lambda d: d.apply(lambda r: r.p in ext.get(r.container, ()), axis=1))])

# The benchmark queue, as run_baseline.sh builds it: budget per cell
# max(60 s, 2 x TSGO's median run time in the sweep), longest first, solvers
# interleaved so none systematically gets the quieter cores. Skipped under
# `submit_all.sh all`, where the benchmark already runs beside the sweep on the
# previous suite's budgets (BENCH_QUEUE=prebuilt).
import os
if os.environ.get("BENCH_QUEUE", "sweep") != "sweep":
    print("[finalize] benchmark queue prebuilt; budgets left as they are")
    sys.exit(0)
from tsgo.experiments.stats import load
t = (load("supplement/tables/sweep.csv").astype({"p": int}).groupby(["container", "metric", "p"])
       .wall_time_s.median())
jobs = []
for c in ["cube", "tetrahedron", "h-box", "pointy-star", "cage"]:
    for m in ["EUCLIDEAN", "TAXICAB", "SUP_NORM"]:
        for p in (10, 20, 40):
            H = max(60.0, 2.0 * float(t.loc[(c, m, p)]))
            for s in range(10):
                for solver in ("tsgo", "ipopt", "slsqp", "gurobi"):
                    if solver == "gurobi" and (m != "EUCLIDEAN" or c not in ("cube", "tetrahedron")):
                        continue
                    jobs.append((H, s, solver, c, m, p))
jobs.sort(key=lambda j: (-j[0], j[1], j[2]))
escape = sys.argv[1]
with open("jobs/queues/bench.queue", "w") as fh:
    for H, s, solver, c, m, p in jobs:
        out = f"jobs/baseline/{solver}/{c}_{m}_p{p}_s{s}.json"
        extra = f" {escape}" if solver == "tsgo" else ""
        fh.write(f"[ -f {out} ] || python -m tsgo.experiments.budget_trial "
                 f"{solver} {c} {m} {p} {s} {H:.1f} jobs/baseline{extra}\n")
budgets = sorted({(c, m, p, H) for H, s, solver, c, m, p in jobs}, key=lambda r: -r[3])
pd.DataFrame(budgets, columns=["container", "metric", "p", "budget_s"]).to_csv(
    "supplement/benchmark/budgets.csv", index=False)
core_h = sum(H for H, *_ in jobs) / 3600
print(f"[finalize] jobs/queues/bench.queue: {len(jobs)} trials, {core_h:.0f} core-hours of budget, "
      f"longest {jobs[0][0] / 60:.0f} min")
EOF
rm -f jobs/.main.csv jobs/.sy.csv
mkdir -p jobs/baseline supplement/benchmark
git rev-parse --short HEAD > supplement/benchmark/COMMIT
