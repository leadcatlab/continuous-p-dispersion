#!/usr/bin/env bash
# The last step of slurm/submit_escape.sh: each study's per-cell CSVs into
# results/escape_<study>.csv, and how many runs the escape phase raised.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f slurm/env.sh ] && source slurm/env.sh

python - ${STUDIES//:/ } <<'EOF'
import glob, sys
import pandas as pd

for study in sys.argv[1:]:
    files = sorted(glob.glob(f"results/escape-sweep/{study}/*.csv"))
    if not files:
        print(f"[merge] {study}: no results")
        continue
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.sort_values(["container", "metric", "p", "seed"])
    out = f"results/escape_{study.replace('-', '_')}.csv"
    df.to_csv(out, index=False)
    up = df.D > df.D_before * (1 + 1e-7)
    print(f"[merge] {study}: {len(df)} runs in {len(files)} cells -> {out}; "
          f"raised {int(up.sum())}, infeasible {int((~df.feasible.astype(bool)).sum())}")
    cells = df[up].groupby(["container", "metric", "p"]).size()
    if len(cells):
        print(cells.rename("raised").to_string())
EOF
