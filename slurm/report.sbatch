#!/usr/bin/env bash
# The last stage of slurm/submit_all.sh: tables, figures, analysis and
# supplement/REPORT.md from whatever has finished. Nothing is written under paper/;
# figures go to supplement/figures, LaTeX tables to supplement/tables/latex.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
export MPLBACKEND=Agg
mkdir -p supplement/figures supplement/tables/latex

step() {  # name command...: run it, note a failure, carry on
  echo "[report] $1"
  shift
  "$@" || echo "[report] FAILED: $*"
}

if [ -f supplement/tables/sweep.csv ]; then
  step summary   python -m tsgo.experiments.summary_final --out supplement/tables/summary_final.csv
  step stats     python -m tsgo.experiments.stats --csv supplement/tables/summary_stats.csv \
                   --latex supplement/tables/latex/summary.tex
  step tables    python -m tsgo.experiments.tables --out supplement/tables/latex
  step figures   python -m tsgo.experiments.make_figures --figures supplement/figures \
                   --archive '' --provenance supplement/figures/PROVENANCE.json \
                   --keep-going
fi
if [ -f supplement/tables/stoyan_yaskov.csv ]; then
  step sy-plot     python -m tsgo.experiments.stoyan_yaskov_plot --out supplement/figures
  step sy-packings python -m tsgo.experiments.stoyan_yaskov_packings --out supplement/figures
fi
if [ -d jobs/baseline/tsgo ]; then
  step bench-analysis python -m tsgo.experiments.baseline_analysis --dir jobs/baseline
  step bench-plots    python -m tsgo.experiments.baseline_plots --dir jobs/baseline --out supplement/figures
  step bench-cells    python -m tsgo.experiments.baseline_cell_plots --trials jobs/baseline --out supplement/benchmark/plots
fi
[ -f supplement/tables/sweep.csv ] && step cells python slurm/cell_summary.py --jobs 4
step export  python -m tsgo.experiments.export_configurations --dest supplement
step verify  bash -c "cd supplement && python verify_packing.py configurations | tail -3"
step report  python slurm/make_report.py
du -sh supplement jobs 2>/dev/null
