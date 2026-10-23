# Supplement: every reported result

This folder holds all of the paper's final results: every run of the cluster suite, the best configuration of every cell, the tables and figures built from them, the equal-budget benchmark and the Gurobi certificates. Everything is organized by container, metric and p. A standalone checker verifies any configuration using only Python 3 and numpy, without the solver.

```
python verify_packing.py configurations                                  # all of them
python verify_packing.py configurations/stoyan-yaskov-8/euclidean/p80.csv # one
```

All results come from one suite: solver commit in `COMMIT`, every run with the escape phase on, one core per run on the campus cluster. `REPORT.md` summarizes it on one page.

## Layout

| path | what |
|---|---|
| `runs/<container>/<metric>/p<p>/seed<k>.npz` | One file per run (6,110): the final centres `X`, radius `D`, the progress trace and the run's full table row. Main grid: 5 containers × 3 metrics × 16 p × 20 seeds, plus the H-box under l-inf at p = 56. Stoyan & Yaskov: 129 cells × 10 seeds |
| `runs/variants/extra-seeds/<container>/euclidean/p<p>/` | Extra seeds of four Stoyan & Yaskov cells under l2, same solver; a declared extension: example 2 p = 95, example 3 p = 88 and example 8 p = 82 (seeds 10-41), example 8 p = 81 (seeds 10-73) |
| `configurations/<container>/<metric>/p<p>.csv` | Centres of the best seed of every cell: 240 main-grid cells, 129 Stoyan & Yaskov cells, the H-box under l-inf at p = 56, and the extra-seeds cells under `configurations/variants/` (374 in all) |
| `configurations.csv` | Index of those files: study, container, metric, p, radius, seed, file |
| `containers/*.json` | Each container as a list of planar faces, given as vertex rings with optional holes, each with a unit normal pointing into the container |
| `tables/sweep.csv` | The gold file: one row per main-grid run (4,800 rows) |
| `tables/hbox_sup_56.csv`, `tables/stoyan_yaskov.csv`, `tables/stoyan_yaskov_ext.csv`, `tables/stoyan_yaskov_extra.csv` | The other studies' run tables, same columns |
| `tables/summary_final.csv`, `tables/summary_stats.csv` | One row per cell: best, median and spread of D over the seeds |
| `tables/cell_summary.csv` | Per cell: feasibility, re-certification margin, fallback restarts, monotonicity in p, wall time |
| `tables/latex/` | The LaTeX tables built from `sweep.csv` |
| `figures/` | Figures built from these results; `PROVENANCE.json` names the run behind each one |
| `benchmark/analysis/` | Equal-budget benchmark (TSGO vs IPOPT, SLSQP, Gurobi): `summary.md`, performance-profile ratios, time to target, per-cell results, and `E_audit.csv.gz`, the reported vs certified radius of every converged local solve |
| `benchmark/budgets.csv` | Wall-time budget per benchmark cell |
| `benchmark/plots/<container>/<metric>/p<p>.png` | Best radius over time for each solver in each benchmark cell |
| `certificates/gurobi.csv` | Gurobi upper bounds on D (convex containers, l2), warm-started from TSGO's best |
| `verify_packing.py` | The checker |

The raw benchmark trials are in `benchmark/trials/<solver>/<container>_<metric>_p<p>_s<seed>.json`, one JSON trace per solver run (1,410 files, 448 MB). The analysis tables above were built from them: `python -m tsgo.experiments.baseline_analysis --dir supplement/benchmark/trials`.

## Configuration files

Each file starts with `#` lines giving:

- the container and its face list (`geometry`, relative to the file);
- the metric, `p` and the claimed `radius`;
- the run it came from (its file under `runs/`, the seed and the solver commit).

Then comes `i,x,y,z`, one row per centre. Values are written at full double precision: each one reads back as exactly the stored number.

## How the check works

For each file, the script recomputes from the centres and the container's faces the largest radius r at which the configuration is feasible:

    r = min( min_{i<j} d(c_i, c_j) / 2 ,  min_i d(c_i, boundary) ),

where d is the file's metric (l2, l1 or l-inf), and every centre must lie inside the container. It then prints r beside the claimed radius. Balls of radius r centred at the c_i then have disjoint interiors and lie inside the container.

- **Inside-ness.** A ray is cast from each centre and its crossings with the faces are counted; an odd count means the centre is inside. This is done along five generic directions, and all five must agree.
- **Distance to a face.** This is exact in all three norms. The nearest point of a face lies in one of two places:
  - inside the face, where the norm's ball first touches the face's plane. That distance is |n·c − b| / ‖n‖\*, using the dual norm.
  - on one of the face's edges. Point-to-segment distances are computed in closed form: by projection for l2, and from the breakpoints of the piecewise-linear objective for l1 and l-inf.
- **Tolerance.** A configuration passes when r ≥ claimed × (1 − 1e-9). The solver shrinks the radius it reports by a relative 1e-9 so that it is safely feasible, so the checker's r comes out about 1e-9 above the claimed value.
- **Result.** All 371 configurations pass. `slurm/cell_summary.py` applies the same check to the final configuration of every one of the 6,110 runs; none fails.

## Regenerating

From the repository root:

```
python finalize_sweep.py --output-root supplement/runs ...      # run tables from the run files (slurm/finalize.sbatch)
python -m tsgo.experiments.export_configurations                # configurations/ and configurations.csv
python slurm/cell_summary.py                                    # tables/cell_summary.csv
python slurm/make_report.py                                     # REPORT.md
```

The export reads the run tables in `tables/` and the run files under `runs/`. It refuses to export a run whose file does not hold the radius its table row names. `slurm/report.sbatch` runs every step in order; `tsgo/paths.py` defines this layout for the code.
