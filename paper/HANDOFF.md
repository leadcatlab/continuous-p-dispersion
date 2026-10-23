# Continuous p-Dispersion — state of play

Written 2026-09-30, after the final sweep. Supersedes the 2026-09-28 version.
Companion to `REVISION_NOTES.md`, which tracks the manuscript item by item.

---

## 1. Where the results stand

`results/sweep.csv` is the **gold file**: one row per (container, metric, p,
seed), 2,190 rows. Written only by `finalize_sweep.py`, which refuses if any
cell is short of seeds, any key is duplicated, or the solver changed mid-sweep.

| | |
|---|---|
| runs | 2,190 = 219 cells x 10 seeds, no gaps, one commit (`712f9e3`) |
| starts | each run is 5 starts: its ladder rung, then 4 restarts at the best D so far (N = 4) |
| feasible | 2,190 (100%); every stored solution re-certified by an exact, cache-free check |
| containers | cube, tetrahedron, h-box, pointy-star, cage |
| metrics | EUCLIDEAN, TAXICAB, SUP_NORM |
| p grid | 1-10, 20, 30, 40, 50; tetrahedron also 35, 56, 84 |
| CPU | 143 CPU-h (the previous sweep took 243) |

Produced by `run_final.sh` (2026-09-29 18:48 to 2026-09-30 13:18). Rows live in
`results/data/`, solutions in `output/<container>/p<p>/<metric>/seed<n>.npz`,
each with a per-attempt trace (`attempt_D_init`, `attempt_D`, `attempt_E`,
`attempt_wall`).

The previous gold file, its parts and its solutions are archived in
`results/archive_2026-09-28/` and `output_archive_2026-09-28/`.

### Against the previous sweep

Better in 49 cells (up to +9.1%, cage l1 p=4), tied in 154, lower in 16.
Eight of the 16 are rounding-level (at most 0.001%). The other eight are cells
where the old best was a rare hit, found by 1-3 of its 50 seeds; the new median
is higher in seven of them and equal in the eighth:

| cell | old best | new best |
|---|---|---|
| cage l1 p=5 | 1.833333 | 1.750000 |
| pointy-star l-inf p=50 | 0.384615 | 0.373134 |
| cage l1 p=30 | 1.309524 | 1.289474 |
| tetrahedron l-inf p=40 | 0.057082 | 0.056630 |
| cage l1 p=20 | 1.393939 | 1.388889 |
| h-box l1 p=30 | 0.348623 | 0.347825 |
| tetrahedron l-inf p=50 | 0.054286 | 0.054204 |
| cage l2 p=20 | 1.129384 | 1.128749 |

More seeds per cell would catch these; 20 seeds would cost another ~143 CPU-h.

### Fallback and monotonicity

8 of 2,190 runs used the half-rung fallback: seven succeeded one half-rung
down, one after two steps. None ended infeasible.

D must not rise with p. Four flagged pairs, the largest 7e-7 relative:
numerical.

---

## 2. What was wrong, and is now fixed

Earlier defects (footpoint geometry, non-unique plane nearest point, E vs
E_and_grad, A5 active-wall tolerance, metric ball volume in D_init) are
described in git history and the manuscript. Found in the 2026-09-28 audit:

**Rounded cache keys.** Three caches keyed points by coordinates rounded to 6
decimals, so a point 5e-7 outside a wall inherited "inside" and a zero penalty
from a neighbour just inside. The optimiser found that pocket: 201 of the 250
"infeasible" seeds in the previous sweep ended with one point parked in it, and
2 rows marked feasible were not. Exact keys now.

**Cage normals.** Two cage faces ("Left/Right Partition Top Face") carried
outward "inward" normals. The active-face rule hid those walls from interior
points, so clearance was over-estimated by up to 3% of the container's extent;
3 stored cage solutions put a ball through a partition top. Tests now check
that every container's normals orient a closed surface consistently.

**Face area ignored holes.** Cage volume read 531.3 instead of 450, so its D_init
was 6% high.

**Reported D could exceed B(X)** by up to 5.7e-6 relative, since E < eps admits
a violation of up to sqrt(eps). Adjust Distance now returns exactly B(X).

**Tables** divided the relative error by our radius instead of the reference,
and rounded D instead of truncating it. **stats.render_figures** could draw an
infeasible solution. **Batch default density** was 0.35; the paper's value is
0.25. **Test suite** failed about one run in six from id-keyed cache reuse.

**Tabu search rarely ran.** With one start per run (N = 0, every sweep before
this one), the first LocalOpt reaches E < eps from any start at or below D-hat,
and tabu search then does zero iterations. The restarts at the best radius
(N = 4) are what engage it. A 12-cell pilot (`results/pilot/summary.md`)
chose this: better in 6 cells by 1.0-6.7% at a third of the CPU.

Tests: 134 pass.

---

## 3. The manuscript

Updated against the new gold file: setup (N = 4, 10 seeds, pilot), the ladder
and its half-rung fallback, a lemma bracketing D* for any norm and container
(with proof), Adjust Distance returning B(X), every figure caption and radius
quoted in the prose, and the runtime section (new exponents; the old small-p
slowdown is gone). Tables and figures regenerated; the PDF builds cleanly.

Three figures were repointed to cells in the grid: `ps_50`, `h-box_sup_50`,
`cage_sup_50` (the old p = 100 and p = 56 cells are no longer run).

The baseline comparison is written up in its own section (see section 4
below); the red `\needsdata` markers are gone.

`H2, journal class` remains won't-do: the author pastes into a C&OR template.

    make tables      # tsgo-stats + tsgo-tables -> paper/generated/
    make figures     # tsgo-figures -> paper/figures/
    make paper       # both, then latexmk

---

## 4. Benchmarking: done (2026-10-01)

Protocol fixed before any run: `paper/BASELINE_PROTOCOL.md`. Results in
`results/baseline/` (1,350 trial files, `analysis/summary.md`) and
`results/certificates/gurobi.csv`; written up in the paper's Section
"Comparison with General-Purpose Solvers" (label `sec:baselines`).

- **Quality (B)**: at equal wall time, best of 10, TSGO beats IPOPT in 33/45
  cells and SLSQP in 24/45; never behind by more than 2.6e-5 (all tetrahedron).
  Wilcoxon p = 3.4e-8 and 6.2e-4. Never behind on non-convex containers; up to
  2.9x IPOPT and 1.6x SLSQP.
- **Speed (A)**: *not* a TSGO win. Where all three reach 99.9% of best (8
  convex cells, p <= 20), IPOPT is 3-20x and SLSQP 4-71x faster. TSGO's case is
  reach: 11 cells only TSGO reaches, 2 only SLSQP. 38/450 TSGO trials had their
  first start still running at the budget (score 0).
- **Feasibility audit (E)**: 40.2% of SLSQP's converged solves over-state D by
  >1e-6, 0.92% are infeasible; IPOPT 17.0%, none by >1%.
- **Certificates (C)**: Gurobi (free restricted licence suffices) closes cube
  p <= 9 and tetrahedron p <= 10; TSGO within 1.5e-5 of every proven optimum
  except cube p = 7 (0.011% short). Kazakov p = 10 is 9.2% below the proven
  optimum. Time-limited Gurobi bounds are upper bounds only, never quoted.

Figures: `baseline_quality.png`, `baseline_anytime.png`,
`baseline_profile.png`, built by `python -m tsgo.experiments.baseline_plots`.

---

## 5. Things that will bite

**Rare hits need seeds.** The eight cells that lost ground against the previous
sweep all had an old best found by 1-3 of 50 seeds. Ten seeds of five starts
are more reliable per run but sample fewer basins.

**Solution files are keyed only by (container, p, metric, seed).** A job run
under a different configuration overwrites them. Experiments therefore write to
their own roots (`output_pilot/`, and `--output-root` on tsgo-batch);
`make_figures` also checks the stored D against the row it came from.

**Keep the Mac awake.** A sweep pauses when the machine sleeps. Run
`caffeinate -i -s -w <sweep pid>` alongside it; with the lid shut it also needs
power and an external display.

**Long runs need `screen`.** The harness's background jobs do not survive
session teardown:

    screen -dmS final bash -c './run_final.sh >> results/final.log 2>&1'
    screen -r final        # attach, ctrl-a d to detach

**A separate pass idles cores.** `run_final.sh` runs the tetrahedral extras
(p = 35, 56, 84) as a second pass of 9 cells, which took 6 of the sweep's 18.5
hours on 9 busy cores out of 11. Fold them into the first pass next time.

---

## 6. Commands

    .venv/bin/python finalize_sweep.py                          # validate + write gold file (10 seeds)
    .venv/bin/python finalize_sweep.py --check-only --allow-partial
    .venv/bin/python results/pilot/analyse.py                   # pilot summary
    .venv/bin/plot-results output/cube/p10/euclidean/seed8.npz   # 3-D viewer
    make tables figures                                          # rebuild from sweep.csv
    .venv/bin/python -m pytest src/tests -q --no-cov
