# Baseline comparison: protocol

Written 2026-09-30, **before any run of this experiment**. Its purpose is to fix
what is compared, how, and how the outcome is read, so that no choice below can
be tuned after seeing results. Any deviation is recorded in section 7 with its
reason and date.

## 1. Questions

- **A. Speed.** How quickly does each solver reach a given radius? (time to
  target, and best D against elapsed time)
- **B. Quality.** At equal wall time, which solver finds the larger radius?
- **C. Optimality.** Where optimality can be proved, how far is TSGO from the
  optimum?
- **E. Feasibility.** Is the radius each solver reports actually supported by
  its configuration?

## 2. Solvers

| name | what it solves | notes |
|---|---|---|
| TSGO | energy model, tabu search, SUMT | this paper's algorithm; configuration of the final sweep (ladder start, fallback), except that restarts at the best D continue until the time budget ends instead of stopping after 4 |
| IPOPT 3.14 | the constrained program below | interior point, via cyipopt; L-BFGS Hessian approximation; analytic Jacobians |
| SLSQP | the constrained program below | scipy; analytic Jacobians |

Both baselines are multi-start: from a random configuration drawn by the same
generator TSGO uses, solve to local optimality, record, repeat until the budget
ends. No baseline sees TSGO's energy, search or results.

### The constrained program

    maximise D  over  (c_1, ..., c_p, D)
    subject to  d(c_i, c_j) >= 2D           for all i < j        (dispersion)
                c_i in C with clearance >= D for all i           (containment)

Formulated as each solver would be given it by someone who knows the problem:

- **Dispersion, l2**: ||c_i - c_j||^2 >= 4 D^2 (smooth).
- **Dispersion, l1 and l-inf**: the norm is not differentiable. It is replaced
  by a smooth approximation (|t| ~ sqrt(t^2 + delta^2); max ~ log-sum-exp with
  temperature delta), delta = 1e-6 times the container's extent. Smoothing can
  over-state distances by O(delta); the certified radius below removes that.
- **Containment, convex containers (cube, tetrahedron)**: one linear constraint
  per face, n_f . c_i - b_f >= D ||n_f||_* (exact in every norm: ||.||_* is the
  dual norm).
- **Containment, non-convex containers (h-box, star, cage)**: a signed clearance
  oracle s(c_i) >= D, where s is the exact distance to the boundary inside the
  container and minus the distance to it outside, with its analytic
  (sub)gradient. The sign gives a solver a direction back when a point leaves
  the container. The oracle uses the same exact geometry routines that certify
  every reported radius; it is geometry, not search.

## 3. Cells, trials, budget

- **Cells**: containers {cube, tetrahedron, h-box, pointy-star, cage} x metrics
  {l2, l1, l-inf} x p in {10, 20, 40}: 45 cells.
- **Trials**: 10 per solver per cell, seeds 0-9. Seed s starts TSGO on ladder
  rung s mod 5, as in the sweep.
- **Budget**: every trial of every solver on a cell gets the same wall time,
  H = max(60 s, 2 x the median run time of TSGO on that cell in the final
  sweep). A local solve that would outlast the budget is stopped at the budget;
  the best radius found so far stands.
- **Hardware**: one core per trial, 11 trials at a time on the same machine
  (Apple M2 Pro), solvers interleaved so no solver systematically gets quieter
  cores.

## 4. What is recorded

For every trial: a trace of (elapsed seconds, best certified D so far), one
entry per completed start; final best certified D; number of starts; and, for
the baselines, the radius the solver itself reported at the end of each local
solve.

**Certified D** is B(X), the largest radius the configuration supports, computed
by the exact routine that certifies TSGO's results (max_feasible_D). Every
comparison uses certified D, for every solver.

## 5. Analysis

- **B**: per cell, best-of-10 and median-of-10 certified D per solver. Across
  cells: wins/ties/losses of TSGO against each baseline (tie: relative
  difference within 1e-6), and a Wilcoxon signed-rank test on the paired
  best-of-10 values.
- **A**: best D against elapsed time (median over trials, per cell). Time to
  target at 99% and 99.9% of the reference radius (the best certified D any
  method found on the cell, or the published value where one exists); a trial
  that never reaches it counts as not reached. Performance profiles over cells
  (fraction of cells in which a solver reaches the target within tau times the
  fastest solver's time).
- **C**: Gurobi spatial branch-and-bound on the constrained program for the
  convex containers, l2, for as many p as close their gap within 1 hour each.
  Reported: the proven upper bound, the gap to TSGO's best, and time.
- **E**: for each baseline local solve that terminated as converged, the
  reported D against certified D; share over-stated by more than 1e-6
  (relative), and the largest over-statement.

## 6. How the outcome is read

Fixed now, so that the paper states whatever these show:

- TSGO is "better" on a cell only if its best-of-10 certified D exceeds a
  baseline's by more than 1e-6 (relative); "faster" only if its median time to
  the 99.9% target is lower.
- Results are reported for every cell in the panel, including cells TSGO loses.
- If TSGO does not beat IPOPT on the convex containers, the paper says so and
  rests its claims on the non-convex containers, the certificates and
  generality.

## 7. Deviations

- 2026-09-30, before launch: trials run 10 at a time instead of 11, leaving
  one core for the Gurobi certificates (C), which run alongside. All three
  solvers share the same 10 slots, interleaved, so the change affects them
  equally.
- 2026-10-09, before launch: TSGO runs the configuration of the final 20-seed
  sweep (Q = max(3, round(ln p)), basin-hopping step 1.0 D at T = 0, final LP
  tightening), with restart patience off so restarts continue until the budget
  ends. Its trace records, after each start, the tightened radius of the best
  configuration so far: what the run would return if stopped there. The
  tightening takes under 0.1 s at p = 40, and its time is counted.
- 2026-10-09, before launch: budgets come from the final sweep
  (results/sweep.csv, 20 seeds per cell) rather than results/summary_final.csv.
- 2026-10-09, before launch: Gurobi is added as a fourth solver on the cells it
  can model, the l2 program on the convex containers (cube, tetrahedron; 6
  cells), started cold on one thread under the same budget, with its Seed
  parameter set to the trial seed. Every incumbent is scored by max_feasible_D,
  because Gurobi accepts violations within its feasibility tolerance and its own
  objective can overstate the radius. Its proven upper bound is traced too.
  The certificates of section C ran warm-started from TSGO's best and logged
  no incumbent over time, so they cannot serve as an anytime baseline.
- 2026-10-09, before launch: trials run 11 at a time, as first registered; the
  certificates no longer run alongside.
- 2026-10-09, relaunch: the first launch (2026-10-08 23:46) ran with the laptop
  lid closed, and the machine slept for most of the next two hours despite
  `caffeinate -i`. TSGO's time limit and trace ran on wall time, which counts
  sleep, while the baselines' budgets ran on perf_counter, which does not. The
  sleep was charged to TSGO alone: two finished TSGO trials had used 1494 s and
  1614 s of a 4091 s budget. Every trial was discarded, and every solver now
  keeps its budget on perf_counter: TSGO's run clock, IPOPT through its
  per-iteration callback, and Gurobi through its callback; the solvers' own
  wall-time limits remain only as backstops. The relaunch also holds
  `caffeinate -s`, which prevents system sleep on AC power.

