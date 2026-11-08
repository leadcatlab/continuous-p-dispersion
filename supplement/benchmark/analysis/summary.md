# Baseline comparison: results

Trials read: 1350 of 1350. Scored by what each trial completed within its budget.

## B. Quality at equal wall time (best of 10, certified D)

| against | TSGO better | tied | TSGO worse | Wilcoxon p (two-sided) | median TSGO/baseline |
|---|---|---|---|---|---|
| ipopt | 30 | 15 | 0 | 5.7e-14 | 1.0190 |
| slsqp | 23 | 22 | 0 | 0.0009 | 1.0000 |

- convex (cube, tetrahedron): vs ipopt: 7 better / 11 tied / 0 worse; vs slsqp: 7 better / 11 tied / 0 worse
- non-convex (h-box, star, cage): vs ipopt: 23 better / 4 tied / 0 worse; vs slsqp: 16 better / 11 tied / 0 worse

## A. Time to target

| target | solver | cells where most trials reach it | median of per-cell median time (s) |
|---|---|---|---|
| 99.0% | tsgo | 45 of 45 | 42.5 |
| 99.0% | ipopt | 18 of 45 | 88.7 |
| 99.0% | slsqp | 25 of 45 | 27.7 |
| 99.9% | tsgo | 37 of 45 | 36.3 |
| 99.9% | ipopt | 13 of 45 | 20.0 |
| 99.9% | slsqp | 20 of 45 | 17.6 |

Performance profile at the 99.9% target: share of cells a solver reaches within tau times the fastest solver's median time.

| solver | tau=1 | tau=2 | tau=10 | ever |
|---|---|---|---|---|
| tsgo | 64% | 69% | 80% | 82% |
| ipopt | 0% | 4% | 20% | 29% |
| slsqp | 18% | 27% | 38% | 44% |

## E. Feasibility of reported radii (converged local solves)

| solver | converged solves | over-stated by >1e-6 | over-stated by >1% | certified radius zero | worst over-statement |
|---|---|---|---|---|---|
| ipopt | 259656 | 17.4% | 0.0% | 0.0% | 50.00% |
| slsqp | 2431797 | 41.8% | 1.0% | 1.1% | 100.00% |

