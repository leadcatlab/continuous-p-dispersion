# Revision round 2: what to check in the review copy

File: `paper/continuous-p-dispersion-in-three-dimensions-review-copy.pdf` (rebuild with `make review`).
Colors: **orange** = a decision for you, **yellow** = a quoted result, **magenta** = a note, blue text = draft prose.
Pages are those of the review copy. Every number below was recomputed from the cluster suite
(commit a0cc61c; `supplement/`); the source to check it against is given in brackets.

## A. Decisions (orange)

- [ ] **p13, Escape phase.** The escape phase is now described as part of every run, replacing
      "a diagnosis, not part of the method". Is that how you want to present it?
- [ ] **p13, D_init ladder.** The example of a low start trapping the search (cube, l-inf, p = 20)
      is from runs without the escape phase; with it, all 20 seeds reach 1/6. Keep the old example
      beside the new result, replace it, or cut it?
- [ ] **p14, Cube l-inf tilings.** "Without the escape phase these tilings were rarer: 8 and 2 of 20
      runs." Keep the comparison with the earlier sweep or drop it?
- [ ] **p17-18, H-box l-inf p = 56.** Now "all 20 runs find it"; the old "0 of 60 with the defaults"
      is kept as the earlier-sweep comparison, plus "this cell is outside the grid of Table 3". Keep?
- [ ] **p19, Cage.** "Without the escape phase, 75 of 900 runs stopped at 1/2" (earlier sweep) and
      "that cell is limited by time rather than by the trap". Keep?
- [ ] **p20, Stoyan & Yaskov example 3.** 88 spheres comes from 32 further seeds (1 of 32 fits),
      declared in the text and a table footnote. Acceptable as an extension of the experiment?
- [ ] **p21-22, Example 6.** "Without the escape phase, eight of the 10 seeds stopped at exactly
      this radius" (earlier sweep). Keep?
- [ ] **p21, Benchmark.** New sentence: this round departs from the pre-registered protocol in three
      recorded ways (escape phase, budgets from the cluster sweep, cluster hardware). Wording OK?
- [ ] **p25, Conclusion.** Rewritten escape-phase paragraph ("now part of every run ... not tuned").

## B. Results to spot-check (yellow, changed this round)

- [ ] **p1, Abstract.** 30 and 23 of 45 cells, factor up to 2.4; more spheres than published in
      seven domains "by up to eight". [supplement/benchmark/analysis/summary.md; Table 9]
- [ ] **p12, Setup.** Hardware: one core of a Campus Cluster node (AMD EPYC 7713). 371 configurations
      verified. [supplement/configurations.csv]
- [ ] **p13-14, Ladder.** All 20 seeds reach 1/6 (cube l-inf p = 20); fallback needed by 36 of 4,800
      runs (33 half a rung, 3 two steps). Ratios 0.77-1.59 and 0.80-1.59 D-hat unchanged.
      [supplement/tables/cell_summary.csv, fallback_runs]
- [ ] **p14-15, Cube.** Within 2.3e-8 of Packomania for p <= 40; short by 5.6e-7 (p = 50) and 6.2e-6
      (p = 64); p = 100 matches. 3x3x3 and 4x4x4 tilings: 20 of 20 runs each. [Table 1]
- [ ] **p15-17, Tetrahedron.** Unchanged claims re-verified: min improvement over Kazakov ~0.8%,
      > 10% at p = 10; D_n matched at every tetrahedral number.
- [ ] **p17-18, H-box p = 56.** All 20 runs at 1/4. [supplement/tables/hbox_sup_56.csv]
- [ ] **p18, Star p = 100.** 24 in the core, 12-13 per spike, outermost at 3.5 of 5 (re-verified,
      unchanged); caption D = 0.36214024.
- [ ] **p19, Cage.** 15 runs stop at 1/2, all l-inf at p = 50 and 100, none under l2; p = 100 l2
      runs reach 0.586-0.741; caption D = 0.74080228 (corrected geometry, 58 faces).
- [ ] **p20, Table 9 (Stoyan & Yaskov).** Ours: 97, 95, 88+, 82, 79, 31, 59, 81; D at theirs and
      first-fit minutes updated; 66 of 80 runs reach radius one at their counts, first within 9 min.
      [supplement/tables/stoyan_yaskov*.csv; `python -m tsgo.experiments.stoyan_yaskov_plot` prints the table]
- [ ] **p20, Gains.** Largest in example 8 (eight spheres), then examples 1 and 2 (four each).
- [ ] **p21-22, Example 6.** No run ends at 0.93364; medians 0.978-0.997 at counts 30-34; three seeds
      fit 30, two fit 31, none 32.
- [ ] **p21, l1 / l-inf.** Best radius 1.17-1.35 (l1, every count fits), 0.84-0.89 (l-inf, none fits).
- [ ] **p22, Quality.** 30 / 23 of 45; Wilcoxon 1.1e-13 / 9.0e-4; convex margin at most 6.8%;
      up to 2.4x IPOPT and 1.6x SLSQP, both cage l1 p = 40.
- [ ] **p22, Speed.** 12 cells where all three reach 99.9%: SLSQP sooner in 7 (up to 16x), IPOPT in 6
      (up to 2.3x), ours first in 5; no TSGO trial scored zero; only ours reaches in 16 cells, none in 8;
      SLSQP first in 8, reaches in 20 (12 after ours); ours first in 29, 37 in all; IPOPT reaches 13.
      [supplement/benchmark/analysis/A_time_to_target.csv]
- [ ] **p22, Feasibility audit.** SLSQP 42.2% / 1.1% / 1.1%; IPOPT 17.7% and under 0.01%.
- [ ] **p22, Optimality.** Within 1.2e-5 of Gurobi's bound; Gurobi in the comparison behind by
      0.7% and 3.1% (cube p = 20, 40), 7.0% and 3.9% (tetrahedron). [supplement/certificates/gurobi.csv]
- [ ] **p23-24, Runtime.** Rewritten for the new time limits: 1,114 of 4,800 runs over 30 minutes;
      exponents 1.19 / 1.06 / 1.00 / 0.66 (cube / H-box / tetrahedron / star), cage 0.29; tetrahedron
      means 557-1,063 s at 35, 56, 84 and 2,271 s at 40; slowest cell cage l-inf p = 100, 4,356 s.
      The interpretation changed (the limits now cap the curves below O(p^2)); read it as text too.

## C. Notes and figures

- [ ] **Magenta "design experiment on earlier code; not rerun"** (p13, also p9 and p18): the pilot,
      the restart-patience replay, the choice of Q, the basin-hopping comparison. Keep as is, rerun,
      or cut?
- [ ] **Magenta "earlier sweep, escape phase off"** (p13-22): every place a number from the earlier
      sweep is kept for comparison; tied to the decisions in A.
- [ ] **p22, Fig. "Example 6" (stoyan_yaskov_ex6)** is still the old before/after figure from the
      continuation runs (magenta "redraw"). Redraw from the new runs, or drop?
- [ ] **Regenerated figures** to look over: cube/tetrahedron comparison (p15-16, axes fixed), cube
      l-inf 64, tetrahedron 10 and 35, H-box 56, star 100, cage 100 (l2, l-inf), runtime (p23),
      benchmark quality / anytime / reach (p22-23), Stoyan & Yaskov summary and packings (p21-22).
- [ ] **Bibliography:** `horvat2012` has no journal or venue.
