# Manuscript inconsistencies

**Status: every item is resolved except H2 (journal class), which is a
structural decision, and the results narrative, which is waiting on the
non-Euclidean regeneration.** Entries below are kept as a record of what was
wrong and what was done about it; the original line numbers referred to the
80-column wrapping at commit `35b32b7` and have long since drifted, so re-grep
the quoted text.

Two entries did not survive contact with the evidence and are marked as such:
**A2(ii)** was overstated — the sum does not endanger a.e. differentiability —
and **C2**'s 163x speedup figure did not reproduce, measuring 185x under a
stated method and 20x at $p=10$.

Three defects were found while fixing the listed ones and are recorded here as
A3a, A3b and A5.

Severity, as originally assigned: **A** = states something false; **B** = states
something the code no longer does; **C** = omission; **D** =
correctness-preserving cleanup.

---

## A. Mathematical errors

### A1 — Rademacher hypothesis **[resolved]**

Replaced by Lemma `lem:lipschitz`, which separates the two steps the original
phrasing conflated: (a) a norm-induced metric is Lipschitz *with respect to the
Euclidean structure*, by equivalence of norms, which is what Rademacher needs
for "almost everywhere" to mean Lebesgue-a.e.; and (d) the squared hinge is
$C^1$, so the chain rule carries differentiability through. A1a, A1b and A1c are
all covered, and the paper restricts to norm-induced metrics throughout (F8).

### A2 — a.e. differentiability of $O_i$ **[resolved; note (ii) was overstated]**

**A2(i) resolved.** The manuscript now carries the argument: on the open set
where the active sets $H^1,\dots,H^p$ are locally constant, $E$ is a finite sum
of locally Lipschitz terms, so Rademacher applies; the excluded configurations
(a footpoint meeting a face boundary) form a finite union of codimension-one
sets, hence have measure zero.

**A2(ii) was wrong as written.** The note claimed "no amount of additional
argument rescues line 499 while the sum stands". That does not follow. A jump
confined to a measure-zero set defeats *global* Lipschitzness, but Rademacher is
being applied locally, on the open complement of that set — where $E$ genuinely
is Lipschitz. The sum therefore supports a.e. differentiability perfectly well,
and B1 is a real improvement rather than a prerequisite. Keeping `sum` costs no
correctness, which is why the default is unchanged.

### A3 — the footpoint is Euclidean, but the paper claims any metric **[resolved]**

Resolved by **option 2**: the construction was generalised rather than the claim
narrowed. Code in `2437912`, tie-break and path-divergence fixes in `d7ccb4c`,
manuscript in the same series.

`Face.project_point` returned the orthogonal foot regardless of metric, and the
face term in `E_and_grad` computed `|h| * ||N||` instead of `|h| / ||N||_*`.
Both over-estimated the clearance to a slanted wall — the unsafe direction, in
which a ball crosses a wall while `E` still reads zero. Measured against
dense-sampled ground truth, the over-estimate reached **+32%** on the
tetrahedron under the sup norm, and **40/40** stored tetrahedron solutions in
both non-Euclidean metrics were infeasible.

The fix is `norms.plane_nearest` and `face_batch.active_feet`, which use
`d(x, P) = |h| / ||N||_*` with the minimiser found among the vertices of the
metric's unit ball. Axis-aligned containers were never affected in *distance* —
there the face normal is an axis direction — though they were affected in
*membership*, see the tie-break note below.

Manuscript: the Active Footpoint definition is now the metric-nearest point,
Equation `eq:plane` states the formula with its proof, and
§`sec:polyhedral` develops the dual-generator construction (C1).

### A3a — the nearest point of a plane is not unique **[resolved]**

Found while fixing A3. Under a polyhedral norm the optimal directions
`U* = {u : ||u|| <= 1, <N,u> = ||N||_*}` form a *face* of the unit ball, not a
point. For the cube's wall `x = -1/2` under the sup norm, four of the eight ball
vertices tie. Every choice gives the same distance but a different position on
the plane — and position is what decides whether the foot lands on the face,
hence whether the face contributes at all. Selecting by `argmax` made the energy
depend on the order the ball's vertices happen to be listed in.

`active_feet` searches the tied set rather than picking from it, trying the
centroid of the tied vertices first: it is in the tied set, it is the most
interior choice, and for an axis-aligned normal it *is* the orthogonal
direction. The manuscript states the non-uniqueness explicitly and makes the
definition existential ("some foot lies on F"), which is what Lemma 2 needs.

### A3b — E and E_and_grad were different functions **[resolved]**

Also found while fixing A3, and the more serious of the two. After `2437912`,
`E` tested the *metric* foot for face membership while `E_and_grad` tested the
*Euclidean* foot and scored it at the *metric* distance. The two disagreed by up
to **2e-2** under the sup norm on every container tested — on the gradient side,
which is the function the optimiser actually descends, while feasibility was
reported from the other one.

Both now call `face_batch.active_feet`. All 15 container × metric cells agree to
machine precision, pinned by
`src/tests/test_container/test_active_feet.py::test_energy_and_gradient_paths_agree`.

### A4 — "$O_i$ is zero iff $c_i \in C$ and $d(c_i,\partial C) \ge D$" **[resolved]**

Follows A3 option 2, as predicted: the covering claim now holds for every
norm-induced metric, and `lem:zero` carries its proof rather than asserting it.
The proof has two steps that were previously missing. First, the face carrying
the nearest boundary point is *active*: if it were not, $c_i$ would lie strictly
on the outward side of its plane, so the segment from that point to $c_i$ would
leave $C$ and re-enter, meeting $\partial C$ strictly nearer to $c_i$ — a
contradiction. Second, either some foot of $c_i$ on that face's plane lies on
the face, in which case it is in $H^i$ and realises the distance, or none does,
in which case the nearest point of the face lies on its relative boundary, a
union of edges, all of which are in $\mathcal{E}$.

### A5 — a point's own wall could drop out of its active set **[resolved]**

Found while auditing the stored solutions for A3. `contains()` admits a point
within `EPS_PLANE = 1e-8` of a face as inside the container, but the active-face
test demanded `h >= 0` strictly. A point a few 1e-9 *outside* a wall was
therefore inside by one rule and had its own wall dropped by the other: its
clearance fell back to a distant edge and a ball resting on the floor scored
`E ~ 1e-16`, a perfect feasibility report. Observed in
`output/cube/p3/sup_norm/seed27.npz`, where a point sits 6.6e-9 below
$z = -1/2$; recomputed after the fix that configuration scores exactly $D^2$.

Independent of metric and of face orientation. Fixed in `d7ccb4c`: the two rules
now read one shared constant, `face.EPS_PLANE`, and the inside branch is
inclusive to it. Only the inside branch — a dropped face there removes a term
from a penalty whose floor is zero, which manufactures feasibility, whereas
outside the penalty is bounded below by $D^2$ and a dropped face can only
over-penalise.

---

---

## B. Paper describes what the code no longer does

### B1 — boundary penalty aggregation: sum vs min **[decided: keep `sum`]**

Kept as `sum`, so no regeneration on this account and `CPDP_BOUNDARY_MODE`
stays at its current default. See A2: the sum does not endanger the
differentiability claim, so the choice is empirical rather than forced.

The A/B data (`results/ab_test`, Euclidean, five seeds across four
container-$p$ cells, unaffected by the footpoint fixes) shows the two trading
off rather than one dominating. Best radius under `min` is 0.975-1.000x that
under `sum`; but `min` is the more reliable per seed, lifting the *mean* radius
from 0.359 to 0.388 on h-box $p=10$ and 0.106 to 0.112 on tetrahedron $p=10$.
The manuscript now states both halves instead of the bare "slightly weaker
best-case packings". Worth revisiting with more seeds if reliability matters
more than the top of the distribution.

### B2 — exterior branch does not match the implementation **[resolved]**

Equation `eq:O_i` carries $(D + 2 d)^2$, and the justification at the old line
460 is replaced by the real reason: the term must be positive *and* have
non-vanishing gradient at the wall, or an escaped point is penalised without
being pulled back.

### B3 — exterior branch omits edges **[resolved]**

Justified rather than changed. Outside the container the penalty has no zero to
reach, so its job is to be positive and point inward, not to measure the
boundary distance exactly; $V \cup H^i$ already fixes a direction of return.
Inside, the edges are indispensable — they cover the case where no foot lands on
its face, which is the branch the proof of Lemma 2 now leans on.

### B4 — $\mu$ schedule **[resolved]**

Algorithm 3 now reads $\mu_0 = 1$ with $\mu \gets 10\mu$ at the top of the loop,
so the first `LocalOptPhi` runs at $\mu = 10$ — matching `tsgo_algorithm.py`
exactly — and $K = 15$ is stated. The "at most 100" sentence is replaced by why
a large $\mu_0$ ill-conditions the first gradient.

### B5 — ray directions are drawn once per run **[resolved]**

Grammar fixed and the reason stated: re-drawing per evaluation would make $E$ a
random function of $X$, so gradient descent would be minimising a stochastic
objective. Fixing them per run keeps $E$ deterministic while avoiding a single
alignment biasing every run.

### B6 — Algorithm 3 has no feasibility fallback **[resolved]**

The bound is now a named quantity, $B(X)$ in Equation `eq:bound`, with the
monotonicity argument that makes the feasible radii exactly $[0, B(X)]$. Both
guards are in the pseudocode: snap down when SUMT ends infeasible, raise up when
it ends conservative. The $D = 0$ stationary point and the H-box $p=1$ collapse
are explained in the text.

---

## C. Missing content

### C1 — the polyhedral-norm construction is a contribution **[resolved]**

Written as §`sec:polyhedral`, "Distances under a polyhedral norm", placed before
the Active Face and Active Footpoint definitions because they now depend on it.
It covers the convexity of $t \mapsto \|(x-s) - t(e-s)\|$ for *every* norm, the
upper-envelope reduction and its finite candidate set for a polyhedral one, the
endpoint/crossing gradient (Danskin and implicit differentiation), and the
plane distance $|h|/\|N\|_*$ with a proof and the non-uniqueness discussion.

The subsection gained weight from A3: the dual-generator argument is one idea
used three times — segment distance, its gradient, and face projection — rather
than the two the original note anticipated.

### C2 — analytic gradient is not described **[resolved]**

A paragraph in §Optimization Model gives the closed forms for each term and the
reason it matters. **The note's 163x figure did not reproduce.** Re-measured on
the unit cube against central differences: 20x at $p=10$ and 185x at $p=60$.
The claim that removing finite differences took the tetrahedron $p=16$
$D_{\mathrm{std}}$ from 9.5e-4 to 0 was dropped — the Euclidean tetrahedron grid
has no $p=16$ cell, so nothing on record supports it.

### C3 — no statement of what is reported **[resolved]**

Stated once, before Equation `eq:E`: $D$ is the radius, minimum pairwise
distance is $2D$, the asymmetry between constraints (ii) and (iii) is
deliberate, and Packomania comparisons are radius to radius.

---

## D. Notation collisions

### D1 — $p$ is both the point count and a face's base point **[resolved]**

The face's base point is now $q_F$, in the Active Face definition and in every
later reference. $p$ is the dispersion-point count only.

### D2 — $E$ is both the edge set and the energy **[resolved]**

Edge set renamed $\mathcal{E}$; $C = (V, \mathcal{E}, \mathcal{F})$.

### D3 — $F$ is both the face set and a single face **[resolved]**

Face set renamed $\mathcal{F}$; $F$ is a single face.

### D4 — $\epsilon$ vs $\varepsilon$ **[resolved]**

`\varepsilon` no longer appears; one glyph throughout.

---

## E. Pseudocode defects

### E1-E6 — pseudocode defects **[resolved]**

E1 the brace, E2 the wrong update variables, E3 $D$ used before definition, E4
$\epsilon$ missing from `\Require`, E5 the unequal first comparison and possibly
unset `BestMove`, E6 the tabu test paid for after `LocalOpt` — all corrected in
Algorithms 2 and 4.

### E7 — MBH placement **[resolved]**

The prose now says MBH sits inside the Tabu Search loop, cross-references the
line, and says why the two compose there: tabu memory stops a move being
re-taken, MBH perturbs whatever that move produced.

---

## F. Wording and precision

### F1-F10 — wording and precision **[resolved]**

F1 squared *shortfalls* not distances; F2 `lem:zero` is a numbered lemma with a
proof; F3 the $\mu_0$ guidance is replaced by the value used and why; F4 the
`LocalOpt` claim is corrected — the code does call it, on the infeasible branch;
F5 the truncated sentence; F6 the straight quotes; F7 one non-convex feasible
set with many local optima; F8 "any norm-induced metric" throughout; F9 Kazakov
no longer cited for a non-convex claim it cannot support; F10
"almost-everywhere differentiable" in the conclusion.

---

## G. Methodology statements sitting in §Results (factual, not narrative)

### G1-G3 — methodology statements in §Results **[resolved]**

G1: $\rho_{\text{init}} = 0.25$, matching the sweep; the inradius cap is now in
Equation `eq:dinit_of_rho` with the reason (1.5x over-estimate on the H-box,
2.1x on the cage, and no $D$ above the inradius is feasible at any $p$); the
small-$p$ special case never existed and is gone.

G2: the "1/3/5 global iterations, $\beta$ 5/10/15" schedule was never run. Every
row of `results/sweep.csv` records $N = 1$, $\beta = 10$, $Q = 3$, 50 seeds. The
text now states those, and that multi-start diversity comes from the seeds.

G3: the DE baseline is replaced by multi-start L-BFGS-B and SLSQP, described
with its actual budget — 10x our wall time on the same cell, capped at 600 s,
median 452 and 152 restarts respectively, five seeds.

---

## H. Formatting and submission

### H1, H3, H4, H5 — preamble **[resolved]**

`showkeys` removed; duplicate `amssymb`/`amsmath`/`algpseudocode` loads removed;
the two unused `\mathscr` macros removed rather than pulling in `mathrsfs` for
nothing; the silent `\def \phi {\varphi}` removed.

### H2 — wrong journal class (line 1–3)

`revtex4-1` with the `prd` (Physical Review D) option, targeting *Journal of
Global Optimization*. JOGO is Springer — needs `svjour3` and the Springer
bibliography style. This will change the layout substantially, so do it early
rather than after the figures are tuned.

---

## I. Second audit of the non-experimental sections

A line-by-line check of §Introduction through §Global Optimization Algorithm
against the code, after the items above were closed. Everything here is now
fixed in the manuscript except I8.

### I1 — the ray count and its justification were unstated **[resolved]**

The text said "several rays". The code uses $k = 8$, and has a real argument
for it: the ambiguous directions occupy a band of width $O(\epsilon/r)$ about a
measure-zero arc, so a $k$-ray majority fails with probability $O((4\pi)^{k/2})$
and the edge count enters only inside a logarithm, giving $k = O(\log
|\mathcal{E}|)$. That is now in the paper — it turns an arbitrary-looking
constant into a bound.

### I2 — the on-boundary rule was undocumented **[resolved]**

`contains()` classifies a point within `EPS_PLANE` of any face as interior
*before casting any ray*, and the manuscript never said so. This is not a
detail: the ray bound degrades as $r \to 0$, which is exactly where converged
dispersion points sit, so the boundary case is handled exactly rather than
probabilistically. It is also the rule the Active Face test must agree with —
the disagreement between the two was A5. Both are now stated together, with the
consequence of letting them drift spelled out.

### I3 — the tabu list description was missing its mechanism **[resolved]**

The paper described a list of "recent moves" evicted by size. The code stores
`(point index, quantized destination cell)` with a grid of side $\delta = D/20$,
a tenure of 7 iterations, a maximum size of 20, and the reverse of every
accepted move. The quantization is load-bearing — a continuous position never
recurs exactly, so a list keyed on positions would forbid nothing — and none of
it was written down.

### I4 — the aspiration criterion was absent **[resolved]**

The code takes a forbidden move anyway when it beats the incumbent best
(`aspired = trial_energy < best_energy`). Neither the prose nor Algorithm 2 had
it. Both do now.

### I5 — E6 was resolved in the wrong direction **[corrected]**

The earlier note asked that the tabu test be moved *before* `LocalOpt` so
forbidden moves are not paid for, and the pseudocode was changed accordingly.
That contradicts the code, and the code is right: aspiration needs the trial
energy, which needs `LocalOpt` to have run. The order cannot be swapped without
losing aspiration. Algorithm 2 now matches the implementation.

### I6 — Algorithm 2's neighbourhood initialisation contradicted the code **[resolved]**

The E5 fix seeded the incumbent with `LocalOpt(MovePoint(X, p_1, q_1))`
unconditionally. The code seeds it with $X$ at energy $\infty$ and a null move,
so an inadmissible $(p_1, q_1)$ is not silently adopted, and skips the tabu
update entirely when no admissible neighbour exists. Fixed to match.

### I7 — MBH's acceptance rule was misdescribed **[resolved]**

The paper said the new local minimum "is accepted only if it offers a strict
improvement". The code calls `scipy.optimize.basinhopping` with ten hops, whose
walk accepts uphill minima with the usual Metropolis probability; what it
returns is the lowest minimum found. The net step is non-worsening because the
input is already locally optimized, but the stated mechanism was wrong.

### I8 — Drezner and Erkut is cited under higher dimensions **[open, for the authors]**

The sentence "In higher dimensions, work is severely limited: \citet{drezner1995}
formulated a nonlinear model restricted to the unit square" places a
two-dimensional result inside the higher-dimensional paragraph. Either it
belongs with the planar work above it, or the intended container was the unit
cube. Left alone: it is a claim about the literature, not about this code.

### I9 — smaller corrections **[resolved]**

Constraint (iii) said it "keeps points away from the edges" when it guards the
whole of $\partial C$. A configuration was called a *set* while every sum over
it is indexed, so it is now a tuple. Equation `eq:vacancy_energy` summed over
$i$ while its summand used $j$.

---

## What is left

1. **H2 — journal class.** `revtex4-1`/`prd` targeting *Journal of Global
   Optimization*, which is Springer and wants `svjour3` with the Springer
   bibliography style. It will move every float and reflow the two-column
   layout, so it is worth doing before the figures are tuned — but it is a
   decision about where to submit as much as a formatting fix, so it is left
   open deliberately.

2. **The results narrative.** §Results for the tetrahedron, the star and the
   sup-norm cube quotes radii from runs that the footpoint fixes invalidated.
   Those cells are being regenerated; the tables and figures rebuild from
   `results/sweep.csv` by `make tables figures`, but the prose around them is
   hand-written and will need revising, downward, once the new numbers land.

3. **The cage under the sup norm.** `rvc_100_sup` claims $D = 0.4999$ at
   $p = 100$, but no cage non-Euclidean runs exist and never did. The figure
   manifest already carries it as `pending` so the build cannot forget it.
   Either run the cell or drop the figure and its sentence.
