# 3D p-Dispersion of Spheres

Simulation code for distributing _p_ spheres in three-dimensional Euclidean space, maximizing a minimum–distance (\(p\)-dispersion) criterion and studying their motion under custom dynamics.

Implements the algorithm in *The Continuous p-Dispersion Problem in Three Dimensions*
(Manoj & Ornik). Section references below point at that manuscript.

---

## Notation

These conventions come from the paper's problem statement and optimization model. Code and
documentation should use them consistently.

### `D` is a **radius**, not a distance between points

This is the single most confusable symbol. The paper states it directly:

> Note that we will interpret the value of `D` to be the radius of the ball centered at each
> dispersion point `c_i` … and `D ∈ R_{>0}` as the *radius* of the solution.

So `D` is the radius of the sphere drawn around each point, and the optimization maximizes it.
The two constraints therefore use **different multiples of `D`**:

| Constraint | Requirement | Why |
|---|---|---|
| Dispersion (ii) | `d(c_i, c_j) ≥ 2D` for all `i ≠ j` | two spheres of radius `D` touch when their centers are `2D` apart |
| Boundary (iii)  | `d(c_i, ∂C) ≥ D` for all `i`     | one sphere of radius `D` touches the wall when its center is `D` away |

Equivalently: the open balls `B°_D(c_i)` must be pairwise disjoint and must not meet `∂C`.

The factor-of-two asymmetry between the two constraints is intentional. Reporting `D` as if it
were the minimum inter-point distance understates the configuration by exactly a factor of 2 —
the minimum inter-point distance is `2D`.

### `d` is the **metric**, not a distance-outside-the-container

`d` denotes the metric on `R³` that the whole problem is stated with respect to. The code selects
it via `--metric`:

| `--metric` | `d(x, y)` | Paper |
|---|---|---|
| `EUCLIDEAN` | `‖x − y‖₂` | ℓ₂ |
| `TAXICAB`   | `‖x − y‖₁` | ℓ₁ |
| `SUP_NORM`  | `‖x − y‖∞` | ℓ∞ |

`d` is overloaded to accept a set as its second argument: `d(c_i, ∂C)` is the point-to-boundary
distance, and in the boundary penalty `d(c_i, v)` is a point-to-segment distance when `v` is an
edge. It never means "how far outside the container a point has escaped" — that quantity is
unnamed in the paper.

### Symbol table

| Symbol | Meaning |
|---|---|
| `p` | number of dispersion points |
| `c_i` | the `i`-th dispersion point, `c_i ∈ R³` |
| `X` | a configuration, `X = {c_1, …, c_p}` |
| `D` | radius of the ball at each point — the quantity being maximized |
| `d` | the metric on `R³` |
| `C` | the polyhedral container; `∂C` its boundary |
| `V`, `E`, `F` | vertex, edge and face sets of `C` |
| `N_F` | inward-oriented normal of face `F` |
| `H^i` | set of active footpoints with respect to `c_i` |
| `B°_D(c_i)` | open ball of radius `D` centered at `c_i` |
| `ℓ_{i,j}` | dispersion penalty, `max{0, 2D − d(c_i, c_j)}` |
| `O_i` | boundary penalty for `c_i` |
| `E(X, D)` | energy / feasibility-residual; `E = 0` ⟺ `(X, D)` is feasible |
| `μ` | SUMT penalty factor in `AdjustDistance` |
| `ε` | feasibility tolerance (`--eps`) |
| `Q`, `β_max` | Tabu Search rearrangement count and no-improvement depth |

### Collisions in the manuscript's notation

Three symbols are reused for two different things. Code should not reproduce these; prefer the
disambiguated names given here.

| Symbol | Two meanings | Suggested code name |
|---|---|---|
| `p` | the number of points, **and** the origin point of a face normal in the Active Face definition | keep `p` for the count; call the other `face_origin` |
| `E` | the container's edge set (as in `V ∪ E ∪ H^i`), **and** the energy function `E(X, D)` | `container.edges` vs `energy` |
| `d` | the metric, **and** in places the generic letter for a distance value | `metric` vs `dist` |

### Relation to reported results

Tables report `D` — the radius. To compare against a source that reports minimum inter-point
distance, multiply by 2. Comparisons against Schaer and Packomania sphere-packing values are
radius-to-radius and need no conversion.

---

## Running

### 1. Clone the package

`git clone <package_name>`

### 2. Install the module inside a virtual environment

`python -m venv <venv_name>`

`source <venv_name>/bin/activate` on linux OR `.\<venv_name>\Scripts\activate` on Windows

`pip install -e .`

### 3. Run the program

`run-tsgo --p <p> --container <container_name> --D_init <initial radius> --plot <True/False> --save <True/False> --num_workers <n> --iterations <N> --parallel_iterations <P>`

Note `--D_init` is an initial **radius**, consistent with the convention above.
