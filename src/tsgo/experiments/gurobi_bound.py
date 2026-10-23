"""Certified bounds for the convex containers via Gurobi spatial branch-and-bound.

The point is a *certificate*, not a race. CpDP is non-convex even when the
container is convex: the dispersion constraint ||c_i - c_j|| >= 2D excludes a
ball rather than including one, so it is reverse-convex and the feasible set is
disconnected with many local optima. A local solver (SLSQP, or TSGO itself)
returns one of them with no guarantee. Gurobi's spatial branch-and-bound
maintains a valid upper bound on D, so a run that closes its gap *proves* the
heuristic's answer optimal, and one that does not still brackets it.

Expect Gurobi to lose badly on time beyond p ~ 8-10. That is the expected and
reportable outcome: the exact method certifies optimality for small p and is
intractable past it.

    max  D
    s.t. n_f . c_i + b_f >= D          for every face f, every i   (linear)
         ||c_i - c_j||^2 >= 4 D^2      for every i < j             (nonconvex)
         c_i.x <= c_{i+1}.x            lexicographic tie-break     (symmetry)

Requires the optional dependency:  pip install -e ".[gurobi]"
"""

from itertools import combinations
from typing import Dict, List, Optional

import numpy as np

from tsgo.container.container import Container

CONVEX_CONTAINERS = ("cube", "tetrahedron")


def half_spaces(container: Container) -> tuple:
    """Inward unit normals and offsets with n . x + b >= 0 inside."""
    N = np.asarray([f.normal for f in container.faces], dtype=float)
    B = np.asarray([-float(np.dot(f.normal, f.vertices[0].as_array()))
                    for f in container.faces], dtype=float)
    return N, B


def is_convex(container: Container, tol: float = 1e-9) -> bool:
    """True when every vertex satisfies every face's inward half-space."""
    N, B = half_spaces(container)
    V = np.asarray([v.as_array() for v in container.vertices], dtype=float)
    return bool(np.all(N @ V.T + B[:, None] >= -tol))


def build(container: Container, p: int, verbose: bool = False):
    """The l2 program for a convex container as a Gurobi model: returns the
    model, the centre variables c[i, k] and the radius variable D."""
    # Check the model applies before requiring the optional dependency, so a
    # non-convex container reports the real reason rather than a missing import.
    if not is_convex(container):
        raise ValueError(
            f"{container.description} is not convex. The half-space formulation only "
            "describes convex containers; a non-convex one needs disjunctive/binary "
            "structure and is not attempted here."
        )

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:                       # optional dependency
        raise RuntimeError(
            "gurobipy is not installed. Install it with:  pip install -e \".[gurobi]\"\n"
            "A licence is required; the free size-limited licence covers only very small models."
        ) from exc

    N, B = half_spaces(container)
    V = np.asarray([v.as_array() for v in container.vertices], dtype=float)
    lo, hi = V.min(axis=0), V.max(axis=0)
    # D cannot exceed the inradius (p = 1 packs one ball at the incentre)
    centre = V.mean(axis=0)
    D_max = float(np.min(N @ centre + B))

    m = gp.Model("cpdp")
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.NonConvex = 2                 # spatial branch-and-bound

    c = m.addVars(p, 3, lb=[lo[k] for _ in range(p) for k in range(3)],
                  ub=[hi[k] for _ in range(p) for k in range(3)], name="c")
    D = m.addVar(lb=0.0, ub=D_max, name="D")
    m.setObjective(D, GRB.MAXIMIZE)

    for i in range(p):                                   # boundary constraint
        for f in range(N.shape[0]):
            m.addConstr(gp.quicksum(N[f, k] * c[i, k] for k in range(3)) + B[f] >= D,
                        name=f"bnd_{i}_{f}")

    for i, j in combinations(range(p), 2):               # dispersion constraint
        m.addConstr(
            gp.quicksum((c[i, k] - c[j, k]) * (c[i, k] - c[j, k]) for k in range(3))
            >= 4.0 * D * D, name=f"disp_{i}_{j}")

    for i in range(p - 1):                               # break the p! symmetry
        m.addConstr(c[i, 0] <= c[i + 1, 0], name=f"sym_{i}")

    return m, c, D


def solve(container: Container, p: int, time_limit: float = 600.0,
          mip_gap: float = 1e-6, threads: int = 0, warm_start: Optional[np.ndarray] = None,
          verbose: bool = False) -> Dict:
    """Maximise D exactly. Returns incumbent, bound, gap, time and status."""
    m, c, D = build(container, p, verbose)
    from gurobipy import GRB                 # build() has checked it is installed
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = mip_gap
    if threads:
        m.Params.Threads = threads

    if warm_start is not None:
        for i in range(min(p, len(warm_start))):
            for k in range(3):
                c[i, k].Start = float(warm_start[i, k])

    m.optimize()

    status = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit",
              GRB.INFEASIBLE: "infeasible", GRB.INTERRUPTED: "interrupted"}.get(m.Status, str(m.Status))
    # A model with no dispersion constraint (p = 1) is solved as a linear
    # program, which has no MIP gap; solved to optimality, its gap is zero.
    try:
        gap = float(m.MIPGap) if m.SolCount else float("nan")
    except AttributeError:
        gap = 0.0 if m.Status == GRB.OPTIMAL else float("nan")
    X = None
    if m.SolCount > 0:
        X = np.array([[c[i, k].X for k in range(3)] for i in range(p)])
    return {
        "p": p, "container": container.description, "status": status,
        "D_incumbent": float(D.X) if m.SolCount else float("nan"),
        "D_bound": float(m.ObjBound) if m.SolCount or m.Status == GRB.TIME_LIMIT else float("nan"),
        "gap": gap,
        "wall_s": float(m.Runtime), "X": X,
    }
