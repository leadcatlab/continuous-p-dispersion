"""Constrained-program baselines for the comparison in paper/BASELINE_PROTOCOL.md.

Each baseline is handed the continuous p-dispersion problem as a constrained
nonlinear program,

    maximise D  s.t.  d(c_i, c_j) >= 2D   (i < j)
                      clearance(c_i) >= D  (all i),

formulated as a practitioner would give it to a general-purpose solver, and
restarted from random configurations until its time budget ends. It sees none
of TSGO's energy model or search. Every radius is scored by max_feasible_D, the
exact routine that certifies TSGO's own results, so all solvers are compared on
certified radii.

Formulation (see the protocol, section 2):
  dispersion  l2: ||c_i - c_j||^2 - 4 D^2 >= 0.
              l1, l-inf: the norm with |t| ~ sqrt(t^2 + delta^2) and
              max ~ delta * log-sum-exp(. / delta), delta = 1e-6 x extent.
  containment convex containers: n_f . (c_i - P_f) - D ||n_f||_* >= 0 per face.
              non-convex: s(c_i) - D >= 0, s the signed clearance.
"""
import time
from typing import Dict, List

import numpy as np

import tsgo.algorithm.energy_functions as ef
from tsgo.container.face_batch import face_arrays
from tsgo.container.norms import dual_norm

CONVEX = ("cube", "tetrahedron")
SMOOTHING = 1e-6          # relative to the container's extent


class Program:
    """The constrained program for one (container, metric, p), with Jacobians."""

    def __init__(self, container, name: str, p: int, metric: str):
        self.C, self.name, self.p, self.metric = container, name, p, metric.upper()
        V = np.asarray([v.as_array() for v in container.vertices], dtype=float)
        self.lo, self.hi = V.min(axis=0), V.max(axis=0)
        self.extent = float(np.max(self.hi - self.lo))
        self.delta = SMOOTHING * self.extent
        self.iu, self.ju = np.triu_indices(p, 1)
        self.convex = name in CONVEX
        if self.convex:
            fa = face_arrays(container)
            self.N, self.P0 = fa["N"], fa["P0"]
            self.mn = np.array([dual_norm(n, self.metric) for n in self.N])
        self.n = 3 * p + 1
        self._cl_key, self._cl = None, None
        self.m = len(self.iu) + (p * len(self.N) if self.convex else p)

    # -- dispersion ----------------------------------------------------------
    def _pair_terms(self, X):
        diff = X[self.iu] - X[self.ju]                      # (P, 3)
        if self.metric == "EUCLIDEAN":
            return np.sum(diff * diff, axis=1), 2.0 * diff, "sq"
        if self.metric == "TAXICAB":
            r = np.sqrt(diff * diff + self.delta ** 2)
            return np.sum(r, axis=1), diff / r, "lin"
        # SUP_NORM: smooth max over the six values +-diff_k
        z = np.concatenate([diff, -diff], axis=1) / self.delta
        zmax = z.max(axis=1, keepdims=True)
        w = np.exp(z - zmax)
        val = self.delta * (np.log(w.sum(axis=1)) + zmax[:, 0])
        w /= w.sum(axis=1, keepdims=True)
        return val, w[:, :3] - w[:, 3:], "lin"

    def _clearances(self, X):
        """Signed clearance and gradient of every point, cached for the last X:
        solvers ask for the constraints and their Jacobian at the same iterate,
        and this is the expensive part of both."""
        key = X.tobytes()
        if key != self._cl_key:
            vals = [ef.signed_clearance_and_grad(X[i], self.C, self.metric) for i in range(self.p)]
            self._cl = (np.array([v for v, _ in vals]), np.array([g for _, g in vals]))
            self._cl_key = key
        return self._cl

    # -- constraints ---------------------------------------------------------
    def constraints(self, z: np.ndarray) -> np.ndarray:
        X, D = z[:-1].reshape(self.p, 3), float(z[-1])
        out = []
        if self.p > 1:
            val, _, kind = self._pair_terms(X)
            out.append(val - (4.0 * D * D if kind == "sq" else 2.0 * D))
        if self.convex:
            h = np.einsum("fj,ifj->if", self.N, X[:, None, :] - self.P0[None, :, :])
            out.append((h - D * self.mn[None, :]).ravel())
        else:
            out.append(self._clearances(X)[0] - D)
        return np.concatenate(out)

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        X, D = z[:-1].reshape(self.p, 3), float(z[-1])
        J = np.zeros((self.m, self.n))
        r = 0
        if self.p > 1:
            _, g, kind = self._pair_terms(X)
            P = len(self.iu)
            rows = np.arange(P)
            for k in range(3):
                J[rows, 3 * self.iu + k] = g[:, k]
                J[rows, 3 * self.ju + k] = -g[:, k]
            J[rows, -1] = -8.0 * D if kind == "sq" else -2.0
            r = P
        if self.convex:
            F = len(self.N)
            for i in range(self.p):
                J[r:r + F, 3 * i:3 * i + 3] = self.N
                J[r:r + F, -1] = -self.mn
                r += F
        else:
            G = self._clearances(X)[1]
            for i in range(self.p):
                J[r, 3 * i:3 * i + 3] = G[i]
                J[r, -1] = -1.0
                r += 1
        return J

    def bounds(self):
        lo = np.concatenate([np.tile(self.lo, self.p), [0.0]])
        hi = np.concatenate([np.tile(self.hi, self.p), [self.extent]])
        return lo, hi

    def start(self, rng) -> np.ndarray:
        X = self.C.generate_initial_solution(self.p, rng).reshape(self.p, 3)
        return np.append(X.ravel(), max(ef.max_feasible_D(X, self.C, self.metric), 1e-9))

    def certified(self, z: np.ndarray) -> float:
        return float(ef.max_feasible_D(z[:-1].reshape(self.p, 3), self.C, self.metric))


# -- local solvers ------------------------------------------------------------
def _ipopt_local(prog: Program, z0: np.ndarray, wall: float):
    import cyipopt
    # Stopped on perf_counter, the clock every solver's budget uses (it stops
    # while the machine sleeps); IPOPT's own max_wall_time is only a backstop.
    deadline = time.perf_counter() + float(wall)

    class _P:
        def intermediate(self, *args): return time.perf_counter() < deadline
        def objective(self, z): return -float(z[-1])
        def gradient(self, z):
            g = np.zeros_like(z); g[-1] = -1.0; return g
        def constraints(self, z): return prog.constraints(z)
        def jacobian(self, z): return prog.jacobian(z).ravel()

    lo, hi = prog.bounds()
    nlp = cyipopt.Problem(n=prog.n, m=prog.m, problem_obj=_P(), lb=lo, ub=hi,
                          cl=np.zeros(prog.m), cu=np.full(prog.m, 2e19))
    for k, v in (("print_level", 0), ("sb", "yes"), ("hessian_approximation", "limited-memory"),
                 ("tol", 1e-8), ("constr_viol_tol", 1e-8), ("max_iter", 3000),
                 ("max_wall_time", max(10 * float(wall), 1e-3))):
        nlp.add_option(k, v)
    z, info = nlp.solve(z0)
    ok = info["status"] in (0, 1)             # solved, or solved to acceptable level
                                              # (5: stopped at the deadline)
    return z, ok, int(info["status"])


def _slsqp_local(prog: Program, z0: np.ndarray, wall: float):
    from scipy.optimize import minimize
    lo, hi = prog.bounds()
    res = minimize(lambda z: -z[-1], z0, jac=lambda z: np.eye(prog.n)[-1] * -1.0,
                   method="SLSQP", bounds=list(zip(lo, hi)),
                   constraints=[{"type": "ineq", "fun": prog.constraints, "jac": prog.jacobian}],
                   options={"maxiter": 500, "ftol": 1e-12})
    return res.x, bool(res.success), int(res.status)


LOCAL = {"ipopt": _ipopt_local, "slsqp": _slsqp_local}


def run_trial(solver: str, container, name: str, p: int, metric: str, seed: int,
              budget: float) -> Dict:
    """Multi-start until `budget` seconds have passed. A local solve already
    under way when the budget ends is finished; the analysis counts only what
    completed within the budget (the trace carries the times)."""
    prog = Program(container, name, p, metric)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    best, trace, solves = 0.0, [], []
    while time.perf_counter() - t0 < budget:
        z0 = prog.start(rng)
        z, ok, status = LOCAL[solver](prog, z0, budget - (time.perf_counter() - t0))
        cert = prog.certified(z)
        best = max(best, cert)
        t = time.perf_counter() - t0
        trace.append((t, best))
        solves.append({"t": t, "reported_D": float(z[-1]), "certified_D": cert,
                       "converged": ok, "status": status})
    return {"trace": trace, "solves": solves, "best": best,
            "wall": time.perf_counter() - t0, "starts": len(solves)}
