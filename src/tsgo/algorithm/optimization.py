import os
import time
from typing import Tuple

import numpy as np
from scipy.optimize import Bounds, basinhopping, minimize

# Analytic gradients (E_and_grad / phi_and_grad) replace scipy's finite
# differencing, which costs 3p+1 energy evaluations per gradient. Set
# CPDP_ANALYTIC_GRAD=0 to fall back to finite differences.
ANALYTIC_GRAD = os.environ.get("CPDP_ANALYTIC_GRAD", "1") != "0"

from tsgo.algorithm import energy_functions as ef
from tsgo.container.container import Container
from tsgo.utils.io import get_datetime

TIMES_CALLED = 10

# Basin hopping. Each hop displaces every coordinate by up to MBH_STEP_FACTOR * D
# and keeps the result only if the energy does not rise (T = 0): monotonic
# basin hopping, as the paper describes it. The previous call used scipy's
# defaults, stepsize 0.5 and T = 1. T = 1 against energies of 1e-9..1e-5
# accepted 99% of hops, a random walk keeping its best point; and an absolute
# step of 0.5 is half the unit cube but 5% of the 10-unit star and cage, so the
# same setting behaved differently in every container. A step in units of D
# scales with the problem. At p = 100 (l2, Q = 5, 15-min budget, 4 seeds) it
# tied the old setting on the cube (3/4 runs at Packomania each) and beat it on
# the pointy star in 4/4 paired seeds (median -0.010% vs -0.061% of the best
# radius found); steps of 0.1 D and 0.3 D were cheaper but reached the best
# basin less often. CPDP_MBH_LEGACY=1 restores the old call, to reproduce
# results made with it.
MBH_STEP_FACTOR = 1.0
MBH_T = 0.0
MBH_LEGACY = os.environ.get("CPDP_MBH_LEGACY", "0") != "0"

local_optimization_times_called = 0
local_optimization_time_total = 0.0

local_optimization_phi_times_called = 0
local_optimization_phi_time_total = 0.0

def local_optimization(
    X0: np.ndarray, 
    D0: float, 
    container: Container, 
    verbose: bool,
    method: str = 'L-BFGS-B', 
    metric: str = 'EUCLIDEAN',
) -> np.ndarray:
    """
    Perform local optimization on the given 3D points.

    Args:
        X0 (np.ndarray): Initial 3D points.
        D0 (float): Initial distance.
        container (Container): The container object defining the problem space.

    Returns:
        List[Point]: Locally optimized 3D points.
    """
    vprint = print if verbose else lambda *args, **kwargs: None
    global local_optimization_times_called, local_optimization_time_total
    
    start_time = time.time()
    local_optimization_times_called += 1
    X0 = X0.flatten()
    if ANALYTIC_GRAD:
        def function(X):
            e, g, _ = ef.E_and_grad(X.reshape(-1, 3), D0, container, metric=metric)
            return e, g
        result = minimize(fun=function, x0=X0, jac=True, method=method, options={'maxiter': 100, 'disp': False})
    else:
        function = lambda X: ef.E(X, D0, container, metric=metric)
        result = minimize(fun=function, x0=X0, method=method, options={'maxiter': 100, 'disp': False})
    optimized_points = result.x.reshape(-1, 3)

    local_optimization_time_total += time.time() - start_time
    
    if local_optimization_times_called % TIMES_CALLED == 0:
        avg_time = local_optimization_time_total / local_optimization_times_called
        vprint(f"\t[local_optimization] Avg. time {local_optimization_times_called * avg_time:.2f}s")
    
    return optimized_points

def local_optimization_phi(
    X0: np.ndarray, 
    D0: float, 
    container: Container, 
    mu: float,
    verbose: bool,
    method: str = 'L-BFGS-B', 
    metric: str = 'EUCLIDEAN'
) -> Tuple[np.ndarray, float]:
    """
    Perform local optimization using the phi function.

    Args:
        X0 (np.ndarray): Initial 3D points.
        D0 (float): Initial distance.
        container (Container): The container object defining the problem space.
        mu (float): Parameter for the phi function.

    Returns:
        Tuple[List[Point], float]: The optimized 3D points and the optimized distance.
    """
    vprint = print if verbose else lambda *args, **kwargs: None
    global local_optimization_phi_times_called, local_optimization_phi_time_total
    
    start_time = time.time()
    local_optimization_phi_times_called += 1
    if ANALYTIC_GRAD:
        def function(Z: np.ndarray):
            return ef.phi_and_grad(Z[:-1].reshape(-1, 3), Z[-1], mu, container, metric=metric)
    else:
        def function(Z: np.ndarray) -> float:
            D = Z[-1]
            return ef.phi(Z[:-1].reshape(-1, 3), D, mu, container, metric=metric)

    x0 = np.append(X0.flatten(), D0)

    # Set bounds: all coordinates unbounded, last variable (D) >= 0
    num_vars = x0.shape[0]
    lower_bounds = [-np.inf] * (num_vars - 1) + [0.0]
    upper_bounds = [np.inf] * num_vars
    bounds = Bounds(lower_bounds, upper_bounds)
    result = minimize(method=method, fun=function, x0=x0, bounds=bounds,
                      jac=True if ANALYTIC_GRAD else None)

    optimized_points = result.x[:-1].reshape(-1, 3)
    optimized_d = result.x[-1]
    local_optimization_phi_time_total += time.time() - start_time
    if local_optimization_phi_times_called % TIMES_CALLED == 0:
        avg_time = local_optimization_phi_time_total / local_optimization_phi_times_called
        vprint(f"\t[local_optimization_phi] Avg. time {local_optimization_phi_times_called * avg_time:.2f}s")

    return optimized_points, optimized_d

def monotonic_basin_hopping(X: np.ndarray, D: float, container: Container, iterations: int = 10, metric: str = 'EUCLIDEAN', seed=None) -> np.ndarray:
    """
    Function that performs monotonic basin hopping on the energy function E.

    Args:
        X (np.ndarray): A configuration of dispersion points.
        D (float): The minimum distance between points.
        container (Container): The container object defining the problem space.
        iterations (int, optional): Number of basin hopping iterations. Defaults to 10.
        metric (str, optional): Distance metric to use. Defaults to 'EUCLIDEAN'.
        seed (optional): Seed or Generator for basinhopping's random displacements.
            Pass one for reproducibility; None draws from numpy's global state.
    Returns:
        np.ndarray: The optimized configuration of points.
    """
    x0 = X.flatten()
    hop = {} if MBH_LEGACY else {"stepsize": MBH_STEP_FACTOR * D, "T": MBH_T}

    # Perform basin hopping to escape local minima
    if ANALYTIC_GRAD:
        def func(x):
            e, g, _ = ef.E_and_grad(x.reshape(-1, 3), D, container, metric=metric)
            return e, g
        result = basinhopping(func=func, x0=x0, niter=iterations, seed=seed,
                              minimizer_kwargs={"method": "L-BFGS-B", "jac": True}, **hop)
    else:
        result = basinhopping(func=lambda x: ef.E(x.reshape(-1, 3), D, container, metric=metric),
                              x0=x0, niter=iterations, seed=seed, **hop)

    return result.x.reshape(-1, 3)
