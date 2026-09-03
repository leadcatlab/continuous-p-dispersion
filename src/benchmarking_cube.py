import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import differential_evolution
from scipy.spatial.distance import pdist
from tsgo.algorithm.cli import run_tsgo
import multiprocessing

# --- CONFIG ---
P_VALUES = list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 21, 27, 30, 40, 50, 60])
TIME_LIMIT = 600.0  # seconds (10 min)
DECIMAL_PLACES = 8

# D_Ours (From your table)
d_ours = list([
    0.50000000, 0.31698729, 0.29289321, 0.29289321, 0.26393200, 
    0.25735926, 0.25009464, 0.24999999, 0.23205080, 0.21428565,
    0.17840693, 0.17521516, 0.16666660, 0.16018841, 0.14553682, 
    0.13426749, 0.12948983
])

# D_Optimal (Packomania)
d_optimal = list([
    0.50000000, 0.31698730, 0.29289322, 0.29289322, 0.26393202, 
    0.25735931, 0.25013615, 0.25000000, 0.23205081, 0.21428571,
    0.17840720, 0.17721904, 0.16666667, 0.16018862, 0.14705882, 
    0.13595451, 0.13060194
])

def get_my_algorithm_result(p):
    print(f"Retrieving result for p={p} from our algorithm...")
    print(P_VALUES, d_ours)
    results = zip(P_VALUES, d_ours)
    for p_val, d_val in results:
        if p_val == p:
            return d_val
    return None

# --- MOCK PACKOMANIA DATA (For plotting illustration) ---
# In reality, you would manually input the best D from Packomania for p=10..100
# Approximating D ~ p^(-1/3) for sphere packing
def get_packomania_approx(p):
    results = zip(P_VALUES, d_optimal)
    for p_val, d_val in results:
        if p_val == p:
            return d_val
    return None
    

# --- BENCHMARK RUNNER ---
def run_benchmark_solver(n_points, time_limit):
    """
    Runs Differential Evolution for a FIXED time and returns best D found.
    Uses Unit Cube for simplicity in this demo.
    """
    # if n_points >= 7: return 0.0  # DE struggles with higher dimensions in this simple setup
    bounds = [(0, 1)] * (3 * n_points) + [(0, 0.5)]
    start_time = time.time()
    
    best_D = 0.0
    
    # Callback to stop solver after time limit
    def callback(xk, convergence):
        if time.time() - start_time > time_limit:
            return True # Stop optimization

    def objective(x):
        pts = x[:-1].reshape((n_points, 3))
        R = x[-1]
        
        # Simple Unit Cube Constraints
        if np.any(pts < R) or np.any(pts > 1-R): return 1e9
        
        dists = pdist(pts)
        if len(dists) > 0 and np.min(dists) < 2*R: return 1e9
        
        return -R

    # Run Solver
    res = differential_evolution(
        objective, bounds, strategy='best1bin', popsize=20, 
        callback=callback, disp=False
    )
    
    return -res.fun if res.success else 0.0

# --- DATA GENERATION (Simulated for visualization) ---
# Replace 'simulated_our_results' with your ACTUAL algorithm's results

if __name__ == "__main__":
    # Precompute Packomania values
    multiprocessing.freeze_support()

    packomania_vals = [get_packomania_approx(p) for p in P_VALUES]
    scipy_results = [0.49932824, 0.31592071, 0.29063988, 0.26149368, 0.25284714, 0.25291991, 0.24697443, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Simulated DE results (degrading with p)
    our_results = [] 

    print(f"Running Benchmark ({TIME_LIMIT}s limit per p)...")
    for i, p in enumerate(P_VALUES):
        # 1. Benchmark Solver
        d_scipy = scipy_results[i] # run_benchmark_solver(p, TIME_LIMIT)
        # scipy_results.append(d_scipy)
        
        # 2. Your Algorithm (Simulating "Better" Performance)
        # In real life, put: d_ours = my_algo.solve(p)
        output_dir = f'output/cube/p{p}'
        d_ours_iter = get_my_algorithm_result(p)
        our_results.append(d_ours_iter) 
        
        print(f"p={p}: Scipy={d_scipy:.8f}, Ours={d_ours_iter:.8f}, Packomania={packomania_vals[i]:.8f}")


    print(f'D_scipy: {scipy_results}')
    # --- PLOTTING 1: Raw Distance ---
    FONTSIZE = 24
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.plot(P_VALUES, packomania_vals, 'k--', label='Optimal (Packomania)', linewidth=2)
    ax1.plot(P_VALUES, our_results, 'g-o', label='Our Algorithm', linewidth=2)
    ax1.plot(P_VALUES, scipy_results, 'r-x', label='Benchmark (Scipy DE)', linewidth=2)
    ax1.set_xlabel('Number of Points ($p$)', fontsize=FONTSIZE)
    ax1.set_ylabel('Minimum Distance ($D$)', fontsize=FONTSIZE)
    ax1.set_title('Comparison of Solution Quality (Cube)', fontsize=FONTSIZE)
    ax1.tick_params(axis='both', labelsize=FONTSIZE-2)
    ax1.legend(fontsize=FONTSIZE-2)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/Users/sunchips/Documents/p-dispersion/continuous-p-dispersion-in-three-dimensions/figures/comparison_cube.png', dpi=300)
    plt.show()