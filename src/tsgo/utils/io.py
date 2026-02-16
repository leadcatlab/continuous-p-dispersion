import os
import pickle
import shutil
from datetime import datetime
from typing import List, Union

import numpy as np

from tsgo.container.container import Container
from tsgo.container.point import Point
from tsgo.algorithm import energy_functions as ef

def get_datetime() -> str:
    """
    Gets the current date and time and formats as yyyy-mm-dd hh:mm:ss.sss

    Returns:
        str: Formatted date and time string.
    """
    datetime_now = datetime.now()
    datetime_now = datetime_now.strftime('%Y-%m-%d %H:%M:%S.') + datetime_now.strftime('%f')[:3]
    return datetime_now


def save_to_file(
    p: int, 
    parallel_iterations: int, # Number of global multi-starts
    iterations: int,          # Number of inner search steps
    D_init: float, 
    X_opt: Union[np.ndarray, List], 
    D_opt: float, 
    container: any, 
    elapsed_time: float, 
    Q,
    beta_max,
    eps,
    optimization_method,
    metric_name: str,
    output_dir: str,

) -> None:
    """
    Saves the global optimization results for the 3D p-dispersion problem.
    Reflects the multi-start pipeline strategy used to navigate non-convex solution spaces.
    """
    
    # 1. Organize Directory Structure
    # Using parallel_iterations to define the folder level for batch comparisons
    subfolder_path = os.path.join(output_dir, metric_name.lower(), f"batch_{parallel_iterations}")
    os.makedirs(subfolder_path, exist_ok=True)
    
    # 2. Define File Paths
    base_filename = f"TSGO_p{p}_iter{iterations}_results"
    txt_path = os.path.join(subfolder_path, f"{base_filename}.txt")
    pkl_path = os.path.join(subfolder_path, f"{base_filename}.pkl")
    
    # 3. Format Text Output
    # Including both global (parallel) and local iteration counts for completeness
    lines = [
        f"--- TSGO 3D Dispersion Results ---",
        f"Saved on: {get_datetime()}",
        f"Container: {container.description}",
        f"Metric: {metric_name}",
        f"Number of Points (p): {p}",
        f"Global Multi-start (Parallel Iterations): {parallel_iterations}",
        f"Inner Search Steps (Iterations): {iterations}",
        f"D_initial: {D_init}",
        f'Q: {Q}',
        f'beta_max: {beta_max}',
        f'eps: {eps}',
        f'Optimization Method: {optimization_method}',
        "-" * 30,
        "Optimal Arrangement of Points (x, y, z):"
    ]
    
    for point in X_opt:
        if isinstance(point, np.ndarray):
            lines.append(f"{point[0]:.8f}, {point[1]:.8f}, {point[2]:.8f}")
        else:
            lines.append(f"{point.x:.8f}, {point.y:.8f}, {point.z:.8f}")
            
    lines.extend([
        "-" * 30,
        f"Optimal Radius (D): {D_opt:.8f}",
        f"Final Energy: {ef.E(X_opt, D_opt, container)}",
        f"Elapsed Time: {elapsed_time:.2f} seconds",
        "-----------------------------------"
    ])
    
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))

    # 4. Pickle Data (Including both iteration parameters)
    with open(pkl_path, "wb") as f:
        pickle.dump({
            'p': p,
            'parallel_iterations': parallel_iterations,
            'iterations': iterations,
            'D_init': D_init,
            'X_opt': X_opt,
            'D_opt': D_opt,
            'container': container,
            'metric': metric_name,
            'elapsed_time': elapsed_time,
            'timestamp': get_datetime()
        }, f)
    
def collect_best(output_dir: str) -> None:
    """
    Collects the best results from multiple runs stored in the output directory.

    Args:
        output_dir (str): The directory where output files are stored.
    """
    # Scan through the output directory recursively for any files that end in .pkl
    # Load each pickle file and extract the D_opt value
    # Get the file with the highest D_opt value
    # copy this file to best_result.pkl in the output directory
    best_D_opt = -np.inf
    best_file = None
    
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".pkl"):
                file_path = os.path.join(root, file)
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                    D_opt = data.get('D_opt', -np.inf)
                    if D_opt > best_D_opt:
                        best_D_opt = D_opt
                        best_file = file_path
    if best_file is not None:
        dest_file = os.path.join(output_dir, "best_result.pkl")
        shutil.copy(best_file, dest_file)
        print(f"[TSGO] Best result collected: {best_file} with D_opt = {best_D_opt}. Saved to: {dest_file}")
        
def collect_best_entrypoint():
    import argparse

    parser = argparse.ArgumentParser(description="Collect Best Results from TSGO Output Directory")
    parser.add_argument("--dir", type=str, required=True, help="Output directory containing TSGO results")
    
    args = parser.parse_args()
    
    collect_best(args.dir)