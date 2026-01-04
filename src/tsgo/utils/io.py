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
    iterations: int, 
    D_init: float, 
    X_opt: Union[np.ndarray, List[Point]], 
    D_opt: float, 
    container: Container, 
    elapsed_time: float, 
    metric_name: str = 'EUCLIDEAN',
    output_dir: str = "output"
) -> None:
    print("[TSGO] Saving results to file.")
    
    # Create output directory structure first
    subfolder_name = os.path.join(output_dir, metric_name.lower(), f"iterations{iterations}")
    os.makedirs(subfolder_name, exist_ok=True)
    
    # Define file paths directly in target location
    txt_filename = os.path.join(subfolder_name, f"TSGO_{p}_data.txt")
    pkl_filename = txt_filename.replace(".txt", ".pkl")
    
    # Write text file
    string = ""
    string += f"Number of Points: {p}\nIterations: {iterations}\nD_initial: {D_init}\nContainer: {container.description}\nMetric: {metric_name}\n"
    string += "\nOptimal Arrangement of Points:\n"
    for point in X_opt:
        if isinstance(point, np.ndarray):
            string += f"{point[0]}, {point[1]}, {point[2]}\n"
        elif isinstance(point, Point):
            string += f"{point.x}, {point.y}, {point.z}\n"
        else:
            string += str(point) + "\n"
    
    string += f"\nOptimal Minimum Distance between Points: {D_opt}\n"
    string += f"Time taken: {elapsed_time:.2f} seconds\n"
    string += f"Saved on: {get_datetime()}\n"
    string += f"Final Energy: {ef.E(X_opt, D_opt, container)}\n"
    
    with open(txt_filename, "w") as f:
        f.write(string)

    # Write pickle file
    with open(pkl_filename, "wb") as f:
        pickle.dump({
            'optimized points': X_opt,
            'D': D_opt,
            'container': container,
            'elapsed_time': elapsed_time,
            'p': p,
            'iterations': iterations,
            'D_init': D_init,
            'metric': metric_name
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
                    D_opt = data.get('D', -np.inf)
                    if D_opt > best_D_opt:
                        best_D_opt = D_opt
                        best_file = file_path
    if best_file is not None:
        dest_file = os.path.join(output_dir, "best_result.pkl")
        shutil.copy(best_file, dest_file)
        print(f"[TSGO] Best result collected: {best_file} with D_opt = {best_D_opt}")
        print(f"[TSGO] Best result saved to: {dest_file}")
        
def collect_best_entrypoint():
    import argparse

    parser = argparse.ArgumentParser(description="Collect Best Results from TSGO Output Directory")
    parser.add_argument("--dir", type=str, required=True, help="Output directory containing TSGO results")
    
    args = parser.parse_args()
    
    collect_best(args.dir)