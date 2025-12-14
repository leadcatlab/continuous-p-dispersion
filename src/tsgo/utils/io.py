import os
import pickle
import shutil
from datetime import datetime
from typing import List, Union

import numpy as np

from tsgo.container.container import Container
from tsgo.container.point import Point

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
    string = ""
    string += f"Number of Points: {p}\nIterations: {iterations}\nD_initial: {D_init}\n"
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
    filename = f"TSGO_{p}_data.txt"
    with open(filename, "w") as f:
        f.write(string)

    # save p, iterations, D_init, X_opt, D_opt, container, elapsed_time, metric to a pickle file
    with open(filename.replace(".txt", ".pkl"), "wb") as f:
        pickle.dump({
            "optimized points": X_opt,
            "D": D_opt,
            "container": container,
            "elapsed_time": elapsed_time,
            "p": p,
            "iterations": iterations,
            "D_init": D_init,
            "metric": metric_name
        }, f)
        f.close()

    subfolder_name = os.path.join(output_dir, container.description, f"p{p}", metric_name.lower(), f"iterations{iterations}")
    os.makedirs(subfolder_name, exist_ok=True)
    
    dest_txt = os.path.join(subfolder_name, os.path.basename(filename))
    dest_pkl = os.path.join(subfolder_name, os.path.basename(filename.replace(".txt", ".pkl")))
    # Remove destination files if they exist
    if os.path.exists(dest_txt):
        os.remove(dest_txt)
    if os.path.exists(dest_pkl):
        os.remove(dest_pkl)

    shutil.move(filename, subfolder_name)
    shutil.move(filename.replace(".txt", ".pkl"), subfolder_name)