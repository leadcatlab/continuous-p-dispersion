import numpy as np
import shutil
import glob
import os
import argparse
import pickle

from tsgo.container.sample_polyhedron import cube, h_box, triangular_pyramid, pointy_star
from tsgo.algorithm.cli import polyhedron_examples

def collect_best(p: int, container_name: str, output_dir: str = "output", metric_name: str = "euclidean") -> None:
    """
    Scan through every .pkl file recursively in the output directory and collect the best results.

    Args:
        p (int): Number of points.
        container (str): container key from polyhedron_examples.
        output_dir (str, optional): Directory to search for .pkl files. Defaults to "output".
    """
    best_result = None
    best_D = -np.inf
    search_pattern = os.path.join(output_dir, container_name, f"p{p}", "**", "*.pkl")
    output_path = ""
    
    for filepath in glob.iglob(search_pattern, recursive=True):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            D_opt = data.get("D", -np.inf)
            if D_opt > best_D:
                best_D = D_opt
                best_result = data
                output_path = filepath
                
    if best_result is not None:
        output_dir = os.path.join(output_dir, container_name, f"p{p}")
        output_path = os.path.join(output_dir, f"best_result_p{p}.pkl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(best_result, f)
        print(f"Best result for p={p} in container={container_name} saved to {output_path}")
        print(f"==> Best D: {best_D}")
    else:
        print(f"No results found for p={p} in container={container_name} within {output_dir}")
        
def main():
    parser = argparse.ArgumentParser(description="Collect best TSGO results.")
    parser.add_argument("--p", type=int, required=True, help="Number of points.")
    parser.add_argument("--container", type=str, required=True, help="Container key from polyhedron_examples.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to search for .pkl files.")
    
    args = parser.parse_args()
    collect_best(args.p, args.container, args.output_dir)
    
if __name__ == "__main__":
    main()