import numpy as np
import glob
import os
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=None, help="P value for results directory")
    args = parser.parse_args()
    
    if args.p is not None:
        pattern = f"results/p{args.p}/best_task_*.npz"
    else:
        pattern = "results/best_task_*.npz"
    
    files = sorted(glob.glob(pattern))
    best_D = -1
    best_result = None
    best_file = None

    for f in files:
        data = np.load(f)
        D = float(data["D"])
        if D > best_D:
            best_D = D
            best_result = data
            best_file = f

    print("All DATA: ", best_result)
    
    print("=== GLOBAL BEST ===")
    print("D:", best_D)
    print("Radius:", best_D / 2)
    print("From:", best_file)
    print("======= END =======")

    np.savez("results/best_global.npz", **best_result)

if __name__ == "__main__":
    main()
