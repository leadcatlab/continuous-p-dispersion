import numpy as np
import glob
import os

def main():
    files = sorted(glob.glob("results/best_task_*.npz"))
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

    print("=== GLOBAL BEST ===")
    print("D:", best_D)
    print("From:", best_file)

    np.savez("results/best_global.npz", **best_result)

if __name__ == "__main__":
    main()
