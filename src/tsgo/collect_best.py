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
    elapsed_time = None

    for f in files:
        data = np.load(f)
        D = float(data["D"])
        if D > best_D:
            best_D = D
            best_result = data
            best_file = f
        if elapsed_time is None:
            elapsed_time = float(data.get("elapsed_time", 0.0))
        else:
            elapsed_time = max(elapsed_time, float(data.get("elapsed_time", 0.0)))
    
    print("=== GLOBAL BEST ===")
    print("D:", best_D)
    print("Radius:", best_D / 2)
    print("From:", best_file)
    print("X:", best_result["X"])
    print("seed:", best_result["seed"])
    print("elapsed_time:", best_result["elapsed_time"])
    print("metric:", best_result["metric"])
    print("iterations:", best_result["iterations"])
    print("D_init:", best_result["D_init"])
    print("container:", best_result["container"])
    print("Q:", best_result["Q"])
    print("beta_max:", best_result["beta_max"])
    print("eps:", best_result["eps"])
    print("======= END =======")

    np.savez("results/best_global.npz", **best_result)
    
    # save all the data to a pickle file
    import pickle
    with open(f"results/best_global_p={args.p}.pkl", "wb") as f:
        pickle.dump({
            "X": best_result["X"],
            "D": best_result["D"],
            "seed": best_result["seed"],
            "elapsed_time": best_result["elapsed_time"],
            "metric": best_result["metric"],
            "iterations": best_result["iterations"],
            "D_init": best_result["D_init"],
            "container": best_result["container"],
            "Q": best_result["Q"],
            "beta_max": best_result["beta_max"],
            "eps": best_result["eps"]
        }, f)

if __name__ == "__main__":
    main()
