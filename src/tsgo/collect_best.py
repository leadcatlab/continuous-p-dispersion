import numpy as np
import glob
import os
import argparse
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=None, help="P value for results directory")
    parser.add_argument("--container", type=str, default="cube", help="Container shape (not used)")
    args = parser.parse_args()
    
    if args.p is not None:
        pattern = f"results/{args.container}/p{args.p}/best_task_*.npz"
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
    elapsed_time = max([data.get("elapsed_time", 0.0) for f in files for data in [np.load(f)]])

    print("=== GLOBAL BEST ===")
    print("D:", best_D)
    print("Radius:", best_D / 2)
    print("From:", best_file)
    print("X:", best_result["X"])
    print("seed:", best_result["seed"])
    print("elapsed_time:", elapsed_time)
    print("metric:", best_result["metric"])
    print("iterations:", best_result["iterations"])
    print("D_init:", best_result["D_init"])
    print("container:", best_result["container"])
    print("Q:", best_result["Q"])
    print("beta_max:", best_result["beta_max"])
    print("eps:", best_result["eps"])
    print("======= END =======")
    
    best_result["elapsed_time"] = elapsed_time
    np.savez(f"results/{args.container}/p{args.p}/best_global.npz", **best_result)
    
    # save all the data to a pickle file
    with open(f"results/{args.container}/p{args.p}/best_global.pkl", "wb") as f:
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

def get_best():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=None, help="P value for results directory")
    parser.add_argument("--container", type=str, default="cube", help="Container shape (not used)")
    args = parser.parse_args()
    
    if args.p is not None:
        pattern = f"results/{args.container}/p{args.p}/best_global.pkl"
    else:
        pattern = "results/best_global.pkl"

    with open(pattern, "rb") as f:
        best_global = pickle.load(f)
        print(best_global)
    
    print("=== GLOBAL BEST ===")
    print("D:", best_global["D"])
    print("Radius:", best_global["D"] / 2)
    print("X:", best_global["X"])
    print("seed:", best_global["seed"])
    print("elapsed_time:", best_global["elapsed_time"])
    print("metric:", best_global["metric"])
    print("iterations:", best_global["iterations"])
    print("D_init:", best_global["D_init"])
    print("container:", best_global["container"])
    print("Q:", best_global["Q"])
    print("beta_max:", best_global["beta_max"])
    print("eps:", best_global["eps"])
    print("======= END =======")

if __name__ == "__main__":
    main()
