import argparse
import os
import numpy as np
import time

from tsgo.cli import run_tsgo, polyhedron_examples

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--p", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--D_init", type=float, required=True)
    parser.add_argument("--container", type=str, required=True)
    parser.add_argument("--metric", type=str, default="EUCLIDEAN")
    parser.add_argument("--solver", type=str, default="L-BFGS-B")
    parser.add_argument("--Q", type=int, default=3)
    parser.add_argument("--beta_max", type=int, default=5)
    parser.add_argument("--eps", type=float, default=1e-16)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = f"results/{args.container}/p{args.p}"
    os.makedirs(args.output_dir, exist_ok=True)

    # Assign workers from SLURM
    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    if slurm_cpus is None:
        raise RuntimeError("SLURM_CPUS_PER_TASK not set")

    num_workers = int(slurm_cpus)

    # Compute seed from SLURM array ID
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
    seed = 1337 + task_id

    np.random.seed(seed)

    container = polyhedron_examples[args.container]

    print(f"[Task {task_id}] Running TSGO with {num_workers} workers and seed {seed}")
    
    start_time = time.time()

    X_best, D_best = run_tsgo(
        p=args.p,
        iterations=args.iterations,
        D_init=args.D_init,
        container=container,
        metric_name=args.metric,
        save=False,
        plot=False,
        num_workers=num_workers,
        optimization_method=args.solver,
        Q=args.Q,
        beta_max=args.beta_max,
        eps=args.eps,
    )
    
    elapsed_time = time.time() - start_time
    print(f"[Task {task_id}] Completed in {elapsed_time:.2f} seconds")

    # Save output
    os.makedirs(args.output_dir, exist_ok=True)
    outfile = os.path.join(args.output_dir, f"best_task_{task_id}.npz")
    np.savez(
        outfile, 
        X=X_best, 
        D=D_best, 
        seed=seed, 
        elapsed_time=elapsed_time, 
        metric=args.metric,
        iterations=args.iterations,
        D_init=args.D_init,
        container=args.container,
        Q=args.Q,
        beta_max=args.beta_max,
        eps=args.eps
    )
    print(f"[Task {task_id}] Saved result to {outfile}")

if __name__ == "__main__":
    main()
