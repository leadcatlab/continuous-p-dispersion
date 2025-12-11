import argparse
import os
import numpy as np
import time

from tsgo.cli import run_tsgo, polyhedron_examples
from tsgo.utils import get_datetime, save_to_file

def main():
    # Gather command-line arguments
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
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        args.output_dir = f"results/{args.container}/p{args.p}"
    os.makedirs(args.output_dir, exist_ok=True)

    # Assign workers from SLURM
    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    if slurm_cpus is None:
        slurm_cpus = 3
    else:
        print(f"SLURM_CPUS_PER_TASK: {slurm_cpus}")
    num_workers = int(slurm_cpus)

    # Create unique seed for each task
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
    seed = np.random.randint(0, 2 ** 31 - 1)
    np.random.seed(seed)

    # Select container
    container = polyhedron_examples[args.container]

    start_time = time.time()
    print(f"[Task {task_id} @ {get_datetime()}] Running TSGO with {num_workers} workers and seed {seed}")
    print(f"[Task {task_id}] Container: {args.container}, p: {args.p}, iterations: {args.iterations}, D_init: {args.D_init}, metric: {args.metric}")
    

    _, D_best = run_tsgo(
        p=args.p,
        iterations=args.iterations,
        D_init=args.D_init,
        container=container,
        metric_name=args.metric,
        save=True,
        output_dir=args.output_dir,
        plot=False,
        num_workers=num_workers,
        optimization_method=args.solver,
        Q=args.Q,
        beta_max=args.beta_max,
        eps=args.eps,
    )
    
    elapsed_time = time.time() - start_time
    print(f"[Task {task_id}] Completed in {elapsed_time:.2f} seconds")
    print(f"[Task {task_id}] Best distance found: {D_best}")
    

if __name__ == "__main__":
    main()
