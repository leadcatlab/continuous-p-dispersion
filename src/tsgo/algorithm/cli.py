import os
import secrets
import shutil
import time

from tsgo.configuration import Config
from tsgo.algorithm import energy_functions as ef
from tsgo.algorithm.tsgo_algorithm import tsgo_algorithm_parallel
from tsgo.container.container import Container
from tsgo.container.point import Point
from tsgo.container.sample_polyhedron import cube, h_box, triangular_pyramid, pointy_star, ribbed_cage, CONTAINERS
from tsgo.plotting.visualizer import plot_result
from tsgo.utils.io import get_datetime, save_to_file

# NOTE: This inverse of this map is the description of each container in sample_polyhedron.py
polyhedron_examples = {
    "cube": cube(radius=0.5),
    "h-box": h_box(),
    "tetrahedron": triangular_pyramid(),
    "pointy-star": pointy_star(),
    "cage": ribbed_cage(),
}

def run_tsgo(
    config: Config
):
    start_time = time.time()
    
    vprint = print if config.verbose else lambda *args, **kwargs: None

    vprint(f"[TSGO] Initializing TSGO Algorithm with configuration: {config}")

    X_opt, D_opt = tsgo_algorithm_parallel(
        p=config.p,
        D_init=config.D_init,
        container=config.container,
        Q=config.Q,
        beta_max=config.beta_max,
        eps=config.eps,
        metric=config.metric,
        iterations=config.iterations,
        processes=config.processes,
        optimization_method=config.optimization_method,
        output_dir=config.output_dir,
        verbose=config.verbose,
        seed=config.seed,
    )
    
    elapsed_time = time.time() - start_time
    print(f"[TSGO @ {get_datetime()}] Optimization completed in {elapsed_time:.8f}s | p={config.p} | seed={config.seed} | D_opt={D_opt:.10f} | E={ef.E(X_opt, D_opt, config.container):.10f}")

    # Save to file
    if config.save:
        save_to_file(
            p=config.p,
            parallel_iterations=config.processes,
            iterations=config.iterations,
            D_init=config.D_init,
            X_opt=X_opt,
            D_opt=D_opt,
            container=config.container,
            elapsed_time=elapsed_time,
            metric_name=config.metric,
            output_dir=config.output_dir,
            Q=config.Q,
            beta_max=config.beta_max,
            eps=config.eps,
            optimization_method=config.optimization_method,
            seed=config.seed,
        )
        shutil.rmtree(f'{config.output_dir}/crashs', ignore_errors=True)

    # Plot results
    if config.plot is True:
        plot_result(X_opt, D_opt, container=config.container, show_normals=False, metric_name=config.metric)
    
    return X_opt, D_opt


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run TSGO Algorithm for Continuous p-Dispersion Problem")
    parser.add_argument("--p", type=int, help="Number of points to place")
    parser.add_argument("--container", type=str, choices=polyhedron_examples.keys(), help="Container shape")
    parser.add_argument("--D_init", type=float, help="Initial minimum distance")
    parser.add_argument("--packing_density", type=float, default=None, help="Packing density for initial solution generation (if applicable)")
    parser.add_argument("--Q", type=int, default=None,
                        help="Tabu neighbourhood size. Default max(3, round(ln p)).")
    parser.add_argument("--beta_max", type=int, default=10, help="Maximum beta value for the algorithm")
    parser.add_argument("--eps", type=float, default=1e-8, help="Epsilon tolerance for optimization")
    parser.add_argument("--metric", type=str, default="EUCLIDEAN", help="Distance metric to use")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations")
    parser.add_argument("--processes", type=int, default=1, help="Number of parallel iterations")
    parser.add_argument("--optimization_method", type=str, default="L-BFGS-B", help="Optimization method to use")
    parser.add_argument("--save", type=bool, default=True, help="Whether to save results to file")
    parser.add_argument("--plot", type=bool, default=False, help="Whether to plot the final configuration")
    parser.add_argument("--output_dir", type=str, default='', help="Output directory for saved files")
    parser.add_argument("--verbose", type=bool, default=False, help="Whether to print detailed logs during optimization")
    parser.add_argument("--seed", type=int, default=None, help="Master RNG seed. Omit to draw one and log it, so ad hoc runs stay reproducible after the fact.")
    
    args = parser.parse_args()
    
    configuration = Config(**{
        "p": args.p,
        "container": args.container,
        "D_init": args.D_init,
        "packing_density": args.packing_density,
        "Q": args.Q,
        "beta_max": args.beta_max,
        "eps": args.eps,
        "metric": args.metric,
        "iterations": args.iterations,
        "processes": args.processes,
        "optimization_method": args.optimization_method,
        "save": args.save,
        "plot": args.plot,
        "output_dir": args.output_dir,
        "verbose": args.verbose,
        "seed": args.seed if args.seed is not None else secrets.randbits(32),
    })

    print(f"[TSGO] seed = {configuration.seed}  (pass --seed {configuration.seed} to reproduce this run)")
    
    # Assign workers from SLURM environment variable if available
    slurm_cpus_per_task = os.getenv("SLURM_CPUS_PER_TASK")
    if slurm_cpus_per_task is None:
        configuration.processes = args.processes if args.processes > 0 else 6
    else:
        print(f"[TSGO] Using 'SLURM_CPUS_PER_TASK' environment variable to set number of workers: {slurm_cpus_per_task}.")
        configuration.processes = int(slurm_cpus_per_task)
        
    
    # Get the task_id from SLURM environment variable if available
    slurm_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    
    run_tsgo(
        config=configuration
    )
    
    if slurm_task_id is not None:
        print(f"[TSGO] Task {slurm_task_id} completed successfully.")
    
if __name__ == "__main__":
    main()