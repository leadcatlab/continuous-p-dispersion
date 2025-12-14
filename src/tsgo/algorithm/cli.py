import os
import time

from tsgo.algorithm import energy_functions as ef
from tsgo.algorithm.tsgo_algorithm import tsgo_algorithm_parallel
from tsgo.container.container import Container
from tsgo.container.point import Point
from tsgo.container.sample_polyhedron import cube, h_box, triangular_pyramid, pointy_star
from tsgo.plotting.visualizer import plot_result
from tsgo.utils.io import save_to_file

# NOTE: This inverse of this map is the description of each container in sample_polyhedron.py
polyhedron_examples = {
    "cube": cube(radius=0.5),
    "h-box": h_box(),
    "tetrahedron": triangular_pyramid(),
    "pointy-star": pointy_star()
}

def run_tsgo(
    p: int = 3,
    iterations: int = 10,
    D_init: float = 0.1,
    container: Container = cube(radius=0.5),
    metric_name: str = 'EUCLIDEAN',
    optimization_method: str = "BFGS",
    Q: int = 3,
    beta_max: int = 5,
    eps: float = 1e-16,
    save: bool = True,
    output_dir: str = "",
    plot: bool = False,
    num_workers: int = 6,
):
    start_time = time.time()
    print(f"[TSGO] Initializing TSGO Algorithm with p={p} points in container {container} using metric {metric_name}.")

    X_opt, D_opt = tsgo_algorithm_parallel(
        p=p,
        iterations=iterations,
        D_init=D_init,
        container=container,
        metric_name=metric_name,
        optimization_method=optimization_method,
        Q=Q,
        beta_max=beta_max,
        eps=eps,
        max_workers=num_workers,
    )
    
    elapsed_time = time.time() - start_time
    print(f"[TSGO] Completed Optimization in {elapsed_time:.2f} seconds")
    
    # Report final outputs
    print(f"[TSGO] Optimization completed with p={p} points in container {container} with metric {metric_name}.")
    print(f"[TSGO] Minimum distance between points: {D_opt}.")
    print(f"[TSGO] Final Energy: {ef.E(X_opt, D_opt, container)}.")

    # Save to file
    if save:
        save_to_file(p, iterations, D_init, X_opt, D_opt, container, elapsed_time, metric_name=metric_name, output_dir=output_dir)

    # Plot results
    if plot:
        plot_result(X_opt, D_opt, container=container, show_normals=False, metric_name=metric_name)
    
    return X_opt, D_opt


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run TSGO Algorithm for Continuous p-Dispersion Problem")
    parser.add_argument("--p", type=int, default=3, help="Number of points to place")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    parser.add_argument("--D_init", type=float, default=0.1, help="Initial minimum distance")
    parser.add_argument("--container", type=str, default="cube", choices=polyhedron_examples.keys(), help="Container shape")
    parser.add_argument("--metric", type=str, default="EUCLIDEAN", help="Distance metric to use")
    parser.add_argument("--solver", type=str, default="L-BFGS-B", help="Optimization method to use")
    parser.add_argument("--Q", type=int, default=3, help="Parameter Q for the algorithm")
    parser.add_argument("--beta_max", type=int, default=5, help="Maximum beta value for the algorithm")
    parser.add_argument("--save", type=bool, default=True, help="Whether to save results to file")
    parser.add_argument("--num_workers", type=int, default=6, help="Number of parallel workers")
    parser.add_argument("--output_dir", type=str, default="", help="Output directory for saved files")
    parser.add_argument("--plot", type=bool, default=False, help="Whether to plot the final configuration")
    
    args = parser.parse_args()
    
    # Setup the output directory
    output_dir = args.output_dir
    if not output_dir:
        output_dir = f"output/{args.container}/p{args.p}"
    print(f"[TSGO] Output directory set to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Assign workers from SLURM environment variable if available
    slurm_cpu_per_task = os.getenv("SLURM_CPU_PER_TASK")
    if slurm_cpu_per_task is None:
        num_workers = args.num_workers if args.num_workers > 0 else 6
    else:
        print(f"[TSGO] Using 'SLURM_CPU_PER_TASK' environment variable to set number of workers: {slurm_cpu_per_task}.")
        num_workers = int(slurm_cpu_per_task)
        
    
    # Get the task_id from SLURM environment variable if available
    slurm_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    
    run_tsgo(
        p=args.p, 
        iterations=args.iterations, 
        D_init=args.D_init, 
        container=polyhedron_examples[args.container], 
        save=args.save, 
        output_dir=output_dir,
        plot=args.plot, 
        num_workers=num_workers, 
        optimization_method=args.solver, 
        Q=args.Q, 
        metric_name=args.metric
    )
    
    if slurm_task_id is not None:
        print(f"[TSGO] Task {slurm_task_id} completed successfully.")
    
if __name__ == "__main__":
    main()
