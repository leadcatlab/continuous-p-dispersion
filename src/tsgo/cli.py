import time

from tsgo.point import Point
from tsgo.container import Container
import tsgo.energy_functions as ef
from tsgo.visualizer import plot_result
from tsgo.utils import save_to_file
from tsgo.sample_polyhedron import cube, h_box, triangular_pyramid, pointy_star
from tsgo.tsgo_algorithm import tsgo_algorithm_parallel
from tsgo.metric_enums import MetricType, get_metric_function, list_metrics

polyhedron_examples = {
    "cube": cube(radius=0.5),
    "h-box": h_box(),
    "tetrahedron": triangular_pyramid(),
    "pointy-star": pointy_star()
}

# https://www.randomwalk.de/sphere/incube/spicbest.txt

def run_tsgo(
    p: int = 3,
    iterations: int = 10,
    D_init: float = 0.1,
    container: Container = cube(radius=0.5),
    metric_name: str = 'EUCLIDEAN',
    save: bool = True,
    plot: bool = False,
    num_workers: int = 6,
    optimization_method: str = "BFGS",
    Q: int = 3,
    beta_max: int = 5,
    eps: float = 1e-16,
):
    start_time = time.time()
    print("[TSGO] Initializing TSGO Algorithm")

    X_opt, D_opt = tsgo_algorithm_parallel(
        iterations=iterations,
        p=p,
        container=container,
        D_init=D_init,
        metric_name=metric_name,
        beta_max=beta_max,
        Q=Q,
        max_workers=num_workers,
        eps=eps,
        optimization_method=optimization_method,
    )

    # final outputs
    print("Optimal configuration of points:\n", X_opt)
    print("Minimum distance between points:", D_opt)
    print(f"Energy of the system: {round(ef.E(X_opt, D_opt, container), 20)}")

    # timing
    elapsed_time = time.time() - start_time
    print(f"[TSGO] Completed Optimization in {elapsed_time:.2f} seconds")

    # save to file
    if save:
        save_to_file(p, iterations, D_init, X_opt, D_opt, container, elapsed_time, metric_name=metric_name)

    # plot the result
    if plot:
        plot_result(X_opt, D_opt, container=container, show_normals=False, metric_name=metric_name)
    return X_opt, D_opt

def main():
    import argparse
    import re
    parser = argparse.ArgumentParser(description="Run TSGO Algorithm for Continuous p-Dispersion Problem")
    parser.add_argument("--p", type=int, default=3, help="Number of points to place")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    parser.add_argument("--D_init", type=float, default=0.1, help="Initial minimum distance")
    parser.add_argument("--container", type=str, default="cube", choices=polyhedron_examples.keys(), help="Container shape")
    parser.add_argument("--save", type=bool, default=True, help="Whether to save results to file")
    parser.add_argument("--plot", type=bool, default=False, help="Whether to plot the result")
    parser.add_argument("--num_workers", type=int, default=6, help="Number of parallel workers")
    parser.add_argument("--solver", type=str, default="L-BFGS-B", help="Optimization method to use")
    parser.add_argument("--Q", type=int, default=3, help="Parameter Q for the algorithm")
    def metric_type(s):
        if s in list_metrics():
            return s
        if re.match(r"^N_NORM_\d+$", s):
            return s
        raise argparse.ArgumentTypeError(
            f"Invalid metric '{s}'. Must be one of {list_metrics()} or match 'N_NORM_<int>'"
        )

    parser.add_argument(
        "--metric",
        type=metric_type,
        default="EUCLIDEAN",
        help="Distance metric to use (one of list_metrics() or N_NORM_<int>)",
    )
    
    args = parser.parse_args()
    
    run_tsgo(
        p=args.p, 
        iterations=args.iterations, 
        D_init=args.D_init, 
        container=polyhedron_examples[args.container], 
        save=args.save, 
        plot=args.plot, 
        num_workers=args.num_workers, 
        optimization_method=args.solver, 
        Q=args.Q, 
        metric_name=args.metric
    )
    
if __name__ == "__main__":
    main()
