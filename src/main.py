from tsgo.algorithm.tsgo_algorithm import tabu_search, adjust_distance
from tsgo.container.point import Point
from tsgo.container.sample_polyhedron import cube, h_box, pointy_star, triangular_pyramid
from tsgo.plotting.visualizer import plot_active_faces, plot_active_footpoints, plot_result
from src.tsgo.algorithm.cli import run_tsgo

if __name__ == "__main__":
    for i in [4, 10]:
        run_tsgo(
            p=i, 
            container=triangular_pyramid(), 
            D_init=0.1, 
            iterations=10, 
            metric_name='EUCLIDEAN', 
            optimization_method='L-BFGS-B', 
            output_dir="output",
            num_workers=5, 
            save=True, 
            plot=False
        )
    for i in [7, 10]:
        run_tsgo(
            p=i, 
            container=h_box(), 
            D_init=0.1, 
            iterations=10, 
            metric_name='EUCLIDEAN', 
            optimization_method='L-BFGS-B', 
            output_dir="output",
            num_workers=5, 
            save=True, 
            plot=False
        )