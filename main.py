import numpy as np

from tsgo.point import Point
from tsgo.sample_polyhedron import cube, h_box, pointy_star, triangular_pyramid
from tsgo.visualizer import plot_active_faces, plot_active_footpoints, plot_result
from tsgo.cli import run_tsgo
from tsgo.tsgo_algorithm import tabu_search, adjust_distance


## USE FOR GENERATING DIAGRAMS FOR THE PAPER
# if __name__ == "__main__":
#     # i need to generate diagrams on the effect of tabu search on a configuration
#     # generate a random solution with p = 5 with D = 0.1 and plot
#     # run tabu search on it for Q = 2 and beta_max = 1
#     # plot the result
#     np.random.seed(42)
#     polyhedron = cube()
#     D = 0.2
#     colors = ['r', 'g', 'b', 'y', 'c']
    
#     initial_config = polyhedron.generate_initial_solution(p=5)
#     plot_result(initial_config, D, polyhedron, title="Initial Configuration", grid=False, colors=colors)
#     post_tabu_search = tabu_search(initial_config, D=D, container=polyhedron, beta_max=1, Q=2, method='L-BFGS-B', metric='EUCLIDEAN')
#     plot_result(post_tabu_search, D, polyhedron, title="Post Tabu Search Configuration", grid=False, colors=colors)
#     post_adjust_distance = adjust_distance(post_tabu_search, D0=D, container=polyhedron, method='BFGS', metric='EUCLIDEAN')
#     plot_result(post_adjust_distance[0], post_adjust_distance[1], polyhedron, title="Distance Adjusted Configuration", grid=False, colors=colors)


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