import numpy as np

from tsgo.algorithm.tsgo_algorithm import tabu_search, adjust_distance
from tsgo.container.point import Point
from tsgo.container.sample_polyhedron import cube, h_box, pointy_star, triangular_pyramid
from tsgo.plotting.visualizer import plot_active_faces, plot_active_footpoints, plot_result
from src.tsgo.algorithm.cli import run_tsgo


if __name__ == "__main__":
    np.random.seed(42)
    polyhedron = cube()
    D = 0.2
    colors = ['r', 'g', 'b', 'y', 'c']
    
    initial_config = polyhedron.generate_initial_solution(p=5)
    plot_result(initial_config, D, polyhedron, title="Initial Configuration", grid=False, colors=colors)
    
    post_tabu_search = tabu_search(initial_config, D=D, container=polyhedron, beta_max=1, Q=2, method='L-BFGS-B', metric='EUCLIDEAN')
    plot_result(post_tabu_search, D, polyhedron, title="Post Tabu Search Configuration", grid=False, colors=colors)
    
    post_adjust_distance = adjust_distance(post_tabu_search, D0=D, container=polyhedron, method='BFGS', metric='EUCLIDEAN')
    plot_result(post_adjust_distance[0], post_adjust_distance[1], polyhedron, title="Distance Adjusted Configuration", grid=False, colors=colors)
