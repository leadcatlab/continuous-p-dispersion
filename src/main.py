from tsgo.algorithm.tsgo_algorithm import tabu_search, adjust_distance
from tsgo.container.point import Point
from tsgo.container.sample_polyhedron import cube, h_box, pointy_star, triangular_pyramid
from tsgo.plotting.visualizer import plot_active_faces, plot_active_footpoints, plot_result
from tsgo.algorithm import energy_functions as ef

if __name__ == "__main__":
    point = Point(1.5, 0.5, 2.5)
    container = h_box()
    
    # Create a diagram for the active footpoints of container wrt point
    plot_active_footpoints(point, container, title="")