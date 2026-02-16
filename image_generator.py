from tsgo.algorithm.tsgo_algorithm import tabu_search
from tsgo.container.sample_polyhedron import h_box, cube, triangular_pyramid, pointy_star, ribbed_cage

h_box_model = h_box()
pointy_star_model = pointy_star()
cage = ribbed_cage()

# plot the hbox with the point (1.5, 0.5, 1.5) highlighted

if __name__ == '__main__':
    # Plot the active faces and footpoints of the h_box with respect to the point (1.5, 0.5, 2.5)
    from tsgo.plotting.visualizer import plot_active_faces, plot_active_footpoints, plot_result
    from tsgo.container.point import Point
    
    # create 9 disperison points with radius 0.1025 in the tetrahedron
    container = triangular_pyramid()
    X = container.generate_initial_solution(p=9)
    D = 0.1025
    X = tabu_search(X, D=D, container=container, beta_max=1, Q=2, optimization_method='L-BFGS-B', metric='EUCLIDEAN')
    plot_result(X, D, container, title="9-Point Dispersion in Triangular Pyramid", grid=False, colors='Blues')