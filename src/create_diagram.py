import numpy as np

from tsgo.container.sample_polyhedron import cube
from tsgo.plotting.visualizer import plot_result

from tsgo.algorithm.tsgo_algorithm import adjust_distance


# if __name__ == "__main__":
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
    # draw a two point configuration in the unit cube with D = 0.3 and have one sphere intersect the boundary of the container
    # for the sphere intersecting the boundary of the container highlight it in red, and have the other sphere in blue
    # draw a third sphere to a feasible area of the container so that it does not intersect the boundary of the container and highlight it in darker blue and draw an arrow
    # from the red sphere to the darker blue sphere to indicate the movement of the point to a feasible area
    polyhedron = cube()
    D = 0.20
    colors = ['grey', 'grey', '#fcdf03']
    initial_config = np.array(
        [
            # [-0.4, 0.4, 0.4],
            [-0.2, -0.2, -0.2],
            [0.3, -0.2, -0.2],
            [0.0, -0.0, 0.2]
        ]
    )
    
    X, D = adjust_distance(initial_config, D0=D, container=polyhedron, method='BFGS', metric='EUCLIDEAN', mu=10.0)
    
    plot_result(X, D, polyhedron, title="Initial Configuration with Boundary Intersection", grid=False, colors=colors)