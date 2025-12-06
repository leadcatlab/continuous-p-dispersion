from tsgo.visualizer import plot_from_file

from tsgo.sample_polyhedron import cube, h_box, triangular_pyramid, pointy_star

if __name__ == "__main__":
    filepath = "output/euclidean/cube_with_radius_1.0/p=10/iterations=10/D_init=0.1/TSGO_10_data.pkl"
    container = pointy_star()
    plot_from_file(filepath)