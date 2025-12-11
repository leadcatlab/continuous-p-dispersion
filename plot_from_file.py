from tsgo.visualizer import plot_from_file

from tsgo.sample_polyhedron import cube, h_box, triangular_pyramid, pointy_star

if __name__ == "__main__":
    filepath = "output/euclidean/cube_with_radius_0.5/p=1/TSGO_1_0.4_cube_with_radius_0.5_data.pkl"
    container = pointy_star()
    plot_from_file(filepath)