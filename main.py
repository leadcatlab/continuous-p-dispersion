import sys, os

from src.point import Point

# Insert the absolute path to the src folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from tsgo_algorithm import tsgo_algorithm
from src.sample_polyhedron import cube, h_box
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from src.plotting import plot_container_with_spheres
import time

def main():
    start_time = time.time()
    print("Starting TSGO algorithm...")
    cube_container = cube()
    p = 4  # Number of points to be placed in the container
    iterations = 1  # Maximum number of iterations for the optimization process  # Packing density of the initial solution
    rho = 0.5

    X_opt, D_opt = tsgo_algorithm(p, cube_container, iterations, rho)
    print("Optimal configuration of points:", X_opt)
    print("Minimum distance between points:", D_opt)
    
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    # Plot the optimal configuration and the container in 3D
    plot_container_with_spheres(cube_container, [Point(*x) for x in X_opt], D_opt, sphere_color='blue', sphere_alpha=0.5, sphere_resolution=20)

main()

# p = 2, iterations = 2, rho = 0.2, container = cube()
# Optimal configuration of points: [array([ 0.366,  0.366, -0.366]), array([-0.366, -0.366,  0.366])]
# Minimum distance between points: 0.6339745867
# https://math.stackexchange.com/questions/3347006/two-sphere-inside-a-cube
# Optimal is (3 - √3) / 2 ~ 0.63397459621