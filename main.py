import sys, os
import src.energy_functions as ef

# Insert the absolute path to the src folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from tsgo_algorithm import tsgo_algorithm
from src.sample_polygons import square
import time

def main():
    start_time = time.time()
    print("Starting TSGO algorithm...")
    square_container = square(side_length=2.0)
    p = 10  # Number of points to be placed in the container
    iterations = 2  # Maximum number of iterations for the optimization process  # Packing density of the initial solution
    rho = 0.5

    X_opt, D_opt = tsgo_algorithm(p, square_container, iterations, rho)
    print("Optimal configuration of points:", X_opt)
    print("Minimum distance between points:", D_opt)
    print(f"Energy of the system: {round(ef.E(X_opt, D_opt, square_container), 20)}")

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

main()

"""
p = 2, square_container, side_length=2.0, iterations = 2, rho = 0.5
Optimal configuration of points: [[1.41421 0.58579]
[0.58579 1.41421]]
Minimum distance between points: 0.5857864551
Energy of the system: 1.0222946731e-10
Time taken: 338.93 seconds
"""

"""
p = 4, square_container, side_length=2.0, iterations = 2, rho = 0.5
Optimal configuration of points: [[0.5 1.5]
 [0.5 0.5]
 [1.5 1.5]
 [1.5 0.5]]
Minimum distance between points: 0.5000000267
Energy of the system: 1.710936e-14
Time taken: 1166.48 seconds
"""