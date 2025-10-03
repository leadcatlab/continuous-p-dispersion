import sys, os

from tsgo.point import Point
import tsgo.energy_functions as ef
from tsgo.visualizer import plot_from_file

import numpy as np
from tsgo.tsgo_algorithm import tsgo_algorithm
from tsgo.sample_polyhedron import cube, h_box, triangular_pyramid, pointy_star
import time

if __name__ == "__main__":
    filepath = "output/pointy-star/p=50/iterations=20/D_init=0.33700855782957184/TSGO_50_data.pkl"
    container = pointy_star()
    plot_from_file(filepath, container=container)