import numpy as np

from tsgo.point import Point
from tsgo.sample_polyhedron import cube, h_box, pointy_star, triangular_pyramid
from tsgo.visualizer import plot_active_faces, plot_active_footpoints, plot_result
from tsgo.cli import run_tsgo

if __name__ == "__main__":
    for container in [pointy_star(), h_box()]:
        for i in range(1, 11):
            run_tsgo(p=i, container=container, iterations=10, D_init=0.1, metric_name='EUCLIDEAN', save=True, plot=False, num_workers=10, optimization_method="L-BFGS-B", Q=3, beta_max=5, eps=1e-16)