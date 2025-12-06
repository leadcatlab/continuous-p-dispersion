import numpy as np

from tsgo.point import Point
from tsgo.sample_polyhedron import cube, h_box, pointy_star
from tsgo.visualizer import plot_active_faces, plot_active_footpoints, plot_result
from tsgo.cli import run_tsgo

if __name__ == "__main__":
    for i in range(4, 11):
        run_tsgo(p=i, container=cube(), iterations=10, D_init=0.1, metric_name='EUCLIDEAN', save=True, plot=False, num_workers=6, optimization_method="BFGS", Q=3, beta_max=5, eps=1e-16)