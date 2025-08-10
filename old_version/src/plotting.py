from src.point import Point
from src.face import Face
from old_container import Container
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List

def plot_container_with_spheres(container, points: List['Point'], D: float,
                                ax=None, sphere_color='red', sphere_alpha=0.3,
                                sphere_resolution=20):
    """
    Render the container and draw semi-transparent spheres of radius D around each point.
    
    :param container:    Container instance with a .render(ax, show=False) method.
    :param points:       List of Point objects (must have .as_array() -> [x,y,z]).
    :param D:            Radius of each sphere.
    :param ax:           Optional matplotlib 3D axis; created if None.
    :param sphere_color: Color for all spheres.
    :param sphere_alpha: Transparency for sphere surfaces.
    :param sphere_resolution: Number of subdivisions in the (u,v) param grid.
    :return:             The matplotlib 3D axis with everything drawn.
    """
    # 1) Set up axes
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
    # 2) Draw the container
    container.render(ax=ax, show=False)
    
    # 3) Precompute parametric grid for a unit sphere
    u = np.linspace(0, 2*np.pi, sphere_resolution)
    v = np.linspace(0,   np.pi, sphere_resolution)
    u_grid, v_grid = np.meshgrid(u, v)
    # unit sphere mesh
    x_unit = np.cos(u_grid) * np.sin(v_grid)
    y_unit = np.sin(u_grid) * np.sin(v_grid)
    z_unit = np.cos(v_grid)
    
    # 4) For each point, shift & scale the sphere mesh and plot
    for pt in points:
        x0, y0, z0 = pt.as_array()
        x = x0 + D * x_unit
        y = y0 + D * y_unit
        z = z0 + D * z_unit
        ax.plot_surface(x, y, z,
                        color=sphere_color,
                        alpha=sphere_alpha,
                        linewidth=0,
                        shade=True)
    
    # 5) Equal aspect ratio so spheres aren’t distorted
    try:
        ax.set_box_aspect([1,1,1])
    except AttributeError:
        # older matplotlib: manually scale axes
        pts = np.array([v.as_array() for v in container.vertices])
        mins = pts.min(axis=0); maxs = pts.max(axis=0)
        ranges = maxs - mins
        max_range = ranges.max() / 2
        mid = (maxs + mins) / 2
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    plt.show()
    return ax
