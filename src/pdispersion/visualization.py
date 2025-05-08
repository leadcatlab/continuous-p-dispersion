# src/pdispersion/visualization.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D       # noqa: F401
from matplotlib import animation

def plot_spheres(ax, centers: np.ndarray, radii: np.ndarray, **kwargs):
    """
    Quick “fake” spheres by scaling scatter‐marker size.
    kwargs go to ax.scatter (e.g. color='C0', alpha=0.6).
    """
    # marker area ~ (radius)**2
    sizes = (radii * 200)**2
    ax.scatter(centers[:,0], centers[:,1], centers[:,2], s=sizes, **kwargs)

def animate_simulation(history: np.ndarray,
                       radii: np.ndarray,
                       interval: int = 50) -> animation.FuncAnimation:
    """
    Build a FuncAnimation of your simulation history.
    history.shape == (steps, n, 3)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # set fixed axes limits
    M = np.max(np.abs(history)) * 1.1
    ax.set_xlim(-M, M); ax.set_ylim(-M, M); ax.set_zlim(-M, M)

    scat = ax.scatter([], [], [], s=[])

    def init():
        scat._offsets3d = ([], [], [])
        scat.set_sizes([])
        return (scat,)

    def update(frame):
        C = history[frame]
        scat._offsets3d = (C[:,0], C[:,1], C[:,2])
        scat.set_sizes((radii * 200)**2)
        return (scat,)

    ani = animation.FuncAnimation(
        fig, update, frames=history.shape[0],
        init_func=init, blit=False, interval=interval
    )
    return ani
