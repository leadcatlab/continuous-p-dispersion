import argparse
import pickle
from typing import List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tsgo.container.point import Point
from tsgo.container.container import Container
from tsgo.algorithm import energy_functions as ef


# Default light direction, in data coordinates: above, front-left. Shading is
# computed from surface normals rather than left to matplotlib's own shading,
# which is washed out by the transparency these plots need to show overlap.
_LIGHT_DIR = np.array([-0.35, -0.55, 0.75])


def _shaded_facecolors(xs, ys, zs, base_color, alpha,
                       ambient=0.42, diffuse=0.58, light=None, normals=None):
    """Per-facet Lambertian colours for a plot_surface grid.

    Returns an (nv, nu, 4) RGBA array. Normals come from the grid itself, so
    this works for any surface the metrics produce -- the round ball of l_2,
    the octahedron of l_1, the cube of l_inf -- rather than assuming a sphere.

    The diffuse term uses |n . L|: these surfaces are translucent and the far
    side is visible through the near one, so a facet pointing away from the
    light should still be lit rather than going flat black.
    """
    from matplotlib.colors import to_rgba

    P = np.stack([np.asarray(xs), np.asarray(ys), np.asarray(zs)], axis=-1)
    if normals is not None:
        n = np.asarray(normals, dtype=float)[:-1, :-1, :]
    else:
        du = P[:-1, 1:, :] - P[:-1, :-1, :]
        dv = P[1:, :-1, :] - P[:-1, :-1, :]
        n = np.cross(du, dv)
        ln = np.linalg.norm(n, axis=-1, keepdims=True)
        n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)
        # A degenerate quad -- the poles of a sphere parameterisation, where
        # the mesh collapses to a point -- has no normal, and leaving it at
        # zero renders an ambient-only dark ring at each pole. Borrow the
        # neighbouring row's normal there.
        bad = np.linalg.norm(n, axis=-1) == 0.0
        if bad.any():
            for i in range(n.shape[0]):
                if not bad[i].any():
                    continue
                src = i - 1 if i > 0 else min(i + 1, n.shape[0] - 1)
                n[i][bad[i]] = n[src][bad[i]]

    L = np.asarray(_LIGHT_DIR if light is None else light, dtype=float)
    L = L / np.linalg.norm(L)
    lam = np.abs(np.einsum("ijk,k->ij", n, L))
    shade = ambient + diffuse * lam

    base = np.asarray(to_rgba(base_color), dtype=float)
    rgba = np.empty(shade.shape + (4,), dtype=float)
    rgba[..., :3] = np.clip(base[:3] * shade[..., None], 0.0, 1.0)
    rgba[..., 3] = alpha

    # plot_surface wants facecolors shaped like the grid; pad the last row and
    # column, which it discards.
    out = np.empty(P.shape[:2] + (4,), dtype=float)
    out[:-1, :-1] = rgba
    out[-1, :] = out[-2, :]
    out[:, -1] = out[:, -2]
    return out


def _surface(ax, xs, ys, zs, col, alpha, shading):
    """plot_surface with per-facet shading when asked, flat colour otherwise."""
    if shading:
        ax.plot_surface(xs, ys, zs, linewidth=0, antialiased=True,
                        facecolors=_shaded_facecolors(xs, ys, zs, col, alpha),
                        shade=False)
    else:
        ax.plot_surface(xs, ys, zs, linewidth=0, antialiased=True,
                        alpha=alpha, color=col)


def plot_result(
    points_xyz: np.ndarray, 
    D: float, 
    container: Container, 
    *, 
    colors=None,
    title=None,
    face_colors: Optional[list] = None,
    sphere_u: int = 48,
    sphere_v: int = 32,
    face_alpha: float = 0.08, 
    face_edge_alpha: float = 0.35,
    sphere_alpha: float = 0.35,
    show: bool = True, ax=None, 
    show_normals: bool = False, 
    metric_name: str = 'EUCLIDEAN', 
    grid: bool = True,
    boundary_limits: Optional[tuple] = None,
    seed: int = 0,
    shading: bool = True,
) -> plt.Axes:
    """
    Plot a polyhedral container and p spheres (centers given by points_xyz) with radius D.

    Args:
        points_xyz: np.ndarray of shape (p, 3) containing sphere centers.
        D: float, sphere radius.
        container: Container object with `faces` (each a Face with `vertices` = List[Point]).
        colors: Optional. If None (default), colors are assigned from a colormap according to point index.
                If a string, it's treated as a matplotlib colormap name and used to color by index.
                If an iterable, it must provide one color spec per point (length p).
        face_colors: Optional[list]. A list of face indices to color red. Faces not in this list are colored blue.
        sphere_u: longitudinal resolution of sphere mesh.
        sphere_v: latitudinal resolution of sphere mesh.
        face_alpha: fill transparency for container faces.
        face_edge_alpha: transparency for face wireframe.
        sphere_alpha: fill transparency for spheres.
        show: whether to call plt.show().
        ax: optional existing 3D axis to draw into.
        show_normals: whether to plot face normals as red arrows.
        metric_name: the name of the metric to use for sphere plotting.
        grid: whether to show grid lines, axes labels, and title.

    Returns:
        The matplotlib Axes3D instance used.
    """
    # --- validate inputs ---
    points_xyz = np.asarray(points_xyz, dtype=float)
    assert points_xyz.ndim == 2 and points_xyz.shape[1] == 3, "points_xyz must be (p,3)"
    assert isinstance(D, (int, float)) and D >= 0, "D must be a non-negative float"

    p = points_xyz.shape[0]

    # --- resolve colors per point ---
    if isinstance(colors, str):
        cmap = plt.cm.get_cmap(colors)
        colors_list = [cmap(0.5 + 0.5 * i / (p - 1) if p > 1 else 0.5) for i in range(p)]
    elif colors is None:
        # default colormap: use a qualitative colormap if few points, otherwise a continuous one
        cmap_name = 'tab20' if p <= 20 else 'viridis'
        cmap = plt.cm.get_cmap(cmap_name)
        colors_list = [cmap(0.5 + 0.5 * i / (p - 1) if p > 1 else 0.5) for i in range(p)]
    else:
        # assume iterable of color specs
        colors_list = list(colors)
        if len(colors_list) != p:
            raise ValueError(f"colors must have length equal to number of points ({p}), got {len(colors_list)}")

    # --- make axis ---
    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection='3d')
        created_fig = True

    # --- helper: add a sphere mesh at center c with radius r and color col ---
    def add_sphere(c, r, col, metric: str = 'EUCLIDEAN'):
        if metric == 'EUCLIDEAN' or metric == 'N_NORM_2':
            # Parameterization
            u = np.linspace(0, 2*np.pi, sphere_u)
            v = np.linspace(0, np.pi, sphere_v)
            uu, vv = np.meshgrid(u, v)
            xs = r * np.cos(uu) * np.sin(vv) + c[0]
            ys = r * np.sin(uu) * np.sin(vv) + c[1]
            zs = r * np.cos(vv) + c[2]
            if shading:
                # Exact outward normals for the ball: no pole degeneracy.
                nrm = np.stack([(xs - c[0]) / r, (ys - c[1]) / r, (zs - c[2]) / r],
                               axis=-1) if r > 0 else None
                fc = _shaded_facecolors(xs, ys, zs, col, sphere_alpha, normals=nrm)
                ax.plot_surface(xs, ys, zs, linewidth=0, antialiased=True,
                                facecolors=fc, shade=False)
            else:
                ax.plot_surface(xs, ys, zs, linewidth=0, antialiased=True,
                                alpha=sphere_alpha, color=col)
        elif metric == 'TAXICAB' or metric == 'N_NORM_1':
            # One triangular face in the first octant: x, y, z >= 0 and x + y + z = r
            u = np.linspace(0, 1, sphere_u)
            v = np.linspace(0, 1, sphere_u)
            uu, vv = np.meshgrid(u, v)

            # Keep only points where u + v <= 1
            mask = (uu + vv) <= 1
            uu = uu[mask]
            vv = vv[mask]

            # Base face
            x0 = r * uu
            y0 = r * vv
            z0 = r * (1 - uu - vv)

            # Eight outward faces generated by sign flips
            signs = [
                (1, 1, 1),
                (1, 1, -1),
                (1, -1, 1),
                (1, -1, -1),
                (-1, 1, 1),
                (-1, 1, -1),
                (-1, -1, 1),
                (-1, -1, -1),
            ]

            for sx, sy, sz in signs:
                xs = c[0] + sx * x0
                ys = c[1] + sy * y0
                zs = c[2] + sz * z0

                # trisurf has no (u, v) grid to take normals from, so use
                # matplotlib's own light source here.
                from matplotlib.colors import LightSource
                ax.plot_trisurf(
                    xs, ys, zs,
                    linewidth=0,
                    antialiased=True,
                    alpha=sphere_alpha,
                    color=col,
                    shade=shading,
                    lightsource=LightSource(azdeg=315, altdeg=50) if shading else None,
                )

            ax.set_box_aspect([1, 1, 1])
        elif metric == 'SUP_NORM':
            cx, cy, cz = c  # center
            u = np.linspace(-r, r, sphere_u)
            v = np.linspace(-r, r, sphere_v)
            uu, vv = np.meshgrid(u, v)

            # 6 faces of the cube: x = ±r, y = ±r, z = ±r (relative to center)

            # x = cx + r (right face)
            xs = np.full_like(uu, cx + r)
            ys = cy + uu
            zs = cz + vv
            _surface(ax, xs, ys, zs, col, sphere_alpha, shading)

            # x = cx - r (left face)
            xs = np.full_like(uu, cx - r)
            ys = cy + uu
            zs = cz + vv
            _surface(ax, xs, ys, zs, col, sphere_alpha, shading)

            # y = cy + r (front face)
            xs = cx + uu
            ys = np.full_like(uu, cy + r)
            zs = cz + vv
            _surface(ax, xs, ys, zs, col, sphere_alpha, shading)

            # y = cy - r (back face)
            xs = cx + uu
            ys = np.full_like(uu, cy - r)
            zs = cz + vv
            _surface(ax, xs, ys, zs, col, sphere_alpha, shading)

            # z = cz + r (top face)
            xs = cx + uu
            ys = cy + vv
            zs = np.full_like(uu, cz + r)
            _surface(ax, xs, ys, zs, col, sphere_alpha, shading)

            # z = cz - r (bottom face)
            xs = cx + uu
            ys = cy + vv
            zs = np.full_like(uu, cz - r)
            _surface(ax, xs, ys, zs, col, sphere_alpha, shading)

            ax.set_box_aspect([1, 1, 1])
        else:
            print(f"Warning: Metric '{metric}' not recognized for sphere plotting. Defaulting to sampling surface.")
            # sample points on the surface of the sphere
            num_samples = sphere_u * sphere_v
            _rng = np.random.default_rng(seed)
            phi = np.arccos(1 - 2 * _rng.random(num_samples))
            theta = 2 * np.pi * _rng.random(num_samples)
            x = r * np.sin(phi) * np.cos(theta) + c[0]
            y = r * np.sin(phi) * np.sin(theta) + c[1]
            z = r * np.cos(phi) + c[2]
            ax.scatter(x, y, z, color=col, alpha=sphere_alpha, s=1)

    # --- plot container faces (both a faint fill and a wireframe outline) ---
    face_polys = []
    face_fill_colors = []
    edge_segments = []
    edge_colors = []
    all_vertices = []

    for fi, face in enumerate(getattr(container, "faces", [])):
        # Each face.vertices is a List[Point]; turn into (n,3) array
        verts = np.array([v.as_array() for v in face.vertices], dtype=float)
        # Fill polygon
        face_polys.append(verts)

        # determine face color: red if fi in face_colors, else blue
        if face_colors is not None:
            if fi in face_colors:
                fcol = 'magenta'
            else:
                fcol = 'cyan'
        else:
            fcol = 'blue'
        face_fill_colors.append(fcol)

        # Wireframe edges (color per-edge matching face)
        for i in range(len(verts)):
            a = verts[i]
            b = verts[(i + 1) % len(verts)]
            edge_segments.append([a, b])
            edge_colors.append(fcol)

        all_vertices.append(verts)

    if face_polys:
        poly_collection = Poly3DCollection(face_polys, facecolors=face_fill_colors, alpha=face_alpha, edgecolor='none')
        ax.add_collection3d(poly_collection)
    if edge_segments:
        line_collection = Line3DCollection(edge_segments, colors=edge_colors, linewidths=0.7, alpha=face_edge_alpha)
        ax.add_collection3d(line_collection)

    # --- plot spheres at each point, using colors_list ---
    for i, c in enumerate(points_xyz):
        add_sphere(c, D, colors_list[i], metric=metric_name)

    # --- nice equal aspect limits (accounting for sphere radius) ---
    if all_vertices:
        container_pts = np.vstack(all_vertices)
    else:
        container_pts = np.zeros((1, 3))
    all_pts = np.vstack([container_pts, points_xyz]) if len(points_xyz) else container_pts

    if boundary_limits is not None:
        xmin, xmax, ymin, ymax, zmin, zmax = boundary_limits
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
    else: 
        mins = all_pts.min(axis=0) - D
        maxs = all_pts.max(axis=0) + D
        centers = (mins + maxs) / 2.0
        ranges = (maxs - mins)
        radius = 0.5 * np.max(ranges)
        ax.set_xlim(centers[0] - radius, centers[0] + radius)
        ax.set_ylim(centers[1] - radius, centers[1] + radius)
        ax.set_zlim(centers[2] - radius, centers[2] + radius)

    # --- labels and aesthetics ---
    if grid:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(f'{len(points_xyz)} spheres (r={D:g}) in container' if title is None else title)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # turn off the axes panes and lines
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.line.set_color((0., 0., 0., 0.))
        ax.yaxis.line.set_color((0., 0., 0., 0.))
        ax.zaxis.line.set_color((0., 0., 0., 0.))
    
    ax.set_box_aspect([1, 1, 1])
    ax.grid(grid)
    
    if show_normals:
        for face in container.faces:
            if face.vertices:
                v = np.mean([v.as_array() for v in face.vertices], axis=0)
                n = face.normal
                ax.quiver(*v, *n, length=0.3, color='r', linewidth=2)

    if show and created_fig:
        plt.tight_layout()
        plt.show()

    return ax

def plot_active_faces(point: Point, container: Container, inside: bool, title: str = None) -> None:
    """
    Plots the container with faces that are active for the given points highlighted.

    Args:
        points (np.ndarray): The configuration of dispersion points.
        container (Container): The container object.
    """
    active_face_indices = set()
    for index, face in enumerate(container.faces):
        if container._is_active_plane(face, point, inside):
            active_face_indices.add(index)
    
    points = point.as_array().reshape(1, 3)

    plot_result(points, D=0.05, container=container, face_colors=list(active_face_indices), show=True, grid=False, title=title)

def plot_active_footpoints(point: Point, container: Container, title: str = None) -> None:
    """
    Plots the container with active footpoints for a given point highlighted.

    Args:
        point (np.ndarray): A single dispersion point.
        container (Container): The container object.
    """
    points = np.array([[point.x, point.y, point.z]])
    footpoints = container.get_active_footpoints(point)
    footpoint_coords = np.array([fp.as_array() for fp in footpoints])
    colors = ['red'] + ['blue'] * len(footpoints)
    points = np.vstack([points, footpoint_coords]) if footpoint_coords.size > 0 else points
    
    # color the active faces pink and the rest grey
    active_face_indices = set()
    inside = container.contains(point)
    for index, face in enumerate(container.faces):
        if container._is_active_plane(face, point, inside):
            active_face_indices.add(index)


    plot_result(points, D=0.05, container=container, colors=colors, face_colors=active_face_indices, show=True, title=title, grid=False)

def plot_from_file(
    file_path: str, 
    boundary_limits: tuple, 
    colors: Union[str, List[str]] = None, 
    grid: bool = False, 
    metric: Optional[str] = None
) -> None:
    """
    Plots the 3D configuration of points from a pickle file.
    Compatible with both legacy (optimized points/D) and updated (X_opt/D_opt) keys.
    """
    # 1. Load the pickle object
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # 2. Extract radius using fallback logic
    # Checks for updated 'D_opt' first, then legacy 'D'
    D = data.get("D_opt") if "D_opt" in data else data.get("D")
    
    # 3. Extract points using fallback logic
    # Checks for updated 'X_opt' first, then legacy 'optimized points'
    points = data.get("X_opt") if "X_opt" in data else data.get("optimized points")
    
    # 4. Extract metadata and container info
    container = data.get("container")
    p = data.get("p")
    p_iters = data.get("parallel_iterations", data.get("iterations", "N/A"))
    
    # 5. Diagnostic Output
    # The energy function E identifies if the solution is feasible (E=0) [cite: 41, 119]
    print(f"--- Configuration for p={p} ---")
    print(f"Radius: {D}")
    print(f"Global/Parallel Iterations: {p_iters}")
    print(f"Final Energy: {ef.E(points, D, container)}")

    # 6. Handle point coloring for p points [cite: 70]
    if isinstance(colors, str):
        colors = [colors for _ in range(p)]
    elif colors is None:
        colors = ['blue' for _ in range(p)]

    # 7. Execute 3D visualization
    # This renders the dispersion points within the polyhedral container [cite: 7, 75]
    print(f"Plotting... (Grid: {grid})")
    
    plot_result(
        points, 
        D, 
        container=container, 
        metric_name=metric or data['metric'], 
        boundary_limits=boundary_limits, 
        colors=colors, 
        show=True, 
        grid=grid
    )
    
def main():
    parser = argparse.ArgumentParser(description="Plot 3D configuration of points from a file.")
    parser.add_argument("--filepath", type=str, help="Path to the file containing point data.", required=True)
    parser.add_argument("--limits", type=float, nargs=6, help="Boundary limits for the plot as xmin xmax ymin ymax zmin zmax", required=False, default=None)
    parser.add_argument("--colors", type=str, default=None, help="Color map name or comma-separated list of colors for points.")
    parser.add_argument("--grid", type=bool, default=False, help="Whether to show grid in the plot.")
    parser.add_argument("--metric", type=str, default='EUCLIDEAN', help="The metric to use for plotting spheres (EUCLIDEAN, TAXICAB, SUP_NORM).")
    args = parser.parse_args()
    
    boundary_limits = tuple(args.limits) if args.limits is not None else None
    if args.colors is not None and ',' in args.colors:
        args.colors = args.colors.split(',')
    elif args.colors is not None:
        args.colors = args.colors.strip()
    plot_from_file(args.filepath, boundary_limits=boundary_limits, colors=args.colors, grid=args.grid, metric=args.metric)
    
if __name__ == "__main__":
    main()
