"""A small orthographic ray caster for packing figures.

matplotlib's 3-D axes sort whole objects by their centroids, so in a dense
packing an obstacle drawn inside the container is painted over every ball, or
under all of them. Here every pixel finds its own nearest surface, so balls
and the container's removed pieces occlude each other correctly. Pure numpy;
the 2-D point-in-polygon test is matplotlib's Path.

    img, cam = render(X, r, container, colors)       # (h, w, 3) floats in [0, 1]
    ax.imshow(img); draw_edges(ax, container, cam)    # walls as lines, hidden ones faint

Balls are Euclidean. The container's own walls are drawn as a faint tint, the
faces of removed pieces (face descriptions starting "piece", as the Stoyan &
Yaskov containers name them) as opaque grey solids.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.path import Path

LIGHT = np.array([-0.35, -0.55, 0.76])
OBSTACLE = "#8f8d86"
WALL_TINT = "#4169e1"


def camera(container, elev: float, azim: float, width: int, pad: float = 0.03) -> Dict:
    """Orthographic camera looking at the container from (elev, azim), in
    degrees, with matplotlib's convention; the image fits its bounding box."""
    el, az = np.radians(elev), np.radians(azim)
    w = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])  # towards the eye
    right = np.cross([0.0, 0.0, 1.0], w)
    right /= np.linalg.norm(right)
    up = np.cross(w, right)
    V = np.asarray([v.as_array() for f in container.faces for v in f.vertices], dtype=float)
    u, v = V @ right, V @ up
    span = max(np.ptp(u), np.ptp(v)) * (1 + 2 * pad)
    cu, cv = (u.min() + u.max()) / 2, (v.min() + v.max()) / 2
    hu, hv = np.ptp(u) * (1 + 2 * pad), np.ptp(v) * (1 + 2 * pad)
    px = span / width
    W, H = int(np.ceil(hu / px)), int(np.ceil(hv / px))
    return {"right": right, "up": up, "w": w, "px": px, "W": W, "H": H,
            "u0": cu - W * px / 2, "v0": cv + H * px / 2, "far": float((V @ w).max()) + 10.0}


def _rays(cam):
    j, i = np.meshgrid(np.arange(cam["W"]), np.arange(cam["H"]))
    u = cam["u0"] + (j + 0.5) * cam["px"]
    v = cam["v0"] - (i + 0.5) * cam["px"]
    O = (u[..., None] * cam["right"] + v[..., None] * cam["up"] + cam["far"] * cam["w"]).reshape(-1, 3)
    return O, -cam["w"]


def _polygon_hits(O, d, face):
    """Ray parameter t at which each ray meets the face (inf where it misses)."""
    n = np.asarray(face.normal.as_array() if hasattr(face.normal, "as_array") else face.normal, float)
    n = n / np.linalg.norm(n)
    P = np.asarray([v.as_array() for v in face.vertices], dtype=float)
    nd = n @ d
    t = np.full(len(O), np.inf)
    if abs(nd) < 1e-12:
        return t, n
    tt = ((P[0] - O) @ n) / nd
    Q = O + tt[:, None] * d
    a = P[1] - P[0]
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    to2 = lambda pts: np.stack([(pts - P[0]) @ a, (pts - P[0]) @ b], axis=1)
    inside = Path(to2(P)).contains_points(to2(Q))
    for hole in getattr(face, "holes", None) or []:
        H = np.asarray([v.as_array() for v in hole], dtype=float)
        inside &= ~Path(to2(H)).contains_points(to2(Q))
    t[inside & (tt > 0)] = tt[inside & (tt > 0)]
    return t, n


def _shade(base, n, d, ambient=0.38, diffuse=0.55, spec=0.18):
    L = LIGHT / np.linalg.norm(LIGHT)
    lam = np.abs(n @ L)[:, None]
    h = L - d
    h /= np.linalg.norm(h)
    s = np.clip(n @ h, 0, 1)[:, None] ** 30
    return np.clip(base * (ambient + diffuse * lam) + spec * s, 0, 1)


def render(X, r: float, container, colors: Sequence, *, elev: float = 22.0, azim: float = -60.0,
           width: int = 900, supersample: int = 2, wall_alpha: float = 0.05,
           obstacle: str = OBSTACLE, background: str = "#ffffff"):
    """Image of balls of radius r at X inside the container, and its camera.

    Rendered at `supersample` times the size and averaged down. Returns
    (image (H, W, 3), cam, depth (H, W)); cam and depth let draw_edges hide the
    walls' edges behind balls and obstacles."""
    X = np.asarray(X, dtype=float)
    cam = camera(container, elev, azim, width * supersample)
    O, d = _rays(cam)
    N = len(O)
    t_best = np.full(N, np.inf)
    rgb = np.tile(np.asarray(to_rgb(background)), (N, 1))
    normal = np.zeros((N, 3))
    base = np.zeros((N, 3))

    cols = np.asarray([to_rgb(c) for c in colors])
    for c, col in zip(X, cols):
        oc = c - O
        tc = oc @ d
        q2 = np.einsum("ij,ij->i", oc, oc) - tc ** 2
        hit = q2 < r * r
        t = np.where(hit, tc - np.sqrt(np.maximum(r * r - q2, 0.0)), np.inf)
        better = t < t_best
        if better.any():
            t_best[better] = t[better]
            P = O[better] + t[better, None] * d
            normal[better] = (P - c) / r
            base[better] = col

    walls = []
    obs = np.asarray(to_rgb(obstacle))
    for f in container.faces:
        t, n = _polygon_hits(O, d, f)
        if (f.description or "").startswith("piece"):
            better = t < t_best
            t_best[better] = t[better]
            normal[better] = n
            base[better] = obs
        else:
            walls.append(t)

    hit = np.isfinite(t_best)
    rgb[hit] = _shade(base[hit], normal[hit], d)
    # Glass walls: every wall in front of the first opaque surface tints it.
    if walls and wall_alpha > 0:
        k = np.zeros(N)
        for t in walls:
            k += t < t_best
        a = 1 - (1 - wall_alpha) ** k
        rgb = rgb * (1 - a[:, None]) + np.asarray(to_rgb(WALL_TINT)) * a[:, None]

    H, W, s = cam["H"], cam["W"], supersample
    img = rgb.reshape(H, W, 3)
    depth = t_best.reshape(H, W)
    H2, W2 = H // s, W // s
    img = img[:H2 * s, :W2 * s].reshape(H2, s, W2, s, 3).mean(axis=(1, 3))
    depth = depth[:H2 * s, :W2 * s].reshape(H2, s, W2, s).min(axis=(1, 3))
    cam = dict(cam, px=cam["px"] * s, W=W2, H=H2)
    return img, cam, depth


def _edges(container) -> List[np.ndarray]:
    """Wall edges, without the cuts that split faces into hole-free pieces: a
    segment two coplanar faces share is interior to the wall, not an edge."""
    seen: Dict[tuple, list] = {}
    for f in container.faces:
        n = np.asarray(f.normal.as_array() if hasattr(f.normal, "as_array") else f.normal, float)
        n = n / np.linalg.norm(n)
        rings = [f.vertices] + list(getattr(f, "holes", None) or [])
        for ring in rings:
            P = np.asarray([v.as_array() for v in ring], dtype=float)
            for a, b in zip(P, np.roll(P, -1, axis=0)):
                key = tuple(sorted([tuple(np.round(a, 6)), tuple(np.round(b, 6))]))
                seen.setdefault(key, []).append(n)
    out = []
    for (a, b), ns in seen.items():
        coplanar = len(ns) >= 2 and all(abs(abs(ns[0] @ m) - 1) < 1e-9 for m in ns[1:])
        if not coplanar:
            out.append(np.array([a, b]))
    return out


def draw_edges(ax, container, cam, depth, color: str = "#2f3a8f", lw: float = 0.5,
               hidden_alpha: float = 0.18, samples: int = 200):
    """Wall edges over the image: solid where visible, faint where a ball or
    obstacle is in front."""
    tol = 2.0 * cam["px"]
    for e in _edges(container):
        s = np.linspace(0, 1, samples)[:, None]
        P = e[0] + s * (e[1] - e[0])
        j = (P @ cam["right"] - cam["u0"]) / cam["px"] - 0.5
        i = (cam["v0"] - P @ cam["up"]) / cam["px"] - 0.5
        t = cam["far"] - P @ cam["w"]
        ii = np.clip(np.round(i).astype(int), 0, cam["H"] - 1)
        jj = np.clip(np.round(j).astype(int), 0, cam["W"] - 1)
        vis = t <= depth[ii, jj] + tol
        # Split into runs of equal visibility and draw each.
        cuts = np.flatnonzero(np.diff(vis.astype(int))) + 1
        for run in np.split(np.arange(samples), cuts):
            if len(run) < 2:
                continue
            ax.plot(j[run], i[run], color=color, lw=lw,
                    alpha=1.0 if vis[run[0]] else hidden_alpha, solid_capstyle="round")
