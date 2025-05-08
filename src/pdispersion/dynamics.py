# src/pdispersion/dynamics.py
import numpy as np

def compute_repulsive_forces(centers: np.ndarray,
                             radii: np.ndarray,
                             ε: float = 1.0) -> np.ndarray:
    """
    A simple inverse‐square repulsion between spheres:
      F_ij = ε * (r_i + r_j)^2 / ||c_i - c_j||^2  *  (c_i - c_j)/||c_i - c_j||
    """
    n = len(centers)
    forces = np.zeros_like(centers)
    for i in range(n):
        for j in range(i+1, n):
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-8:
                continue
            magnitude = ε * (radii[i] + radii[j])**2 / dist**2
            f = magnitude * (diff / dist)
            forces[i] +=  f
            forces[j] += -f
    return forces

def simulate(centers: np.ndarray,
             velocities: np.ndarray,
             radii: np.ndarray,
             dt: float,
             steps: int,
             ε: float = 1.0) -> np.ndarray:
    """
    Run a simple forward‐Euler simulation:
      v_{n+1} = v_n + dt * F(c_n)
      c_{n+1} = c_n + dt * v_{n+1}
    Returns an array of shape (steps, n, 3) storing centers over time.
    """
    history = np.empty((steps, *centers.shape))
    c = centers.copy()
    v = velocities.copy()
    for t in range(steps):
        F = compute_repulsive_forces(c, radii, ε)
        v += dt * F
        c += dt * v
        history[t] = c
    return history
