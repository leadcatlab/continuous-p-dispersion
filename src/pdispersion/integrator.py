import numpy as np

def euler_step(positions: np.ndarray, velocities: np.ndarray, dt: float):
    """Simple forward‐Euler: x_{n+1} = x_n + dt * v_n"""
    return positions + dt * velocities
