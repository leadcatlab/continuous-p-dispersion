# scripts/run_simulation.py
import numpy as np
import matplotlib.pyplot as plt

from pdispersion.dynamics      import simulate
from pdispersion.visualization import animate_simulation

# 1) your “p‐dispersion” initializer (e.g. from optimizer.py)
from pdispersion.optimizer     import max_min_dispersion

# -- example setup: n random points in [-1,1]^3, uniform radius
n = 20
rng = np.random.default_rng(0)
centers = rng.uniform(-1,1,(n,3))
radii  = np.full(n, 0.1)

# 2) optionally run your static optimizer to spread them out
centers = max_min_dispersion(centers, radii, p=2.0)

# 3) give them zero initial velocity (or random)
velocities = np.zeros_like(centers)

# 4) simulate
history = simulate(
    centers, velocities, radii,
    dt=0.005, steps=500, ε=1.0
)

# 5) animate
ani = animate_simulation(history, radii, interval=30)

# to display in a script:
plt.show()

# to save as an mp4 (requires ffmpeg):
# ani.save("dispersion.mp4", dpi=150, writer="ffmpeg")
