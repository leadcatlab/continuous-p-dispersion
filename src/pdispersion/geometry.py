import numpy as np

class Sphere:
    def __init__(self, center: np.ndarray, radius: float):
        self.center = np.asarray(center, dtype = float)
        self.radius = float(radius)

    def distance_to(self, other: "Sphere") -> float:
        return np.linalg.norm(self.center - other.center) - (self.radius + other.radius)
