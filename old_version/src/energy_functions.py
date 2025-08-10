import numpy as np
from old_version.src.container import Container

# penalty factors
alpha = 1.0
gamma = 2.0
lamb = 0.5

def lij(Xi, Xj, D):
    return max((2 * D) - np.linalg.norm(Xi - Xj), 0)

def Oi0(Xi, D, container):
    vertices_and_footpoints = container.vertices + container.get_active_footpoints(Xi)
    if container.contains(Xi, num_rays=10):
        return sum(max(0, D - np.linalg.norm(Xi - v)) ** 2 for v in vertices_and_footpoints)
    else:
        return gamma * (D + min([np.linalg.norm(Xi - v) for v in vertices_and_footpoints])) ** 2


# D: allowed minimum distance between two points
# container: Ω, the container of the multiply connected domain
# X: candidate solution of p points, 
#   X = [(x_1, y_1), (x_2, y_2), ..., (x_p, y_p)] 
#   where (x_i, y_i) is a Point object
def candidate_quality(X, D, container, flag=False):
    X = X.reshape(-1, 2)  # Ensure X is a 2D array with shape (p, 2)
    dispersion_point_violation = 0
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            dispersion_point_violation += (lij(X[i], X[j], D)) ** 2
    container_violation = sum(Oi0(Xi, D, container) for Xi in X)
    if flag:
        print(f"Dispersion Point Violation: {dispersion_point_violation}")
        print(f"Container Violation: {container_violation}")
    return dispersion_point_violation + alpha * container_violation

def point_energy(X, i, D, container):
    point = X[i]
    dispersion_violation = sum(lij(point, X[j], D) ** 2 for j in range(len(X)) if j != i)
    container_violation = Oi0(point, D, container)
    return dispersion_violation + alpha * container_violation

def vacancy_energy(X, vacancy_point, D, container):
    dispersion_penalty = sum(lij(vacancy_point, X[j], D) ** 2 for j in range(len(X)))
    container_penalty = Oi0(vacancy_point, D, container)
    return dispersion_penalty + alpha * container_penalty

def E(X, D, container):
    dispersion_penalty = 0
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            dispersion_penalty += lij(X[i], X[j], D) ** 2
    container_penalty = sum(Oi0(Xi, D, container) for Xi in X)
    return dispersion_penalty + alpha * container_penalty

def phi(µ, X, D, container):
    return -((2 * D) ** 2) + µ * E(X, D, container)

