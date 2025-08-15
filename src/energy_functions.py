import numpy as np
import heapq

ALPHA = 1.0
GAMMA = 2.0
LAMBDA = 0.5

# NOTE: D will symbolize the radius of the sphere centered around each dispersion point

# @param x_i: first dispersion point
# @param x_j: second dispersion point
# @param D: radius of the sphere
# @return: the penalty term for the pairwise distance between dispersion points x_i and x_j
def l_ij(x_i, x_j, D):
    return max(0, (2 * D) - np.linalg.norm(x_i - x_j))

# @param x_i: the dispersion point
# @param D: radius of the sphere
# @param container: the container object
# @return: the penalty term for the dispersion point x_i with respect to the boundary of the container
def Oi0(x_i, D, container):
    x_i = np.array(x_i)
    vertices_and_footpoints = np.array(container.vertices_as_list + container.get_active_footpoints(x_i, as_numpy=False))
    if container.contains(x_i):
        return sum(max(0, D - np.linalg.norm(x_i - v)) ** 2 for v in vertices_and_footpoints)
    else:
        return GAMMA * (D + min(np.linalg.norm(x_i - v) for v in vertices_and_footpoints)) ** 2

# @param X: the set of dispersion points
# @param D: radius of the sphere
# @param container: the container object
# @param verbose: whether to print detailed information
# @return: the total energy of the system (holding D constant)
def E(X, D, container, verbose=False):
    dispersion_penalty = sum(l_ij(X[i], X[j], D) ** 2 for i in range(len(X)) for j in range(i + 1, len(X)))
    container_penalty = sum(Oi0(Xi, D, container) for Xi in X)
    
    if verbose:
        print("Dispersion Penalty:", dispersion_penalty)
        print("Container Penalty:", container_penalty)

    return dispersion_penalty + ALPHA * container_penalty

# @param X: the set of dispersion points
# @param i: the index of the point to calculate the energy
# @param D: the radius of the sphere
# @param container: the container object
# @return: the energy of the point X[i] within the system
def point_energy(X, i, D, container):
    dispersion_penalty = sum(l_ij(X[i], X[j], D) ** 2 for j in range(len(X)) if j != i)
    container_penalty = Oi0(X[i], D, container)
    return dispersion_penalty + ALPHA * container_penalty

# @param X: the set of dispersion points
# @param vacancy_point: the vacancy point
# @param D: the radius of the sphere
# @param container: the container object
# @return: the energy of the vacancy point within the system
def vacancy_energy(X, vacancy_point, D, container):
    dispersion_penalty = sum(l_ij(vacancy_point, X[j], D) ** 2 for j in range(len(X)))
    container_penalty = Oi0(vacancy_point, D, container)
    return dispersion_penalty + ALPHA * container_penalty

# @param X: the set of dispersion points
# @param D: the radius of the sphere
# @param mu: the chemical potential
# @param container: the container object
# @return: the grand potential of the system
def phi(X, D, mu, container):
    return (-np.power(D, 2, dtype=float)) + mu * E(X, D, container)


def select_high_energy_points(X, D, container, Q):
    energies = []
    for i in range(len(X)):
        heapq.heappush(energies, (-point_energy(X, i, D, container), i))
    high_energy_points = [heapq.heappop(energies)[1] for _ in range(Q)]
    return high_energy_points

def find_vacancy_sites(X, D, container, Q):
    random_points = [container._sample_random_point() for _ in range(5 * len(X))]
    random_points = [(point, vacancy_energy(X, point, D, container)) for point in random_points]

    # create a heap with the lowest Q energies
    random_points.sort(key=lambda x: x[1])
    return np.array([point for point, _ in random_points[:Q]])


    # Return the vacancy sites sorted by energy
    vacancy_sites.sort(key=lambda x: x[0])
    return np.array([site[1] for site in vacancy_sites[:Q]])