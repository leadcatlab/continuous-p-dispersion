from src.tsgo_algorithm import tsgo_algorithm, tabu_search, adjust_distance
from src.tsgo_helpers import is_feasible
from src.sample_polyhedron import cube, h_box

def test_is_feasible():
    # generate 100 random points in the cube container
    container = cube()
    points = [container.random_point(as_array=True) for _ in range(100)]
    D = 0.001  # Set a small distance to check feasibility
    assert is_feasible(points, D, container), "The points should be feasible within the container."


def test_adjust_distance():
    container = cube()
    initial_solution = container.generate_initial_solution(1)
    print(f"Initial Solution: {initial_solution}")
    initial_distance = 0.01

    X, D = adjust_distance(initial_solution, initial_distance, container, max_iter=15)
    assert len(X) == 1, "The number of points in the solution should remain the same."
    assert D > 0, "The distance D should be positive."
    assert D >= initial_distance, "The distance D should be at least the initial distance."
    assert D < 1, "The distance D should be less than 1 to ensure points are within the container."
    assert is_feasible(X, D, container), "The adjusted solution should be feasible."
    print("Adjusted value of D:", D)
    

