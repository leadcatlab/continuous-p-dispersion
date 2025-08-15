from src.sample_polygons import square
import src.tsgo_algorithm as tsgo
import time

def test_adjust_distance():
    square_container = square(side_length=2.0)
    X0, D0 = square_container.generate_initial_solution(1), 0.5
    start_time = time.time()
    X, D = tsgo.adjust_distance(X0, D0, square_container)
    end_time = time.time()
    assert X is not None
    assert D is not None
    print(f"Adjust Distance Time: {end_time - start_time} seconds")