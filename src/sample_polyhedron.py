import numpy as np
from src.container import Container

def square():
    vertices = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
    container = Container(vertices, description="Square Container")
    return container

def h_container():
    vertices = [[0, 0], [1, 0], [1, 1], [2, 1], [2, 0], [3, 0], [3, 3], [2, 3], [2, 2], [1, 2], [1, 3], [0, 3]]
    container = Container(vertices, description="H-Shaped Container")
    return container