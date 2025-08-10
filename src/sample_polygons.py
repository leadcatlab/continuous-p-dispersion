from src.container import Container

def square(side_length: float = 1.0) -> Container:
    vertices = [ [0.0, 0.0], [side_length, 0.0], [side_length, side_length], [0.0, side_length] ]
    return Container(vertices, f"Square with side length {side_length}.")

def h_shaped_box() -> Container:
    vertices = [
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0], [2.0, 0.0], [3.0, 0.0], 
        [3.0, 3.0], [2.0, 3.0], [2.0, 2.0], [1.0, 2.0], [1.0, 3.0], [0.0, 3.0]
    ]
    return Container(vertices, "H-shaped box.")

def v_box(diag_len: float = 1.0) -> Container:
    vertices = [
        [0.0, 0.0], [diag_len, diag_len], [0.0, diag_len / 2], [-diag_len, diag_len]
    ]
    return Container(vertices, f"V-shaped box with diagonal length {diag_len}.")