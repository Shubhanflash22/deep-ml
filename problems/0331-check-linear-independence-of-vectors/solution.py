import numpy as np

def is_linearly_independent(vectors: list[list[float]]) -> bool:
    A = np.array(vectors, dtype=float)

    return np.linalg.matrix_rank(A) == len(vectors)