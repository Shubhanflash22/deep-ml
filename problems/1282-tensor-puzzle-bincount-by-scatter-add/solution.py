import numpy as np

def bincount(a: np.ndarray, n_bins: int) -> np.ndarray:
    bins = np.arange(n_bins)

    # Compare every element against every bin
    matrix = (a[:, None] == bins[None, :])

    # Sum each column
    return np.sum(matrix, axis=0)
