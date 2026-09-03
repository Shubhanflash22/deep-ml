import numpy as np

def row_normalize(counts: list[list[float]]) -> list[list[float]]:
    """Convert a count matrix into a row-stochastic probability matrix."""
    n = len(counts)
    m = len(counts[0])
    for i in range(0,n):
        temp = np.sum(counts[i])
        if(temp>0):
            counts[i] = counts[i]/temp
    return counts