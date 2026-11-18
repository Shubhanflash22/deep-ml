import numpy as np

def make_diagonal(x):
    n = len(x)
    matrix = np.zeros((n, n))
    for i in range(0,n):
        matrix[i][i] = x[i]
    return matrix