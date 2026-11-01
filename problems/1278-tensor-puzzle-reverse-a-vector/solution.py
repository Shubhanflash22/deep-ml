import numpy as np

def flip(a: np.ndarray) -> np.ndarray:
    """Reverse 1-D array a without slicing a[::-1]."""
    # Your code here
    n = len(a)
    out = np.zeros(n)
    j = 0
    for i in range(n - 1, -1, -1):
        out[j] = a[i]
        j = j + 1
    return out
