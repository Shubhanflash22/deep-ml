import numpy as np

def roll(a: np.ndarray) -> np.ndarray:
    """Circular left shift by one."""
    n = len(a)
    out = np.empty_like(a)
    for i in range(n):
        out[i] = a[(i + 1) % n]
    return out
