
import numpy as np
from itertools import product

def polynomial_basis(s, n):
    s = np.asarray(s, dtype=float)
    k = len(s)

    # Generate exponent vectors in lexicographic order
    exponents = product(range(n + 1), repeat=k)

    # Calculate each polynomial feature
    features = [
        np.prod(s ** np.array(exp))
        for exp in exponents
    ]

    return np.array(features, dtype=float)

def gradient_mc_poly(samples, n, alpha, n_passes):
    # Initialize weights to zero
    k = len(samples[0][0])
    num_features = (n + 1) ** k
    w = np.zeros(num_features, dtype=float)

    # Repeat training passes
    for _ in range(n_passes):
        for s, G in samples:
            x = polynomial_basis(s, n)

            # Current value prediction
            v = np.dot(w, x)

            # Gradient Monte Carlo update
            w += alpha * (G - v) * x

    return w