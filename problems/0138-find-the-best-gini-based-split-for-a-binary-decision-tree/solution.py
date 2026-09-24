import numpy as np
from typing import Tuple

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""

    def gini(labels):
        if len(labels) == 0:
            return 0.0

        p = np.mean(labels)
        return 1 - p**2 - (1 - p)**2

    best_feature = -1
    best_threshold = None
    best_impurity = float("inf")

    n, d = X.shape

    for j in range(d):
        for threshold in X[:, j]:
            left = y[X[:, j] <= threshold]
            right = y[X[:, j] > threshold]

            if len(left) == 0 or len(right) == 0:
                continue

            weighted_gini = (
                len(left) / n * gini(left)
                + len(right) / n * gini(right)
            )

            if weighted_gini < best_impurity:
                best_impurity = weighted_gini
                best_feature = j
                best_threshold = threshold

    return best_feature, best_threshold