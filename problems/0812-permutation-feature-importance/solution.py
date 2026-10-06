import numpy as np

def permutation_importance(X, y, weights, bias, permutations):
    """
    Compute permutation feature importance for a linear regression model.

    Args:
        X: 2D array-like of shape (n_samples, n_features)
        y: 1D array-like of shape (n_samples,)
        weights: 1D array-like of shape (n_features,)
        bias: float, the intercept
        permutations: list of length n_features; permutations[j] is a list of
                      permutation index arrays to apply to column j

    Returns:
        List of feature importances (length n_features).
    """
    X = np.array(X)
    y = np.array(y)
    weights = np.array(weights)
    baseline_pred = X @ weights + bias
    def r2_score(y, pred):
        ss_res = np.sum((y - pred) **2)
        ss_tot = np.sum((y - np.mean(y)) **2)
        return 1 - (ss_res / ss_tot)
    baseline_r2 = r2_score(y, baseline_pred)
    importances = []
    for j in range(X.shape[1]):
        scores = []

        for perm in permutations[j]:
            X_perm = X.copy()
            X_perm[:, j] = X[perm, j]

            pred = X_perm @ weights + bias
            scores.append(r2_score(y, pred))

        importances.append(
            baseline_r2 - np.mean(scores)
        )

    return importances
