import numpy as np


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares with an intercept.

    Args:
        X (np.ndarray): (n, p) design matrix without an intercept column.
        y (np.ndarray): (n,) target.

    Returns:
        np.ndarray: (p + 1,) coefficients, intercept first.
    """
    X_design = np.column_stack([np.ones(len(X)), X])
    beta,_,_,_ = np.linalg.lstsq(X_design, y, rcond = None)
    return beta


def omitted_variable_bias(X: np.ndarray, y: np.ndarray, omit_idx: int) -> tuple:
    """Return (full_kept, short, bias) for a two-column X."""
    # Which column are we keeping?
    keep_idx = 1 - omit_idx

    # Full model: y ~ x1 + x2
    full = ols(X, y)

    # Coefficient on the kept variable
    full_kept = full[keep_idx + 1]

    # Short model: y ~ kept variable
    short = ols(X[:, [keep_idx]], y)
    short_coef = short[1]

    # Auxiliary regression:
    # omitted ~ kept
    auxiliary = ols(X[:, [keep_idx]], X[:, omit_idx])
    delta = auxiliary[1]

    # Omitted-variable bias = beta_omitted * delta
    beta_omitted = full[omit_idx + 1]
    bias = beta_omitted * delta

    return full_kept, short_coef, bias
