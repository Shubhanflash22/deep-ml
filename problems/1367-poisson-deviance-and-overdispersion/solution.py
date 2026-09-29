import numpy as np


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Poisson deviance, using the convention 0 * log(0) = 0."""
    y = np.asarray(y)
    mu = np.asarray(mu)

    # Only evaluate log(y / mu) where y > 0
    log_term = np.where(
        y > 0,
        y * np.log(np.where(y > 0, y / mu, 1.0)),
        0.0
    )

    return float(2 * np.sum(log_term - (y - mu)))


def dispersion_ratio(y: np.ndarray, mu: np.ndarray, n_params: int) -> float:
    """Pearson chi-square divided by (n - n_params)."""
    y = np.asarray(y)
    mu = np.asarray(mu)

    pearson_chi2 = np.sum((y - mu) ** 2 / mu)
    return float(pearson_chi2 / (len(y) - n_params))