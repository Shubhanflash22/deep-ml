
import numpy as np

def laplace_logreg(X, y, prior_var, n_iter=50):
    # Initialize weights
    N, D = X.shape
    w = np.zeros(D)
    I = np.eye(D)

    for _ in range(n_iter):
        # Compute predicted probabilities
        mu = 1.0 / (1.0 + np.exp(-X @ w))

        # Gradient of log posterior
        grad = X.T @ (y - mu) - w / prior_var

        # Hessian of negative log posterior
        S = mu * (1.0 - mu)
        H = X.T @ (S[:, None] * X) + I / prior_var

        # Newton step
        step = np.linalg.solve(H, grad)

        w_new = w + step

        # Early stopping
        if np.linalg.norm(step) < 1e-12:
            w = w_new
            break

        w = w_new

    # Compute covariance at MAP estimate
    mu = 1.0 / (1.0 + np.exp(-X @ w))
    S = mu * (1.0 - mu)
    H = X.T @ (S[:, None] * X) + I / prior_var

    Sigma = np.linalg.inv(H)

    return w, Sigma


def predictive(X_test, w_map, Sigma, method, n_samples=1000, seed=0):
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    if method == "plugin":
        return sigmoid(X_test @ w_map)

    elif method == "mc":
        rng = np.random.default_rng(seed)

        # Lower Cholesky factor
        L = np.linalg.cholesky(Sigma)

        # Draw standard normal samples
        Z = rng.standard_normal((n_samples, len(w_map)))

        # Transform samples into posterior weight samples
        W = w_map + Z @ L.T

        # Average predictions across samples
        return sigmoid(X_test @ W.T).mean(axis=1)

    elif method == "probit":
        # Predictive mean and variance
        m = X_test @ w_map
        v = np.sum((X_test @ Sigma) * X_test, axis=1)

        # Probit approximation to logistic predictive
        kappa = 1.0 / np.sqrt(1.0 + np.pi * v / 8.0)

        return sigmoid(kappa * m)

    else:
        raise ValueError("Unknown method. Use 'plugin', 'mc', or 'probit'.")
