import numpy as np

def map_latent_to_real(latent_labeled: list, real_labeled: list, latent_query: list) -> np.ndarray:
    """
    Fit an affine map from latent actions to real actions using a small labeled set,
    and apply it to a batch of latent queries.

    Args:
        latent_labeled: (N, d_latent) labeled latent actions.
        real_labeled:   (N, d_real)   corresponding real actions.
        latent_query:   (M, d_latent) latent actions to translate.

    Returns:
        numpy array of shape (M, d_real) with predicted real actions.
    """
    X = np.array(latent_labeled)
    Y = np.array(real_labeled)
    Q = np.array(latent_query)

    # Add column of 1s for the bias/intercept
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    Q = np.hstack([Q, np.ones((Q.shape[0], 1))])

    # Least-squares affine fit
    W = np.linalg.pinv(X) @ Y

    # Predict real actions
    return Q @ W