import numpy as np

def incremental_pca(batches, n_components):
    # batches: list of batches of samples (list of list of list of floats)
    # n_components: int, number of principal components to return
    # Returns: list of lists, shape (n_components, n_features)
    N = 0
    mean = None
    scatter = None
    for batch in batches:
        X = np.array(batch, dtype = float)
        if X.size == 0:
            continue

        m = X.shape[0]
        batch_mean = np.mean(X, axis = 0)

        centered = X - batch_mean
        batch_scatter = centered.T @ centered

        if N == 0:
            mean = batch_mean
            scatter = batch_scatter
            N = m
        else:
            delta = batch_mean - mean
            correction = (N*m) / (N+m)*np.outer(delta,delta)
            scatter = scatter + batch_scatter + correction
            mean = (N*mean + m*batch_mean) / (N+m)
            N = N+m
        
    covariance = scatter/N
    eigenvalue, eigenvector = np.linalg.eigh(covariance)

    idx = np.argsort(eigenvalue)[::-1]
    eigenvector = eigenvector[:,idx]
    components = eigenvector[:, :n_components].T
        # Deterministic sign flip
    for i in range(components.shape[0]):
        comp = components[i]
        max_idx = np.argmax(np.abs(comp))
        if comp[max_idx] < 0:
            components[i] *= -1

    return components.tolist()