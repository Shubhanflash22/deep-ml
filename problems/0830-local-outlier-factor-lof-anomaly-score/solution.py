import numpy as np

def local_outlier_factor(X, k):
    """
    Compute the Local Outlier Factor (LOF) score for each point in X.

    Args:
        X: array-like of shape (n, d)
        k: int, neighborhood size

    Returns:
        list of length n with the LOF score of each point
    """
    X = np.asarray(X, dtype = float)
    n = len(X)
    # Step 1: Pairwise Euclidean distance matrix
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    # Step 2: k nearest neighbors (excluding self)
    neighbors = []
    k_distance = np.zeros(n)
    for i in range(n):
        order = np.argsort(dist[i])       # self is first (distance = 0)
        neigh = order[1:k+1]              # exclude self
        neighbors.append(neigh)
        k_distance[i] = dist[i, neigh[-1]]

    # Step 3: Local Reachability Density (LRD)
    lrd = np.zeros(n)
    for i in range(n):
        reach_dists = []
        for o in neighbors[i]:
            reach_dist = max(k_distance[o], dist[i, o])
            reach_dists.append(reach_dist)
        lrd[i] = 1.0 / np.mean(reach_dists)
    
    # Step 4: Local Outlier Factor (LOF)
    lof = np.zeros(n)
    for i in range(n):
        ratios = [lrd[o] / lrd[i] for o in neighbors[i]]
        lof[i] = np.mean(ratios)

    return lof.tolist()

