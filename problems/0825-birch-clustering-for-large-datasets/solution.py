import numpy as np

def birch_cluster(X, threshold):
    """
    Single-level BIRCH clustering.

    X: array-like of shape (n_samples, n_features)
    threshold: maximum allowed subcluster radius

    Returns:
        List of centroids, sorted lexicographically.
    """

    # Convert input into a NumPy array
    X = np.asarray(X, dtype=float)

    # Each subcluster stores:
    # N  = number of points
    # LS = sum of points along each dimension
    # SS = sum of squared points along each dimension
    subclusters = []

    # Process each point one at a time
    for x in X:

        # If there are no subclusters yet,
        # create the first one directly from x.
        if not subclusters:
            subclusters.append([
                1,
                x.copy(),
                x ** 2
            ])
            continue

        # Find the existing subcluster whose
        # centroid is closest to x.
        best_idx = 0
        best_dist = float("inf")

        for i, (N, LS, SS) in enumerate(subclusters):

            # Current centroid = LS / N
            centroid = LS / N

            # Euclidean distance from x to centroid
            dist = np.linalg.norm(x - centroid)

            # Only update on strictly smaller distance.
            # Therefore, ties keep the smaller index.
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        # Get the closest subcluster
        N, LS, SS = subclusters[best_idx]

        # Tentatively add x to that subcluster
        new_N = N + 1
        new_LS = LS + x
        new_SS = SS + x ** 2

        # New centroid after absorbing x
        new_centroid = new_LS / new_N

        # Variance for each dimension:
        # E[x^2] - E[x]^2
        variance = new_SS / new_N - new_centroid ** 2

        # Floating-point arithmetic can produce tiny negatives,
        # e.g. -1e-15 instead of 0.
        variance = np.maximum(variance, 0)

        # Radius = sqrt(sum of per-dimension variances)
        new_radius = np.sqrt(np.sum(variance))

        # If the new radius is within the threshold,
        # permanently absorb x.
        if new_radius <= threshold:
            subclusters[best_idx] = [
                new_N,
                new_LS,
                new_SS
            ]

        # Otherwise, don't modify the closest cluster.
        # Create a brand-new cluster containing only x.
        else:
            subclusters.append([
                1,
                x.copy(),
                x ** 2
            ])

    # Extract the final centroids
    centroids = []

    for N, LS, SS in subclusters:
        centroid = LS / N
        centroids.append(centroid.tolist())

    # Sort lexicographically
    centroids.sort()

    return centroids