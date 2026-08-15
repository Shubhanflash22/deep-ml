def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    n_features = len(vectors)
    n_observations = len(vectors[0])

    # Calculate means
    means = [sum(feature) / n_observations for feature in vectors]

    # Build covariance matrix
    covariance_matrix = []

    for i in range(n_features):
        row = []
        for j in range(n_features):
            covariance = sum(
                (vectors[i][k] - means[i]) *
                (vectors[j][k] - means[j])
                for k in range(n_observations)
            ) / (n_observations - 1)
            row.append(covariance)
        covariance_matrix.append(row)
    return covariance_matrix