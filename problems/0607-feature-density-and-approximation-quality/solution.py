import numpy as np

def feature_density_approximation(
    train_states: np.ndarray,
    train_values: np.ndarray,
    test_states: np.ndarray,
    test_values: np.ndarray,
    n_centers: int,
    sigma: float
) -> dict:
    """
    Approximate a value function using RBF features and evaluate quality.
    
    Args:
        train_states: 1D array of training state positions in [0, 1]
        train_values: 1D array of true values at training states
        test_states: 1D array of test state positions in [0, 1]
        test_values: 1D array of true values at test states
        n_centers: number of RBF centers placed uniformly in [0, 1]
        sigma: bandwidth parameter for the RBF features
    
    Returns:
        dict with keys 'centers', 'weights', 'predictions', 'rmse'
    """
    # Step 1: Place RBF centers uniformly in [0, 1]
    if n_centers == 1:
        centers = np.array([0.5])
    else:
        centers = np.linspace(0.0, 1.0, n_centers)

    # Step 2: Compute training feature matrix
    Phi_train = np.exp(
        -((train_states[:, None] - centers[None, :]) ** 2)
        / (2 * sigma ** 2)
    )

    # Step 3: Fit weights using least squares
    weights = np.linalg.lstsq(
        Phi_train,
        train_values,
        rcond=None
    )[0]

    # Step 4: Compute test feature matrix
    Phi_test = np.exp(
        -((test_states[:, None] - centers[None, :]) ** 2)
        / (2 * sigma ** 2)
    )

    # Step 5: Predict test values
    predictions = Phi_test @ weights

    # Step 6: Calculate RMSE
    rmse = np.sqrt(np.mean((predictions - test_values) ** 2))

    # Step 7: Round all results to 4 decimal places
    return {
        'centers': np.round(centers, 4).tolist(),
        'weights': np.round(weights, 4).tolist(),
        'predictions': np.round(predictions, 4).tolist(),
        'rmse': round(float(rmse), 4)
    }