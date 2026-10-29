import numpy as np

def gradient_boosting_step(X, y, current_predictions, learning_rate=0.1):
    """
    Perform one step of gradient boosting regression using a decision stump.
    
    Args:
        X: Feature matrix (list of lists), shape (n_samples, n_features)
        y: Target values (list), shape (n_samples,)
        current_predictions: Current ensemble predictions (list), shape (n_samples,)
        learning_rate: Learning rate for the update (default 0.1)
    
    Returns:
        List of updated predictions rounded to 4 decimal places
    """
    from sklearn.tree import DecisionTreeRegressor

    X = np.array(X)
    y = np.array(y)
    current_predictions = np.array(current_predictions)

    # Calculate residuals (negative gradients for squared error loss)
    residuals = y - current_predictions

    # Fit a decision stump to the residuals
    stump = DecisionTreeRegressor(max_depth=1)
    stump.fit(X, residuals)

    # Predict residual corrections
    updates = stump.predict(X)

    # Update the current predictions
    new_predictions = current_predictions + learning_rate * updates

    # Return predictions rounded to 4 decimal places
    return np.round(new_predictions, 4).tolist()