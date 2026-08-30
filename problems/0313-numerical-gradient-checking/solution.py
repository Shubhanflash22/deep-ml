import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    """

    # Create an array to store the numerical gradient
    numerical_grad = np.zeros_like(x, dtype=float)

    # Compute each partial derivative one element at a time
    for i in range(len(x)):

        # Make copies so we don't modify the original x
        x_plus = x.copy()
        x_minus = x.copy()

        # Perturb only the i-th element
        x_plus[i] += epsilon
        x_minus[i] -= epsilon

        # Centered finite difference:
        # df/dx_i ≈ [f(x + ε) - f(x - ε)] / (2ε)
        numerical_grad[i] = (
            f(x_plus) - f(x_minus)
        ) / (2 * epsilon)

    # Calculate the numerator:
    # difference between numerical and analytical gradients
    numerator = np.linalg.norm(
        numerical_grad - analytical_grad
    )

    # Calculate the denominator:
    # sum of the magnitudes of both gradients
    denominator = (
        np.linalg.norm(numerical_grad)
        + np.linalg.norm(analytical_grad)
    )

    # Edge case:
    # If both gradients are zero, they match perfectly
    if denominator == 0:
        relative_error = 0.0
    else:
        relative_error = numerator / denominator

    return numerical_grad, relative_error