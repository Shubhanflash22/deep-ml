import numpy as np
from typing import List, Tuple

def sigmoid(x):
    if x>=0:
        ans = 1/(1 + np.exp(-x))
    else:
        ans = np.exp(x)/(1 + np.exp(x))
    return ans


def fit_bradley_terry(comparisons: List[Tuple[int, int]], n_items: int, learning_rate: float = 0.5, n_iterations: int = 100) -> np.ndarray:
    """
    Fit Bradley-Terry model parameters using maximum likelihood estimation.
    
    Args:
        comparisons: List of (winner_idx, loser_idx) tuples
        n_items: Total number of items to rank
        learning_rate: Step size for gradient ascent
        n_iterations: Number of optimization iterations
    
    Returns:
        np.ndarray: Estimated strength parameters of shape (n_items,)
    """
    beta = np.zeros(n_items)
    for i in range(n_iterations):
        grad = np.zeros(n_items)
        for winner, loser in comparisons:
            diff = beta[winner]- beta[loser]
            p = sigmoid(diff)
            error = 1-p

            grad[winner] = grad[winner] + error
            grad[loser] = grad[loser] - error 

        beta = beta + learning_rate * grad
        beta = beta - np.mean(beta)
    return beta