import torch

def mse(pred, target):
    """
    Compute mean squared error between pred and target.

    Args:
        pred (torch.Tensor): Predicted values.
        target (torch.Tensor): Ground-truth values (same shape as pred).

    Returns:
        float: Mean of squared differences.
    """
    # TODO
    diff = pred - target
    squared_diff = diff ** 2
    mean_squared_error = torch.mean(squared_diff)
    return mean_squared_error.item()
