import numpy as np

def xgboost_objective(gradients: np.ndarray, hessians: np.ndarray,
                      left_indices: np.ndarray, right_indices: np.ndarray,
                      lambda_reg: float = 1.0, gamma: float = 0.0) -> dict:
    """
    Calculate XGBoost objective function components for a potential split.
    
    Args:
        gradients: First-order gradients for each sample
        hessians: Second-order hessians for each sample
        left_indices: Indices of samples going to left child
        right_indices: Indices of samples going to right child
        lambda_reg: L2 regularization parameter
        gamma: Tree complexity penalty
        
    Returns:
        Dictionary with 'left_weight', 'right_weight', and 'gain'
    """
    gl = 0
    hl = 0
    gr = 0
    hr = 0
    for i in left_indices:
        gl = gl + gradients[i]
        hl = hl + hessians[i]
    for j in right_indices:
        gr = gr + gradients[j]
        hr = hr + hessians[j]
    gp = gl + gr
    hp = hl + hr
    left_weight = -gl / (hl + lambda_reg)
    right_weight = -gr / (hr + lambda_reg)
    tot_weight = -gp / (hp + lambda_reg)
    gain = 0.5 * ((-gl*left_weight) + (-gr*right_weight) - (-gp*tot_weight)) - gamma
    return {'left_weight': round(left_weight, 4), 'right_weight': round(right_weight, 4),'gain': round(gain, 4)}