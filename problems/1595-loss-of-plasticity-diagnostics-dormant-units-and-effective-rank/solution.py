import numpy as np

def dormant_ratio(H, tau):
    """Return fraction of dormant units and sorted dormant indices."""
    H = np.asarray(H, dtype=float)
    if H.ndim != 2:
        raise ValueError("H must be a 2D array.")
    n_units = H.shape[1]
    if n_units == 0:
        return 0.0, []
    # Calculate mean absolute activation for each unit
    unit_scores = np.mean(np.abs(H), axis=0)
    # Calculate average activity across the layer
    layer_mean = np.mean(unit_scores)
    # Units below or equal to the relative activity threshold are dormant
    threshold = tau * layer_mean
    dormant_indices = np.where(unit_scores <= threshold)[0].tolist()
    ratio = len(dormant_indices) / n_units
    return ratio, dormant_indices

def effective_rank(H, eps=1e-12):
    """Calculate effective rank of column-centred activations."""
    H = np.asarray(H, dtype=float)
    H_centered = H - np.mean(H, axis=0, keepdims=True)
    singular_values = np.linalg.svd(
        H_centered, compute_uv=False
    )
    total = np.sum(singular_values)
    if total < eps:
        return 0.0
    p = singular_values / total
    entropy = -np.sum(p * np.log(p + eps))
    return float(np.exp(entropy))

def plasticity_report(H, W, tau, decimals=4):
    """Return summary of plasticity metrics."""
    ratio, _ = dormant_ratio(H, tau)
    rank = effective_rank(H)
    return {
        "dormant_ratio": round(float(ratio), decimals),
        "effective_rank": round(float(rank), decimals),
        "weight_norm": round(float(np.linalg.norm(W)), decimals)
    }