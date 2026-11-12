import numpy as np

def tile_coding_step(base_alpha: float, n_tilings: int, weights: list, active_tiles: list, target: float) -> dict:
    """
    Perform a single value function update using tile coding with adjusted step size.
    
    Args:
        base_alpha: The base learning rate before adjustment
        n_tilings: Number of tilings used in the tile coding scheme
        weights: List of current weight values for all tiles
        active_tiles: List of indices of tiles active for the current state
        target: The target value (e.g., a return or TD target)
    
    Returns:
        Dictionary with 'adjusted_alpha', 'prediction_before',
        'prediction_after', and 'updated_weights'
    """
    adjusted_alpha = base_alpha / n_tilings
    pred_bef = sum(weights[i] for i in active_tiles)
    error = target - pred_bef
    updated_weights = weights.copy()
    for i in active_tiles:
        updated_weights[i] = updated_weights[i] + adjusted_alpha * error
    pred_aft = sum(updated_weights[i] for i in active_tiles)
    return {
        'adjusted_alpha': round(adjusted_alpha, 4),
        'prediction_before': round(pred_bef, 4),
        'prediction_after': round(pred_aft, 4),
        'updated_weights': [round(w, 4) for w in updated_weights]
    }