import numpy as np

def acceptance_rate_vs_temperature(draft_logits: np.ndarray, target_logits: np.ndarray, temperatures: np.ndarray) -> list:
    """
    Compute speculative decoding expected acceptance rate at various temperatures.
    
    Args:
        draft_logits: Logits from draft model, shape (vocab_size,)
        target_logits: Logits from target model, shape (vocab_size,)
        temperatures: Array of temperature values to evaluate
    
    Returns:
        List of acceptance rates (floats rounded to 4 decimal places)
    """
    ans = []

    for T in temperatures:
        # Scale logits
        draft_scaled = draft_logits / T
        target_scaled = target_logits / T

        # Stable softmax for draft
        draft_exp = np.exp(draft_scaled - np.max(draft_scaled))
        draft_prob = draft_exp / np.sum(draft_exp)

        # Stable softmax for target
        target_exp = np.exp(target_scaled - np.max(target_scaled))
        target_prob = target_exp / np.sum(target_exp)

        # Acceptance rate
        acceptance = np.sum(np.minimum(draft_prob, target_prob))

        ans.append(round(float(acceptance), 4))

    return ans