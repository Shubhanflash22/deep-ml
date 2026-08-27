import numpy as np

def causal_mask_attention(attn_weights: np.ndarray, method: str = 'tril') -> list:
    """Apply causal masking two ways and return the resulting attention matrix as a nested list."""
    if method == "tril":
        T = attn_weights.shape[0]

        mask = np.tril(np.ones((T, T)))

        masked = attn_weights * mask

        # Renormalize each row
        masked = masked / masked.sum(axis=1, keepdims=True)

        return masked.tolist()

    elif method == "triu":
        T = attn_weights.shape[0]

        # Reconstruct scores
        scores = np.log(attn_weights)

        # Mask future positions
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores[mask] = -np.inf

        # Softmax
        scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        scores = scores / scores.sum(axis=1, keepdims=True)

        return scores.tolist()

    else:
        raise ValueError("method must be 'tril' or 'triu'")