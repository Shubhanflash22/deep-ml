import numpy as np

def qk_norm_attention(Q: np.ndarray,
                      K: np.ndarray,
                      V: np.ndarray,
                      temperature: float = 1.0) -> tuple:
    """
    Apply QK-Norm attention: L2-normalize queries and keys before computing
    scaled dot-product attention.

    Args:
        Q: Query matrix, shape (seq_len_q, d_k)
        K: Key matrix, shape (seq_len_k, d_k)
        V: Value matrix, shape (seq_len_k, d_v)
        temperature: Temperature scaling parameter

    Returns:
        Tuple (attention_output, attention_weights)
    """
    eps = 1e-8

    # Step 1: L2-normalize each row of Q
    Q_norm = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), eps)

    # Step 2: L2-normalize each row of K
    K_norm = K / np.maximum(np.linalg.norm(K, axis=1, keepdims=True), eps)

    # Step 3: Compute attention scores
    scores = (Q_norm @ K_norm.T) / temperature

    # Step 4: Numerically stable softmax
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Step 5: Compute attention output
    attention_output = attention_weights @ V

    return attention_output, attention_weights