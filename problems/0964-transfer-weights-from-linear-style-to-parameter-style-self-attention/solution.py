
import numpy as np

def transfer_and_attend(X, W_q_lin, W_k_lin, W_v_lin):
    X = np.array(X, dtype=float)

    # 1. Transfer weights by transposing
    W_q = np.array(W_q_lin, dtype=float).T
    W_k = np.array(W_k_lin, dtype=float).T
    W_v = np.array(W_v_lin, dtype=float).T

    # 2. Compute queries, keys, and values
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    # 3. Compute scaled attention scores
    d_out = Q.shape[1]
    scores = (Q @ K.T) / np.sqrt(d_out)

    # 4. Apply causal mask
    seq_len = X.shape[0]
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    scores[mask] = -1e9

    # 5. Numerically stable softmax
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    attention = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # 6. Compute attention output
    output = attention @ V

    return output.tolist()