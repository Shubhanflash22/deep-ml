import numpy as np

def embedding_via_one_hot(token_ids, W):
    """
    Compute token embeddings via one-hot encoding and matrix multiplication.

    Args:
        token_ids: list or 1D array of integer token IDs
        W: numpy array of shape (vocab_size, embed_dim)

    Returns:
        numpy array of shape (len(token_ids), embed_dim)
    """
    vocab_size = W.shape[0]
    embed_dim = W.shape[1]
    n = len(token_ids)
    H = np.zeros((n,vocab_size))
    for i in range(0,n):
        H[i][token_ids[i]] = 1
    ans = H @ W
    return ans