def attention_memory(T: int, H: int, W: int, num_heads: int, bytes_per_element: int) -> list:
    """
    Compute memory (in bytes) of attention score tensors for full space-time
    attention and for space-time factorized attention.
    """
    # Total number of spatial tokens
    N = H * W

    # Full space-time attention: (T * H * W)^2 scores per head
    full_bytes = num_heads * (T * N) ** 2 * bytes_per_element

    # Factorized attention:
    # Temporal attention: N independent sequences of length T
    temporal_scores = N * T**2

    # Spatial attention: T independent sequences of length N
    spatial_scores = T * N**2

    factorized_bytes = num_heads * (temporal_scores + spatial_scores) * bytes_per_element

    return [full_bytes, factorized_bytes]