import numpy as np

def guidance_attention_mask(
    chunk_sizes: list[int],
    current_chunk: int
) -> np.ndarray:
    """
    Build a boolean attention mask for chunked autoregressive video generation.

    Args:
        chunk_sizes:   tokens per chunk, in order
        current_chunk: index of the chunk being generated

    Returns:
        Boolean array of shape (total_tokens, total_tokens).
        True means the row token can attend to the column token.
    """
    total_tokens = sum(chunk_sizes)
    mask = np.zeros((total_tokens, total_tokens), dtype=bool)

    start = sum(chunk_sizes[:current_chunk])
    end = start + chunk_sizes[current_chunk]

    # Rule 1: Historical tokens attend bidirectionally
    mask[:start, :start] = True

    # Rule 2: Current tokens attend to all historical tokens
    mask[start:end, :start] = True

    # Rule 3: Current tokens attend causally within their chunk
    current_size = end - start
    mask[start:end, start:end] = np.tril(
        np.ones((current_size, current_size), dtype=bool)
    )

    return mask
