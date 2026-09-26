import numpy as np

def unet_time_embedding(
    timesteps: list,
    embed_dim: int,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    max_period: int = 10000
) -> np.ndarray:

    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    t = np.asarray(timesteps)

    # Stage 1: sinusoidal embedding
    half_dim = embed_dim // 2

    i = np.arange(half_dim)

    freqs = np.exp(
        -np.log(max_period) * i / half_dim
    )

    args = t[:, None] * freqs[None, :]

    emb = np.concatenate([
        np.sin(args),
        np.cos(args)
    ], axis=1)

    # Stage 2: MLP
    h = emb @ W1 + b1

    # SiLU
    h = h * (1 / (1 + np.exp(-h)))

    out = h @ W2 + b2

    return out