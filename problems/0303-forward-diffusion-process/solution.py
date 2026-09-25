import numpy as np

def forward_diffusion(
    x_0: np.ndarray,
    t: int,
    beta_start: float,
    beta_end: float,
    num_timesteps: int,
    noise: np.ndarray
) -> np.ndarray:
    """
    Apply forward diffusion process to add noise to input data.
    """
    # 1. Linear beta schedule
    betas = np.linspace(beta_start, beta_end, num_timesteps)

    # 2. alpha = 1 - beta
    alphas = 1 - betas

    # 3. Cumulative product of alphas
    alpha_bar = np.cumprod(alphas)

    # 4. Get alpha_bar at timestep t (t is 1-indexed)
    alpha_bar_t = alpha_bar[t - 1]

    # 5. Closed-form forward diffusion
    x_t = (
        np.sqrt(alpha_bar_t) * x_0
        + np.sqrt(1 - alpha_bar_t) * noise
    )

    return x_t