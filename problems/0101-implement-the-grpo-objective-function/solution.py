import numpy as np

def grpo_objective(rhos, A, pi_theta_old, pi_theta_ref, epsilon=0.2, beta=0.01) -> float:
    """
    Compute the GRPO objective function.

    Args:
        rhos: List of likelihood ratios (pi_theta / pi_theta_old).
        A: List of advantage estimates.
        pi_theta_old: List of old policy probabilities (per-sample, not normalized).
        pi_theta_ref: List of reference policy probabilities (per-sample, not normalized).
        epsilon: Clipping parameter for the surrogate objective.
        beta: KL divergence penalty coefficient.

    Returns:
        The computed GRPO objective value.
    """
    rhos = np.asarray(rhos, dtype=float)
    A = np.asarray(A, dtype=float)
    pi_theta_old = np.asarray(pi_theta_old, dtype=float)
    pi_theta_ref = np.asarray(pi_theta_ref, dtype=float)

    # Current policy probabilities
    pi_theta = rhos * pi_theta_old

    # PPO clipped surrogate objective
    clipped_rhos = np.clip(rhos, 1 - epsilon, 1 + epsilon)
    surrogate = np.minimum(rhos * A, clipped_rhos * A)

    # Importance-weighted KL estimator
    ratio = pi_theta_ref / pi_theta
    kl = rhos * (ratio - np.log(ratio) - 1.0)

    # Average objective
    return float(np.mean(surrogate - beta * kl))