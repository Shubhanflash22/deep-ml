
import numpy as np


def gae_encode(A, X, W1, W2):
    """
    Graph Autoencoder encoder using a two-layer GCN.

    A: Adjacency matrix (N x N)
    X: Node feature matrix (N x F)
    W1: First weight matrix (F x H)
    W2: Second weight matrix (H x D)

    Returns:
        Z: Node embeddings (N x D)
    """
    N = A.shape[0]

    # Add self-loops
    A_hat = A + np.eye(N)

    # Compute degree matrix
    degrees = np.sum(A_hat, axis=1)

    # Symmetric normalization
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt

    # First GCN layer with ReLU
    H = np.maximum(0, A_norm @ X @ W1)

    # Second GCN layer
    Z = A_norm @ H @ W2

    return Z


def gae_logits(Z, pairs):
    """
    Compute inner-product decoder logits for node pairs.

    Z: Node embeddings (N x D)
    pairs: List or array of node pairs (M x 2)

    Returns:
        logits: Array of shape (M,)
    """
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)

    src = pairs[:, 0]
    dst = pairs[:, 1]

    return np.sum(Z[src] * Z[dst], axis=1)


def gae_loss(A, Z, pos_weight=None):
    """
    Full reconstruction BCE loss averaged over all N x N entries.

    If pos_weight is None, calculate:
        (N^2 - sum(A)) / sum(A)

    Returns:
        Python float
    """
    N = A.shape[0]

    logits = Z @ Z.T

    if pos_weight is None:
        num_positive = np.sum(A)
        num_negative = N * N - num_positive

        if num_positive == 0:
            pos_weight = 1.0
        else:
            pos_weight = num_negative / num_positive

    # Numerically stable BCE with logits
    loss = (
        pos_weight * A * np.logaddexp(0, -logits)
        + (1 - A) * np.logaddexp(0, logits)
    )

    return float(np.mean(loss))


def sampled_gae_loss(Z, pos_pairs, neg_pairs):
    """
    Mean binary cross-entropy over all sampled positive
    and negative pairs together.

    Positive pairs have label 1.
    Negative pairs have label 0.

    Returns:
        Python float
    """
    pos_logits = gae_logits(Z, pos_pairs)
    neg_logits = gae_logits(Z, neg_pairs)

    # BCE for positive edges (label 1)
    pos_loss = np.logaddexp(0, -pos_logits)

    # BCE for negative edges (label 0)
    neg_loss = np.logaddexp(0, neg_logits)

    # Mean over ALL pairs together
    all_losses = np.concatenate([pos_loss, neg_loss])

    return float(np.mean(all_losses))


def vgae_sample(mu, logvar, eps):
    """
    Reparameterization trick for VGAE.

    z = mu + exp(0.5 * logvar) * eps
    """
    return mu + eps * np.exp(0.5 * logvar)


def vgae_kl(mu, logvar):
    """
    KL divergence between q(Z|X,A) and standard normal prior.

    Sum over latent dimensions and average over nodes.

    Returns:
        Python float
    """
    kl = -0.5 * np.sum(
        1 + logvar - mu**2 - np.exp(logvar),
        axis=1
    )

    return float(np.mean(kl))
