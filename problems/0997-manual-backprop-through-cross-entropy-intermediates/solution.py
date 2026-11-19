
import numpy as np

def manual_cross_entropy_backward(logits: np.ndarray, y: np.ndarray) -> dict:
    """
    Manually compute gradients through every intermediate of softmax + NLL loss.
    """
    n, C = logits.shape

    # Forward pass intermediates
    logit_maxes = np.max(logits, axis=1, keepdims=True)
    norm_logits = logits - logit_maxes
    counts = np.exp(norm_logits)
    counts_sum = np.sum(counts, axis=1, keepdims=True)
    counts_sum_inv = 1.0 / counts_sum
    probs = counts * counts_sum_inv
    logprobs = np.log(probs)

    # Backward pass
    dlogprobs = np.zeros_like(logprobs)
    dlogprobs[np.arange(n), y] = -1.0 / n

    dprobs = dlogprobs / probs

    # probs = counts * counts_sum_inv
    dcounts = dprobs * counts_sum_inv
    dcounts_sum_inv = np.sum(dprobs * counts, axis=1, keepdims=True)

    # counts_sum_inv = 1 / counts_sum
    dcounts_sum = -dcounts_sum_inv / (counts_sum ** 2)

    # counts_sum = sum(counts)
    dcounts += dcounts_sum

    # counts = exp(norm_logits)
    dnorm_logits = dcounts * counts

    # norm_logits = logits - logit_maxes
    dlogit_maxes = -np.sum(dnorm_logits, axis=1, keepdims=True)

    # logit_maxes = max(logits)
    # The max operation routes gradients to the maximum element.
    max_indices = np.argmax(logits, axis=1)
    dlogits = dnorm_logits.copy()
    dlogits[np.arange(n), max_indices] += dlogit_maxes[:, 0]

    return {
        'dlogprobs': dlogprobs,
        'dprobs': dprobs,
        'dcounts_sum_inv': dcounts_sum_inv,
        'dcounts_sum': dcounts_sum,
        'dcounts': dcounts,
        'dnorm_logits': dnorm_logits,
        'dlogit_maxes': dlogit_maxes,
        'dlogits': dlogits,
    }
