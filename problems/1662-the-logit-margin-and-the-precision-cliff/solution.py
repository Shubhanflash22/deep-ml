import numpy as np
def logit_margin(z, t):
    """Margin m = z_t - log sum_{j != t} exp(z_j), computed stably."""
    z = np.asarray(z, dtype=np.float64)
    # Remove the target logit.
    other = np.delete(z, t)
    # Stable log-sum-exp.
    a = np.max(other)
    log_sum_exp = a + np.log(np.sum(np.exp(other - a)))
    return float(z[t] - log_sum_exp)

def precision_cliff(dtype):
    """Find the smallest margin at which the target probability rounds to 1."""
    left = 0.0
    right = 100.0
    one = dtype(1)
    while right - left >= 1e-9:
        mid = (left + right) / 2.0
        # Perform the arithmetic in the requested dtype.
        exp_value = dtype(np.exp(-mid))
        denominator = dtype(one + exp_value)
        probability = dtype(one / denominator)
        if probability == one:
            right = mid
        else:
            left = mid
    return round(right, 4)