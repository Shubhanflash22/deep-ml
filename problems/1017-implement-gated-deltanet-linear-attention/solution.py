import numpy as np

def gated_deltanet(q, k, v, a, b, g, A_log, rms_weight, eps=1e-6):
    """
    Simplified Gated DeltaNet linear attention forward pass.
    """
    q = np.asarray(q, dtype=float)
    k = np.asarray(k, dtype=float)
    v = np.asarray(v, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    g = np.asarray(g, dtype=float)
    rms_weight = np.asarray(rms_weight, dtype=float)

    T, d = q.shape

    # 1. L2 normalize q and k
    q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + eps)
    k_norm = k / (np.linalg.norm(k, axis=1, keepdims=True) + eps)

    # 2. Decay gate
    softplus_a = np.logaddexp(0.0, a)
    alpha = np.exp(-softplus_a * np.exp(A_log))

    # 3. Update gate
    beta = 1.0 / (1.0 + np.exp(-b))

    # 4. Recurrent memory
    S = np.zeros((d, d), dtype=float)
    outputs = []

    for t in range(T):
        # a. Decay
        S *= alpha[t]

        # b. Prediction error
        prediction = S.T @ k_norm[t]
        delta = (v[t] - prediction) * beta[t]

        # c. Memory update
        S += np.outer(k_norm[t], delta)

        # d. Output
        o = S.T @ q_norm[t]

        # 5. RMSNorm
        rms = np.sqrt(np.mean(o ** 2) + eps)
        o_norm = (o / rms) * rms_weight

        # 6. SiLU output gate
        sigmoid_g = 1.0 / (1.0 + np.exp(-g[t]))
        y = o_norm * (g[t] * sigmoid_g)

        outputs.append(y)

    # Round to 4 decimal places and return nested Python lists
    return np.round(np.array(outputs), 4).tolist()