
import numpy as np

def transformer_encoder_layer(X: np.ndarray, weights: dict, num_heads: int, eps: float = 1e-5) -> np.ndarray:
    """
    Forward pass of a single Transformer Encoder Layer.
    """

    batch_size, seq_len, d_model = X.shape

    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    d_k = d_model // num_heads

    # Helper: Layer Normalization
    def layer_norm(x, gamma, beta):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)

        return gamma * (x - mean) / np.sqrt(var + eps) + beta

    # 1. Multi-Head Self-Attention

    # Linear projections
    Q = X @ weights['W_q']
    K = X @ weights['W_k']
    V = X @ weights['W_v']

    # Split into attention heads
    Q = Q.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    # Scaled dot-product attention
    scores = Q @ K.transpose(0, 1, 3, 2)
    scores = scores / np.sqrt(d_k)

    # Numerically stable softmax
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    attention_weights = np.exp(scores)
    attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True)

    # Weighted sum of values
    attention_output = attention_weights @ V

    # Concatenate attention heads
    attention_output = attention_output.transpose(0, 2, 1, 3)
    attention_output = attention_output.reshape(batch_size, seq_len, d_model)

    # Output projection
    attention_output = attention_output @ weights['W_o']

    # 2. First Add & Norm
    X1 = layer_norm(
        X + attention_output,
        weights['gamma1'],
        weights['beta1']
    )

    # 3. Position-wise Feed-Forward Network

    ffn_hidden = X1 @ weights['W1'] + weights['b1']
    ffn_hidden = np.maximum(0, ffn_hidden)  # ReLU

    ffn_output = ffn_hidden @ weights['W2'] + weights['b2']

    # 4. Second Add & Norm
    output = layer_norm(
        X1 + ffn_output,
        weights['gamma2'],
        weights['beta2']
    )

    return output