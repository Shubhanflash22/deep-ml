import numpy as np

def ffn(x: list[float], W1: list[list[float]], b1: list[float], W2: list[list[float]], b2: list[float], dropout_p: float=0.1, seed: int=42) -> list[float]:
	"""
	Implement a position-wise feed-forward block with residual and dropout.
	"""
	W1 = np.array(W1)
	W2 = np.array(W2)
	b1 = np.array(b1)
	b2 = np.array(b2)
	x = np.array(x)

	hidden = np.maximum(0, W1 @ x + b1)   # element-wise ReLU
	output = W2 @ hidden + b2

	# Dropout
	np.random.seed(seed)
	mask = (np.random.rand(*output.shape) >= dropout_p)
	output = output * mask / (1 - dropout_p)

	ans = output + x
	return np.round(ans, 4).tolist()