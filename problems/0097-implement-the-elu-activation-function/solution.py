import numpy as np
def elu(x: float, alpha: float = 1.0) -> float:
	"""
	Compute the ELU activation function.

	Args:
		x (float): Input value
		alpha (float): ELU parameter for negative values (default: 1.0)

	Returns:
		float: ELU activation value
	"""
	if (x>0):
		return x
	abc = np.exp(x)
	val = alpha * (abc - 1)
	return round(val,4)