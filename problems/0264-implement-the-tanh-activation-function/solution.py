import math
import numpy as np

def tanh(x: float) -> float:
	"""
	Implements the Tanh (hyperbolic tangent) activation function.

	Args:
		x (float): Input value

	Returns:
		float: The tanh of the input, rounded to 4 decimal places
	"""
	num = np.exp(2*x) - 1
	dem = np.exp(2*x) + 1
	return num/dem