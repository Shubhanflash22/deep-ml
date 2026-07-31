import numpy as np

def softmax(values):
	m = max(values)
	values = values - m
	exp_val = np.exp(values)
	ans = exp_val / np.sum(exp_val)
	return ans

def pattern_weaver(n, crystal_values, dimension):
	# Your code here
	x = np.array(crystal_values, dtype = float)
	Q = x
	K = x
	V = x
	scores = np.outer(Q, K) / np.sqrt(dimension)
	attention = np.array([softmax(row) for row in scores])
	output = attention @ V
	return np.round(output,4)