import numpy as np

def kernel_function(x1, x2):
	n = len(x1)
	ans = 0
	for i in range(0,n):
		temp = x1[i] * x2[i]
		ans = ans + temp
	return ans
