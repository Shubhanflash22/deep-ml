import numpy as np

def compressed_row_sparse_matrix(dense_matrix):
	m = len(dense_matrix)
	n = len(dense_matrix[0])
	vals = [] 
	col_idx = [] 
	row_ptr = [0]
	for i in range(0,m):
		for j in range(0,n):
			temp = dense_matrix[i][j]
			if temp!=0:
				vals.append(temp)
				col_idx.append(j)
		row_ptr.append(len(vals))
	return vals, col_idx, row_ptr