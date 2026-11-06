import numpy as np

def f_score(y_true, y_pred, beta):
	"""
	Calculate F-Score for a binary classification task.

	:param y_true: Numpy array of true labels
	:param y_pred: Numpy array of predicted labels
	:param beta: The weight of precision in the harmonic mean
	:return: F-Score rounded to three decimal places
	"""
	n = len(y_true)
	tp = 0
	fp = 0
	tn = 0
	fn = 0
	for i in range(0,n):
		a = y_true[i]
		b = y_pred[i]
		if a == 1 and b == 1:
			tp = tp + 1
		elif a == 1 and b == 0:
			fn = fn + 1
		elif a == 0 and b == 0:
			tn = tn + 1
		else:
			fp = fp + 1
	prec = tp / (tp + fp)
	rec = tp / (tp + fn)
	f1 = (1 + beta * beta) * prec * rec / ((beta*beta*prec) + rec) 
	return round(f1,3)