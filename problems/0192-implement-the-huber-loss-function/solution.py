def huber_loss(y_true, y_pred, delta=1.0):
	"""
	Compute the Huber Loss between true and predicted values.

	Args:
		y_true (float | list[float]): Ground truth values
		y_pred (float | list[float]): Predicted values
		delta (float): Transition threshold between MSE and MAE behavior

	Returns:
		float: Average Huber loss
	"""
	# Convert inputs to lists if they are single float values
	if not isinstance(y_true, list):
		y_true = [y_true]
	if not isinstance(y_pred, list):
		y_pred = [y_pred]
		
	total_loss = 0.0
	n = len(y_true)
	
	for yt, yp in zip(y_true, y_pred):
		error = yt - yp
		abs_error = abs(error)
		
		if abs_error <= delta:
			# Quadratic behavior (MSE)
			total_loss += 0.5 * (error ** 2)
		else:
			# Linear behavior (MAE)
			total_loss += delta * (abs_error - 0.5 * delta)
			
	return total_loss / n
