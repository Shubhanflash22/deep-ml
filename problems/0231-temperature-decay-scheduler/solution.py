import numpy as np

def temperature_decay(
    schedule_type: str,
    initial_temp: float,
    current_step: int,
    total_steps: int,
    final_temp: float = 0.01,
    decay_rate: float = 0.95
) -> float:
	"""
	Compute temperature at current training step using decay schedule.
	
	Temperature controls randomness in neural network outputs:
	- High temperature: More random, more exploration
	- Low temperature: More deterministic, more exploitation
	
	Args:
		schedule_type: Decay schedule type
		  'linear': Steady linear decrease
		  'exponential': Fast early decay, slow later
		  'cosine': Smooth cosine curve
		  'constant': No decay
		initial_temp: Starting temperature
		current_step: Current training step (0 to total_steps)
		total_steps: Total number of training steps
		final_temp: Minimum temperature (floor)
		decay_rate: Decay rate per step (for exponential)
	
	Returns:
		Temperature value at current step
	"""
	if schedule_type == "linear":
		ans = max(final_temp, initial_temp - (initial_temp - final_temp) * current_step / total_steps)
	elif schedule_type == "exponential":
		ans = max(final_temp, initial_temp * decay_rate ** current_step)
	elif schedule_type == "cosine":
		ans = final_temp + 0.5 * (initial_temp - final_temp) * (1 + np.cos(22/7 * current_step / total_steps))
	else:
		ans = initial_temp
	return ans