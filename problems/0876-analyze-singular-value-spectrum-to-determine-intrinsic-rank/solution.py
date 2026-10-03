import numpy as np

def suggest_rank(delta_W: np.ndarray, energy_threshold: float) -> int:
	"""
	Return the smallest rank k such that the top-k singular values of delta_W
	capture at least `energy_threshold` of the total squared-singular-value energy.
	"""
	U, S, Vt = np.linalg.svd(delta_W, full_matrices=False)
	energy = S ** 2
	total_energy = energy.sum()

    if total_energy == 0:
        return 0

    cumulative_energy = np.cumsum(energy)
    ratio = cumulative_energy / total_energy

    k = np.searchsorted(ratio, energy_threshold) + 1

    return k