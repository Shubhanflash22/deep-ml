import numpy as np

def learn_successor_representation(
    experience: list,
    n_states: int,
    gamma: float,
    alpha_sr: float,
    alpha_w: float
) -> tuple:
    
    M = np.zeros((n_states, n_states))
    w = np.zeros(n_states)

    for state, reward, next_state, done in experience:
        # 1. Update reward weights
        w[state] += alpha_w * (reward - w[state])

        # 2. Build SR TD target
        target = np.zeros(n_states)
        target[state] = 1.0

        if not done:
            target += gamma * M[next_state]

        # 3. Update current state's SR row
        M[state] += alpha_sr * (target - M[state])

    # 4. Reconstruct value function
    V = M @ w

    return (
        np.round(M, 4).tolist(),
        np.round(w, 4).tolist(),
        np.round(V, 4).tolist()
    )