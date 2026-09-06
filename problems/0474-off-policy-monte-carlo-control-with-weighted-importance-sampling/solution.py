import numpy as np

def off_policy_mc_control(episodes: list, behavior_policy: dict,
                          n_states: int, n_actions: int,
                          gamma: float = 1.0) -> list:

    Q = np.zeros((n_states, n_actions))
    C = np.zeros((n_states, n_actions))

    for episode in episodes:

        G = 0.0
        W = 1.0

        # Process episode backwards
        for t in range(len(episode) - 1, -1, -1):

            state, action, reward = episode[t]

            G = gamma * G + reward

            C[state, action] += W

            Q[state, action] += (W / C[state, action]) * (
                G - Q[state, action]
            )

            # Greedy action under current Q
            greedy_action = np.argmax(Q[state])

            # If behavior action differs from target policy, stop
            if action != greedy_action:
                break

            # Importance sampling ratio
            prob = behavior_policy[(state, action)]
            W *= 1.0 / prob

    return np.round(Q, 4).tolist()