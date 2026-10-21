
def random_walk_td_mc(episodes: list, n_states: int, gamma: float = 1.0, alpha: float = 0.1) -> tuple:
    """
    Run TD(0) and first-visit Monte Carlo prediction on random walk episodes.
    """
    # Initialize state values
    td_values = [0.0] * n_states
    mc_values = [0.0] * n_states

    # Monte Carlo return tracking
    returns_sum = [0.0] * n_states
    returns_count = [0] * n_states

    # Process episodes in order
    for episode in episodes:

        # TD(0): online updates after each transition
        for state, reward, next_state, done in episode:
            next_value = 0.0 if done else td_values[next_state]

            td_values[state] += alpha * (
                reward + gamma * next_value - td_values[state]
            )

        # First-visit Monte Carlo
        visited = set()
        G = 0.0

        # Traverse episode backward to calculate returns
        for state, reward, next_state, done in reversed(episode):
            G = reward + gamma * G

            # Only count the first occurrence in forward order
            if state not in visited:
                visited.add(state)
                returns_sum[state] += G
                returns_count[state] += 1

    # Average returns for each state
    for state in range(n_states):
        if returns_count[state] > 0:
            mc_values[state] = returns_sum[state] / returns_count[state]

    return td_values, mc_values
