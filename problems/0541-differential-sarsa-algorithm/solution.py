def differential_sarsa(transitions: dict, initial_state: str, alpha: float, beta: float, num_steps: int) -> tuple:
    """
    Differential Sarsa for the average-reward continuing setting.
    
    Args:
        transitions: dict mapping (state, action) -> (reward, next_state)
        initial_state: starting state
        alpha: step size for Q-value updates
        beta: step size for average reward estimate
        num_steps: number of steps to simulate
    
    Returns:
        Tuple of (Q, R_bar) where Q is a dict {(state, action): float}
        and R_bar is a float.
    """

    # Initialize Q-table
    Q = {key: 0.0 for key in transitions}

    # Initialize average reward
    R_bar = 0.0

    state = initial_state

    # Helper to get greedy action with lexicographic tie-breaking
    def greedy_action(state):
        actions = [action for (s, action) in transitions if s == state]

        return min(
            actions,
            key=lambda action: (-Q[(state, action)], action)
        )

    # Initial action
    action = greedy_action(state)

    for _ in range(num_steps):
        reward, next_state = transitions[(state, action)]

        # Select next action greedily
        next_action = greedy_action(next_state)

        # Differential TD error
        delta = (
            reward
            - R_bar
            + Q[(next_state, next_action)]
            - Q[(state, action)]
        )

        # Update average reward
        R_bar += beta * delta

        # Update Q-value
        Q[(state, action)] += alpha * delta

        # Move to next state/action
        state = next_state
        action = next_action

    return Q, R_bar