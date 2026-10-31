import numpy as np

def bellman_q_value(states: list, actions: list, transition_probs: dict, rewards: dict, policy: dict, gamma: float, theta: float = 1e-6) -> dict:
    """
    Compute Q^pi(s, a) for all state-action pairs using iterative policy evaluation
    based on the Bellman expectation equation for action-values.

    Args:
        states: List of state identifiers
        actions: List of action identifiers
        transition_probs: Dict mapping (s, a, s') -> P(s'|s,a)
        rewards: Dict mapping (s, a, s') -> R(s,a,s')
        policy: Dict mapping (s, a) -> pi(a|s)
        gamma: Discount factor
        theta: Convergence threshold

    Returns:
        Dict mapping (s, a) -> Q-value rounded to 4 decimal places
    """

    # Initialize Q-values to zero
    Q = {(s, a): 0.0 for s in states for a in actions}

    while True:
        delta = 0.0
        Q_new = Q.copy()

        for s in states:
            for a in actions:
                q_value = 0.0

                # Bellman expectation equation:
                # Q(s,a) = sum_s' P(s'|s,a) *
                #          [R(s,a,s') + gamma * sum_a' pi(a'|s') Q(s',a')]

                for s_next in states:
                    probability = transition_probs.get((s, a, s_next), 0.0)
                    reward = rewards.get((s, a, s_next), 0.0)

                    # Expected value of following the policy at the next state
                    next_value = sum(
                        policy.get((s_next, a_next), 0.0) *
                        Q.get((s_next, a_next), 0.0)
                        for a_next in actions
                    )

                    q_value += probability * (
                        reward + gamma * next_value
                    )

                Q_new[(s, a)] = q_value
                delta = max(delta, abs(q_value - Q[(s, a)]))

        Q = Q_new

        # Stop when values converge
        if delta < theta:
            break

    # Round results to 4 decimal places
    return {key: round(value, 4) for key, value in Q.items()}