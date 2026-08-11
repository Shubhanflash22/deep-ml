import numpy as np

def options_value_iteration(n_states, terminal_states, transitions, options, gamma, theta=1e-10):
    """
    Perform SMDP value iteration using temporally extended options.
    """

    n_options = len(options)

    # Option models
    option_rewards = []
    option_transitions = []

    for option in options:
        initiation = option["initiation"]
        policy = option["policy"]
        termination = option["termination"]

        # R(s,o)
        R = np.zeros(n_states)

        # F(s,o,s')
        F = np.zeros((n_states, n_states))

        # States where the option is defined
        states = list(initiation)
        idx = {s: i for i, s in enumerate(states)}
        m = len(states)

        # -----------------------------
        # Solve for R
        # -----------------------------
        A = np.eye(m)
        b = np.zeros(m)

        for s in states:
            i = idx[s]

            if s in terminal_states:
                continue

            a = policy[s]

            for ns, p, r in transitions[s][a]:
                beta = termination.get(ns, 1.0)

                b[i] += p * r

                if ns in idx:
                    A[i, idx[ns]] -= gamma * p * (1 - beta)

        if m > 0:
            sol = np.linalg.solve(A, b)
            for s in states:
                R[s] = sol[idx[s]]

        # -----------------------------
        # Solve for F
        # -----------------------------
        for target in range(n_states):

            A = np.eye(m)
            b = np.zeros(m)

            for s in states:
                i = idx[s]

                if s in terminal_states:
                    continue

                a = policy[s]

                for ns, p, _ in transitions[s][a]:
                    beta = termination.get(ns, 1.0)

                    if ns == target:
                        b[i] += gamma * p * beta

                    if ns in idx:
                        A[i, idx[ns]] -= gamma * p * (1 - beta)

            if m > 0:
                sol = np.linalg.solve(A, b)

                for s in states:
                    F[s][target] = sol[idx[s]]

        option_rewards.append(R)
        option_transitions.append(F)

    # ==========================================
    # SMDP Value Iteration
    # ==========================================

    V = np.zeros(n_states)
    policy = [0] * n_states

    while True:

        delta = 0
        newV = V.copy()

        for s in range(n_states):

            if s in terminal_states:
                continue

            best_val = -float("inf")
            best_option = 0

            for o in range(n_options):

                if s not in options[o]["initiation"]:
                    continue

                q = option_rewards[o][s]

                q += np.dot(option_transitions[o][s], V)

                if q > best_val:
                    best_val = q
                    best_option = o

            newV[s] = best_val
            policy[s] = best_option

            delta = max(delta, abs(newV[s] - V[s]))

        V = newV

        if delta < theta:
            break

    V = [round(float(v), 4) for v in V]

    return V, policy