import numpy as np

def residual_gradient_td(episodes, features, gamma, alpha, n_passes):
    """
    Run the Residual Gradient algorithm for policy evaluation
    with linear function approximation.
    """
    
    n_states, d = features.shape
    
    # Initialize weights
    w = np.zeros(d)
    
    # Training passes
    for _ in range(n_passes):
        for episode in episodes:
            for state, reward, next_state in episode:
                
                # Current state feature vector
                phi_s = features[state]
                
                # Current value estimate
                v_s = np.dot(phi_s, w)
                
                # Handle terminal state
                if next_state == -1:
                    phi_next = np.zeros(d)
                    v_next = 0.0
                else:
                    phi_next = features[next_state]
                    v_next = np.dot(phi_next, w)
                
                # TD error
                delta = reward + gamma * v_next - v_s
                
                # Residual gradient update
                w += alpha * delta * (phi_s - gamma * phi_next)
    
    return w