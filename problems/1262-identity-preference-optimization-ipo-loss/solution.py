import numpy as np

def ipo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta):

    policy_chosen_logps = np.asarray(policy_chosen_logps)
    policy_rejected_logps = np.asarray(policy_rejected_logps)
    ref_chosen_logps = np.asarray(ref_chosen_logps)
    ref_rejected_logps = np.asarray(ref_rejected_logps)

    policy_log_ratio = policy_chosen_logps - policy_rejected_logps
    ref_log_ratio = ref_chosen_logps - ref_rejected_logps

    logits = policy_log_ratio - ref_log_ratio

    loss = (logits - 1 / (2 * beta)) ** 2

    return round(float(np.mean(loss)), 6)