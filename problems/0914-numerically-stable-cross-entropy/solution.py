import torch

def cross_entropy(logits, targets):
    # Compute log(sum(exp(logits))) in a numerically stable way
    log_normalizer = torch.logsumexp(logits, dim=-1)

    # Get the logit corresponding to the correct class for each sample
    correct_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)

    # Cross-entropy = -correct logit + log(sum(exp(all logits))
    loss = -correct_logits + log_normalizer

    # Return mean loss across the batch
    return loss.mean()