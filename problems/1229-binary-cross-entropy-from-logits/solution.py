import torch
import torch.nn.functional as F

def bce_with_logits(logits, targets):
    """Mean BCE-with-logits loss, numerically stable, rounded to 4 decimals."""
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
    return round(loss.item(), 4)
