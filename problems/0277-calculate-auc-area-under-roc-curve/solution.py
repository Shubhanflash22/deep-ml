import numpy as np

def calculate_auc(y_true, y_scores):
    """
    Calculate the Area Under the ROC Curve (AUC).
    
    Args:
        y_true: List or array of binary ground truth labels (0 or 1)
        y_scores: List or array of predicted probabilities or confidence scores
        
    Returns:
        AUC value as a float
    """
    n = len(y_true)

    # 1. Count positive and negative labels
    n_pos = sum(y_true)
    n_neg = n - n_pos

    # 2. Handle edge cases
    if n_pos == 0 or n_neg == 0:
        return 0.0

    # 3. Sort (score, label) pairs by score descending
    pairs = sorted(zip(y_scores, y_true), reverse=True)

    # 4. Initialize ROC state
    tp = 0
    fp = 0
    prev_tpr = 0.0
    prev_fpr = 0.0
    auc = 0.0

    i = 0

    # 5. Process all examples with the same score together
    while i < n:
        score = pairs[i][0]

        while i < n and pairs[i][0] == score:
            label = pairs[i][1]
            if label == 1:
                tp += 1
            else:
                fp += 1
            i += 1

        # 6. Calculate the current ROC point
        tpr = tp / n_pos
        fpr = fp / n_neg

        # 7. Add the trapezoid's area
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2

        prev_tpr = tpr
        prev_fpr = fpr

    return auc