import numpy as np

def micro_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the micro-averaged F1 score for multi-label classification.

    Args:
        y_true: Binary indicator array of shape (n_samples, n_labels)
        y_pred: Binary indicator array of shape (n_samples, n_labels)

    Returns:
        Micro-averaged F1 score as a float.
    """
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    a = len(y_true[0])
    b = len(y_true)
    for i in range(0,b):
        for j in range(0,a):
            x = y_true[i][j]
            y = y_pred[i][j]
            if x == 1 and y == 1:
                tp = tp+1
            elif x == 1 and y == 0:
                fn = fn + 1
            elif x == 0 and y ==1:
                fp = fp + 1
            else:
                tn = tn + 1
    
    f1 = (2*tp) / ((2*tp) + fp + fn)
    return f1