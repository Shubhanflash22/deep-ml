import numpy as np

def fit_sigmoid_scaling(nll_list, acc_list, predict_nll):
    """
    Fit acc = 1 / (1 + exp(a*nll + b)) via least squares in logit space and
    predict accuracy at predict_nll.

    Returns:
        [a, b, predicted_acc] as a list of floats.
    """
    nll = np.asarray(nll_list, dtype=float)
    acc = np.asarray(acc_list, dtype=float)

    # Transform accuracy into logit space:
    # z = log((1 - acc) / acc) = a*nll + b
    z = np.log((1 - acc) / acc)

    # Design matrix: [nll, 1]
    X = np.column_stack([nll, np.ones_like(nll)])

    # Ordinary least squares
    a, b = np.linalg.lstsq(X, z, rcond=None)[0]

    # Predict using the fitted sigmoid
    predicted_acc = 1 / (1 + np.exp(a * predict_nll + b))

    return [float(a), float(b), float(predicted_acc)]