import numpy as np

def mutual_information(joint_prob: list[list[float]]) -> float:
    joint_prob = np.array(joint_prob)

    # P(X): row sums
    px = np.sum(joint_prob, axis=1)

    # P(Y): column sums
    py = np.sum(joint_prob, axis=0)

    mi = 0.0

    for i in range(joint_prob.shape[0]):
        for j in range(joint_prob.shape[1]):
            pxy = joint_prob[i, j]

            if pxy > 0:
                mi += pxy * np.log(pxy / (px[i] * py[j]))

    return mi