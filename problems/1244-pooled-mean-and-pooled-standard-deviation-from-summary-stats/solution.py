import numpy as np
def pooled_stats(m1, s1, n1, m2, s2, n2):
    # m1, s1, n1: mean, sample sd, size of group 1
    # m2, s2, n2: mean, sample sd, size of group 2
    # return (combined_mean, combined_sd)
    cmean = (n1*m1 + n2*m2) / (n1+n2)
    a = (n1 - 1)*s1*s1 
    b = (n2 - 1)*s2*s2
    c = n1 * (m1 - cmean) * (m1 - cmean)
    d = n2 * (m2 - cmean) * (m2 - cmean)
    e = n1 + n2 - 1
    csd = np.sqrt((a + b + c + d)/e)
    return (cmean, csd)