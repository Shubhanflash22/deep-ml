def rejection_sampling_best_of_k(candidates, scores):
    """
    Select the highest-scoring candidate per prompt.

    Args:
        candidates: list of N lists, each containing K candidate outputs.
        scores: list of N lists, each containing K reward scores.

    Returns:
        List of N selected candidates.
    """
    n = len(candidates)
    ans = []
    for i in range(0,n):
        k = len(candidates[i])
        temp = 0
        maxi = scores[i][0]
        for j in range(1,k):
            if scores[i][j]>maxi:
                temp = j
                maxi = scores[i][j]
        ans.append(candidates[i][temp])
    return ans