import numpy as np

def compute_norm(arr: np.ndarray, norm_type: str) -> float:
    """
    Compute the specified norm of the input array.
    
    Args:
        arr: Input numpy array (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', or 'frobenius')
    
    Returns:
        The computed norm as a float
    """
    arr = arr.ravel() 
    ans = 0
    l = len(arr)
    if norm_type == "l1":
        for i in range(0,l):
            ans = ans + abs(arr[i])
    elif norm_type == "l2":
        for i in range(0,l):
            ans = ans + (arr[i])*(arr[i])
        ans = np.sqrt(ans)
    else: 
        for i in range(0,l):
            ans = ans + (arr[i])*(arr[i])
        ans = np.sqrt(ans)
    return ans