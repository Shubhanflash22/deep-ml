def label_encode_ordinal(values: list, order: list) -> list:
    """
    Encode ordinal categorical values to integers based on specified order.
    
    Args:
        values: List of categorical values to encode
        order: List specifying the order of categories from lowest (0) to highest
    
    Returns:
        List of integers representing the encoded values
    """
    m = len(values)
    n = len(order)
    ans = [-1] * m
    for i in range(0,n):
        temp = order[i]
        for j in range(0,m):
            if values[j] == temp:
                ans[j] = i
    return ans