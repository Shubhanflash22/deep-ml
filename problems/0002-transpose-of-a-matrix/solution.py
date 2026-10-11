def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
        
    Returns:
        The transposed matrix of shape (n, m)
    """
    # Handle the edge case of an empty matrix
    if not a or not a[0]:
        return []
        
    num_rows = len(a)
    num_cols = len(a[0])
    
    # 1. Initialize an empty matrix of the target shape (n x m)
    transposed = []
    
    # 2. Loop through each column of the original matrix
    for col in range(num_cols):
        new_row = []
        # 3. Loop through each row of the original matrix
        for row in range(num_rows):
            new_row.append(a[row][col])
        # 4. Add the newly built row to our transposed matrix
        transposed.append(new_row)
        
    return transposed
