def dp_drills(weights, values, capacity, m, n, s1, s2):

    # -------------------------
    # 1. 0/1 Knapsack
    # -------------------------
    num_items = len(weights)

    dp = [[0] * (capacity + 1) for _ in range(num_items + 1)]

    for i in range(1, num_items + 1):
        weight = weights[i - 1]
        value = values[i - 1]

        for w in range(capacity + 1):

            # Don't take item
            dp[i][w] = dp[i - 1][w]

            # Take item if it fits
            if weight <= w:
                dp[i][w] = max(
                    dp[i][w],
                    value + dp[i - 1][w - weight]
                )

    knapsack = dp[num_items][capacity]


    # -------------------------
    # 2. Grid Paths
    # -------------------------
    grid = [[0] * n for _ in range(m)]

    # First column
    for i in range(m):
        grid[i][0] = 1

    # First row
    for j in range(n):
        grid[0][j] = 1

    # Remaining cells
    for i in range(1, m):
        for j in range(1, n):
            grid[i][j] = (
                grid[i - 1][j]
                + grid[i][j - 1]
            )

    grid_paths = grid[m - 1][n - 1]


    # -------------------------
    # 3. Longest Common Subsequence
    # -------------------------
    len1 = len(s1)
    len2 = len(s2)

    lcs_dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):

            if s1[i - 1] == s2[j - 1]:
                lcs_dp[i][j] = 1 + lcs_dp[i - 1][j - 1]

            else:
                lcs_dp[i][j] = max(
                    lcs_dp[i - 1][j],
                    lcs_dp[i][j - 1]
                )

    lcs = lcs_dp[len1][len2]


    # -------------------------
    # Complexity
    # -------------------------
    complexity = {
        'knapsack': 'O(n*W) time, O(n*W) space',
        'grid_paths': 'O(m*n) time, O(m*n) space',
        'lcs': 'O(m*n) time, O(m*n) space'
    }

    return {
        'knapsack': knapsack,
        'grid_paths': grid_paths,
        'lcs': lcs,
        'complexity': complexity
    }