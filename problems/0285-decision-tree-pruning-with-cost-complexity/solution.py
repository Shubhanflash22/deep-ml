def compute_pruning_alphas(tree: dict) -> list:
    """
    Computes effective alpha values for cost-complexity pruning.

    Args:
        tree: Dictionary representing a decision tree node with keys:
              - 'samples': number of samples reaching this node
              - 'errors': misclassification count if node becomes a leaf
              - 'left': left child subtree (dict) or None
              - 'right': right child subtree (dict) or None

    Returns:
        List of effective alpha values for internal nodes, sorted ascending.
    """
    alphas = []

    def dfs(node):
        # Leaf node
        if node["left"] is None and node["right"] is None:
            return node["errors"], 1  # (subtree_error, leaf_count)

        # Recurse on children
        left_error, left_leaves = dfs(node["left"])
        right_error, right_leaves = dfs(node["right"])

        subtree_error = left_error + right_error
        leaf_count = left_leaves + right_leaves

        # Effective alpha
        alpha = (node["errors"] - subtree_error) / (leaf_count - 1)
        alphas.append(alpha)

        return subtree_error, leaf_count

    dfs(tree)
    return sorted(alphas)