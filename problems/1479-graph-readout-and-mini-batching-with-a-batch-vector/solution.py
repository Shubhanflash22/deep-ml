import numpy as np


def block_diagonal(mats):
    """Combine square matrices into a block-diagonal matrix."""
    if len(mats) == 0:
        return np.zeros((0, 0))

    rows = sum(m.shape[0] for m in mats)
    cols = sum(m.shape[1] for m in mats)

    result = np.zeros((rows, cols), dtype=np.result_type(*mats))

    row_start = 0
    col_start = 0

    for m in mats:
        r, c = m.shape
        result[row_start:row_start + r, col_start:col_start + c] = m
        row_start += r
        col_start += c

    return result


def batch_graphs(adjs, feats):
    """
    Combine multiple graphs into one disconnected graph.

    Returns:
        A_block_diagonal: Block-diagonal adjacency matrix.
        X_stacked: Concatenated node features.
        batch_vector: Graph index for each node.
    """
    if len(adjs) != len(feats):
        raise ValueError("adjs and feats must contain the same number of graphs")

    if len(adjs) == 0:
        return (
            np.zeros((0, 0)),
            np.zeros((0, 0)),
            np.array([], dtype=int)
        )

    for A, X in zip(adjs, feats):
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Each adjacency matrix must be square")
        if X.ndim != 2 or X.shape[0] != A.shape[0]:
            raise ValueError("Each feature matrix must have one row per node")

    A_block_diagonal = block_diagonal(adjs)
    X_stacked = np.concatenate(feats, axis=0)

    batch_vector = np.concatenate([
        np.full(X.shape[0], graph_idx, dtype=int)
        for graph_idx, X in enumerate(feats)
    ])

    return A_block_diagonal, X_stacked, batch_vector


def readout(H, batch, mode="sum"):
    """
    Aggregate node embeddings into one embedding per graph.

    Args:
        H: Node embeddings, shape (total_nodes, d).
        batch: Graph index for each node.
        mode: "sum", "mean", or "max".

    Returns:
        Graph embeddings, shape (num_graphs, d).
    """
    if H.ndim != 2:
        raise ValueError("H must be a 2D array")

    if batch.ndim != 1 or len(batch) != H.shape[0]:
        raise ValueError("batch must contain one graph index per node")

    if mode not in ("sum", "mean", "max"):
        raise ValueError("mode must be 'sum', 'mean', or 'max'")

    if len(batch) == 0:
        return np.zeros((0, H.shape[1]), dtype=H.dtype)

    if np.any(batch < 0):
        raise ValueError("Graph indices must be nonnegative")

    num_graphs = int(batch.max()) + 1

    if not np.issubdtype(batch.dtype, np.integer):
        if not np.all(batch == np.floor(batch)):
            raise ValueError("Graph indices must be integers")

    result = np.zeros((num_graphs, H.shape[1]), dtype=H.dtype)

    if mode == "max":
        result.fill(-np.inf)
        np.maximum.at(result, batch, H)

        # Handle graphs with no nodes, if graph indices have gaps.
        result[~np.isfinite(result)] = 0
        return result

    np.add.at(result, batch, H)

    if mode == "mean":
        counts = np.bincount(batch, minlength=num_graphs)
        nonempty = counts > 0
        result[nonempty] /= counts[nonempty, None]

    return result


def unbatch(H, batch):
    """Split node embeddings into one array per graph."""
    if H.ndim != 2:
        raise ValueError("H must be a 2D array")

    if batch.ndim != 1 or len(batch) != H.shape[0]:
        raise ValueError("batch must contain one graph index per node")

    if len(batch) == 0:
        return []

    if np.any(batch < 0) or not np.all(batch == np.floor(batch)):
        raise ValueError("Graph indices must be nonnegative integers")

    num_graphs = int(batch.max()) + 1

    return [H[batch == i] for i in range(num_graphs)]