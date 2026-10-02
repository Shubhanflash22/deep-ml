import numpy as np

def match_anchors(anchors, gt_boxes, pos_threshold=0.5, neg_threshold=0.4):
    """
    Assign each anchor a training label via IoU matching.

    Args:
        anchors: (N, 4) boxes in xyxy format [x1, y1, x2, y2]
        gt_boxes: (M, 4) ground-truth boxes in xyxy format
        pos_threshold: IoU >= this → positive (default 0.5)
        neg_threshold: IoU <  this → negative (default 0.4)

    Returns:
        labels:     (N,) int array with values {1=pos, 0=neg, -1=ignore}
        matched_gt: (N,) int array of matched GT index, or -1
    """
    anchors = np.asarray(anchors)
    gt_boxes = np.asarray(gt_boxes)

    N = len(anchors)
    M = len(gt_boxes)

    if N == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    if M == 0:
        return np.zeros(N, dtype=int), np.full(N, -1, dtype=int)

    # Areas
    aw = np.maximum(0, anchors[:, 2] - anchors[:, 0])
    ah = np.maximum(0, anchors[:, 3] - anchors[:, 1])
    a_area = aw * ah

    gw = np.maximum(0, gt_boxes[:, 2] - gt_boxes[:, 0])
    gh = np.maximum(0, gt_boxes[:, 3] - gt_boxes[:, 1])
    g_area = gw * gh

    # Intersection: (N, M)
    x1 = np.maximum(anchors[:, None, 0], gt_boxes[None, :, 0])
    y1 = np.maximum(anchors[:, None, 1], gt_boxes[None, :, 1])
    x2 = np.minimum(anchors[:, None, 2], gt_boxes[None, :, 2])
    y2 = np.minimum(anchors[:, None, 3], gt_boxes[None, :, 3])

    inter = (
        np.maximum(0, x2 - x1) *
        np.maximum(0, y2 - y1)
    )

    union = a_area[:, None] + g_area[None, :] - inter

    iou = np.divide(
        inter,
        union,
        out=np.zeros_like(inter, dtype=float),
        where=union > 0
    )

    # Best GT for each anchor
    best_gt = np.argmax(iou, axis=1)
    best_iou = np.max(iou, axis=1)

    # Default = ignore
    labels = np.full(N, -1, dtype=int)
    matched_gt = np.full(N, -1, dtype=int)

    # Positive
    pos = best_iou >= pos_threshold
    labels[pos] = 1
    matched_gt[pos] = best_gt[pos]

    # Negative
    neg = best_iou < neg_threshold
    labels[neg] = 0

    # Force every GT to have a positive anchor
    for j in range(M):
        anchor = np.argmax(iou[:, j])
        labels[anchor] = 1
        matched_gt[anchor] = j

    return labels, matched_gt