def calculate_batch_health(predictions: list, confidence_threshold: float = 0.5) -> dict:
    """
    Calculate health metrics for a batch prediction job.
    
    Args:
        predictions: list of prediction results, each a dict with 'status' and optionally 'confidence'
        confidence_threshold: threshold below which a prediction is considered low confidence
    
    Returns:
        dict with keys: 'success_rate', 'avg_confidence', 'low_confidence_rate'
        All values as percentages (0-100), rounded to 2 decimal places.
    """
    total = len(predictions)

    if total == 0:
        return {}

    success = 0
    confidence_sum = 0
    confidence_count = 0
    low_confidence = 0

    for item in predictions:
        if item['status'] == "success":
            success += 1

        if 'confidence' in item:
            confidence = item['confidence']
            confidence_sum += confidence
            confidence_count += 1

            if confidence < confidence_threshold:
                low_confidence += 1

    success_rate = success / total

    if confidence_count == 0:
        avg_confidence = 0
        low_confidence_rate = 0
    else:
        avg_confidence = confidence_sum / confidence_count
        low_confidence_rate = low_confidence / confidence_count

    return {
        'success_rate': round(success_rate * 100, 2),
        'avg_confidence': round(avg_confidence * 100, 2),
        'low_confidence_rate': round(low_confidence_rate * 100, 2)
    }