def compute_throughput(intervals):
    """
    Compute step-wise, cumulative, and overall tokens-per-second.

    Args:
        intervals: list of (tokens_processed, elapsed_seconds) tuples.

    Returns:
        dict with keys 'step_tps', 'cumulative_tps', 'overall_tps'.
    """
    if not intervals:
        return {
        'step_tps': [],
        'cumulative_tps': [],
        'overall_tps': 0.0
        }

    step_tps = []
    cumulative_tps = []

    total_tokens = 0
    total_seconds = 0

    for tokens, seconds in intervals:
        if seconds <= 0:
            raise ValueError("Elapsed seconds must be positive")

        if tokens < 0:
            raise ValueError("Tokens processed cannot be negative")

        step_tps.append(round(tokens / seconds, 2))

        total_tokens += tokens
        total_seconds += seconds

        cumulative_tps.append(
            round(total_tokens / total_seconds, 2)
        )

    overall_tps = round(total_tokens / total_seconds, 2)

    return {
        'step_tps': step_tps,
        'cumulative_tps': cumulative_tps,
        'overall_tps': overall_tps
    }
