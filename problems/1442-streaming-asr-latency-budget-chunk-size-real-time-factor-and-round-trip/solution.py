import math


def chunk_latency(chunk_s, rtf, rtt_s):
    """Total latency for one audio chunk."""
    wait = chunk_s
    model = chunk_s * rtf
    network = rtt_s
    return wait + model + network


def largest_chunk_under(budget_s, rtf, rtt_s, step=0.05):
    if step <= 0:
        raise ValueError("step must be positive")

    chunk_s = step
    largest = None

    while chunk_latency(chunk_s, rtf, rtt_s) <= budget_s:
        largest = round(chunk_s, 10)
        chunk_s += step

    return largest


def concurrent_streams(rtf, gpu_share=1.0):
    """Estimate the number of concurrent streams supported."""
    if rtf <= 0:
        raise ValueError("rtf must be positive")
    if gpu_share < 0:
        raise ValueError("gpu_share must be non-negative")

    return math.floor(gpu_share / rtf)


def end_of_speech_latency(hangover_s, chunk_s, rtf, rtt_s):
    """Latency after the user stops speaking until the result is ready."""
    return hangover_s + chunk_latency(chunk_s, rtf, rtt_s)


def budget_breakdown(chunk_s, rtf, rtt_s):
    """Return wait, model, and network fractions of total latency."""
    if chunk_s < 0 or rtf < 0 or rtt_s < 0:
        raise ValueError("chunk_s, rtf, and rtt_s must be non-negative")

    wait = chunk_s
    model = chunk_s * rtf
    network = rtt_s
    total = wait + model + network

    if total == 0:
        return {
            "wait": 0.0,
            "model": 0.0,
            "network": 0.0,
        }

    return {
        "wait": round(wait / total, 4),
        "model": round(model / total, 4),
        "network": round(network / total, 4),
    }