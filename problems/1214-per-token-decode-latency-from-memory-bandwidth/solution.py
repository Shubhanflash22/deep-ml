def estimate_decode_latency(num_params, bytes_per_param, bandwidth_bytes_per_s):
    ans = []
    latency_ms = num_params * bytes_per_param * 1000/ bandwidth_bytes_per_s
    tokens_per_sec = 1000/latency_ms
    ans.append(latency_ms)
    ans.append(tokens_per_sec)
    return ans
    # Return [latency_ms, tokens_per_sec]