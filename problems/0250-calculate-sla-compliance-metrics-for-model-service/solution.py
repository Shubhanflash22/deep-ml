import numpy as np
def calculate_sla_metrics(requests: list, latency_sla_ms: float = 100.0) -> dict:
    """
    Calculate SLA compliance metrics for a model serving endpoint.
    
    Args:
        requests: list of request results, each a dict with 'latency_ms' and 'status'
        latency_sla_ms: maximum acceptable latency in ms for SLA compliance
    
    Returns:
        dict with keys: 'latency_sla_compliance', 'error_rate', 'overall_sla_compliance'
        All values as percentages (0-100), rounded to 2 decimal places.
    """
    ans = {}
    success = 0
    success_latency = 0
    n = 0
    if len(requests) == 0:
        return {}
    for item in requests:
        n = n + 1
        if item['status'] == "success":
            success = success + 1
            if item['latency_ms']<= latency_sla_ms:
                success_latency = success_latency + 1
    if success == 0:
        success1 = 1
    else:
        success1 = success
    latency_sla_compliance = success_latency/success1
    error_rate = (n-success)/n
    overall_sla_compliance = success_latency/n
    ans = {'latency_sla_compliance': round(latency_sla_compliance * 100, 2), 'error_rate': round(error_rate * 100, 2),
        'overall_sla_compliance': round(overall_sla_compliance * 100, 2)}
    return ans