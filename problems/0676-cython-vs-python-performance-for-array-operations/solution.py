import math

def analyze_array_performance(n: int, operations: list) -> dict:
    """
    Analyze and compare array operation performance across Python, Cython, and NumPy.

    Args:
        n: Array size (number of elements).
        operations: List of dicts containing operation names and performance parameters.

    Returns:
        Dictionary containing per-operation metrics, pipeline totals,
        recommendation, and crossover sizes.
    """
    methods = ["python", "cython", "numpy"]

    results = []
    crossover_sizes = {}

    # Keep unrounded totals for accurate pipeline comparisons.
    pipeline_totals = {method: 0.0 for method in methods}

    for operation in operations:
        name = operation["name"]
        times = {}

        # Calculate execution time in microseconds.
        for method in methods:
            per_elem = operation[f"{method}_per_elem_ns"]
            overhead = operation[f"{method}_overhead_ns"]

            time_ns = overhead + n * per_elem
            times[method] = time_ns / 1000.0

            pipeline_totals[method] += times[method]

        python_time = times["python"]
        cython_time = times["cython"]
        numpy_time = times["numpy"]

        # Calculate speedups relative to Python.
        cython_speedup = (
            python_time / cython_time if cython_time != 0 else float("inf")
        )
        numpy_speedup = (
            python_time / numpy_time if numpy_time != 0 else float("inf")
        )

        # min() preserves the required alphabetical tie-breaking order:
        # cython < numpy < python.
        best = min(
            ["cython", "numpy", "python"],
            key=lambda method: times[method]
        )

        results.append({
            "name": name,
            "python_us": round(python_time, 2),
            "cython_us": round(cython_time, 2),
            "numpy_us": round(numpy_time, 2),
            "cython_speedup": round(cython_speedup, 2),
            "numpy_speedup": round(numpy_speedup, 2),
            "best": best
        })

        # Calculate the smallest array size where NumPy beats Cython.
        cython_pe = operation["cython_per_elem_ns"]
        numpy_pe = operation["numpy_per_elem_ns"]
        cython_oh = operation["cython_overhead_ns"]
        numpy_oh = operation["numpy_overhead_ns"]

        if cython_pe <= numpy_pe:
            crossover = -1
        else:
            ratio = (numpy_oh - cython_oh) / (cython_pe - numpy_pe)
            crossover = 0 if ratio <= 0 else math.ceil(ratio)

        crossover_sizes[name] = crossover

    # Select the fastest overall method, breaking ties alphabetically.
    recommended = min(
        ["cython", "numpy", "python"],
        key=lambda method: pipeline_totals[method]
    )

    return {
        "operations": results,
        "pipeline_python_us": round(pipeline_totals["python"], 2),
        "pipeline_cython_us": round(pipeline_totals["cython"], 2),
        "pipeline_numpy_us": round(pipeline_totals["numpy"], 2),
        "recommended": recommended,
        "crossover_sizes": crossover_sizes
    }