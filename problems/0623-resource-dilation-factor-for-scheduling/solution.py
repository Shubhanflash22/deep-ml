def compute_dilation_factor(tasks: list, capacity: int) -> dict:
    n = len(tasks)

    # 1. Critical path
    memo = {}

    def critical_path(i):
        if i in memo:
            return memo[i]

        deps = tasks[i]["dependencies"]

        if not deps:
            memo[i] = tasks[i]["duration"]
        else:
            memo[i] = tasks[i]["duration"] + max(
                critical_path(dep) for dep in deps
            )

        return memo[i]

    critical_path_length = max(
        (critical_path(i) for i in range(n)),
        default=0
    )

    # 2. Total work
    total_work = sum(
        task["duration"] * task["resources"]
        for task in tasks
    )

    # 3. Lower bound
    lower_bound = max(
        critical_path_length,
        total_work / capacity
    )

    # 4. Greedy scheduling
    start_times = [None] * n
    finish_times = [None] * n

    running = []
    completed = set()
    scheduled = set()

    current_time = 0.0
    remaining_resources = capacity

    while len(completed) < n:

        # Complete tasks at current time
        for finish_time, task_idx in running[:]:
            if finish_time <= current_time:
                completed.add(task_idx)
                remaining_resources += tasks[task_idx]["resources"]
                running.remove((finish_time, task_idx))

        # Schedule ready tasks in index order
        for i in range(n):
            if i in scheduled:
                continue

            task = tasks[i]

            if not all(
                dep in completed
                for dep in task["dependencies"]
            ):
                continue

            if task["resources"] > remaining_resources:
                continue

            start_times[i] = current_time
            finish_time = current_time + task["duration"]

            finish_times[i] = finish_time
            scheduled.add(i)

            remaining_resources -= task["resources"]
            running.append((finish_time, i))

        # Move to next completion
        if running:
            current_time = min(
                finish_time for finish_time, _ in running
            )
        elif len(scheduled) < n:
            raise ValueError("Tasks contain a dependency cycle.")

    actual_makespan = max(finish_times, default=0.0)

    dilation_factor = (
        actual_makespan / lower_bound
        if lower_bound > 0
        else 0.0
    )

    return {
        "dilation_factor": float(round(dilation_factor, 4)),
        "actual_makespan": float(round(actual_makespan, 4)),
        "lower_bound": float(round(lower_bound, 4)),
        "start_times": [float(round(t, 4)) for t in start_times]
    }