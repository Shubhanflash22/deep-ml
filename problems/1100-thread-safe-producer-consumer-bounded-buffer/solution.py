from collections import deque


def run_producer_consumer(capacity, producer_items, num_consumers):
    buffer = deque()
    consumed = []

    producers = [list(items) for items in producer_items]
    producer_pos = [0] * len(producers)

    # Simulate concurrent producers/consumers.
    while True:
        progress = False

        # Producers add items when there is space.
        for i in range(len(producers)):
            if producer_pos[i] < len(producers[i]) and len(buffer) < capacity:
                buffer.append(producers[i][producer_pos[i]])
                producer_pos[i] += 1
                progress = True

        # Consumers remove available items.
        for _ in range(num_consumers):
            if buffer:
                consumed.append(buffer.popleft())
                progress = True

        # Stop when everything has been produced and consumed.
        all_produced = all(
            producer_pos[i] == len(producers[i])
            for i in range(len(producers))
        )

        if all_produced and not buffer:
            break

        if not progress:
            break

    return sorted(consumed)