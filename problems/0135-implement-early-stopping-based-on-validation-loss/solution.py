from typing import Tuple

def early_stopping(
    val_losses: list[float],
    patience: int,
    min_delta: float
) -> Tuple[int, int]:

    mini = val_losses[0]   # CHANGED
    temp = 0
    idx = 0
    n = len(val_losses)

    for i in range(1, n):   # CHANGED

        if mini - val_losses[i] > min_delta:   # CHANGED
            mini = val_losses[i]
            idx = i
            temp = 0

        else:
            temp += 1

            if temp == patience:
                return (i, idx)   # CHANGED

    return (n - 1, idx)   # CHANGED