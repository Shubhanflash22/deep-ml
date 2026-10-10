import numpy as np

class ZeroCopyBatchLoader:
    def __init__(self, data: np.ndarray, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.n_samples, self.n_features = data.shape
        self.batch_size = batch_size

        # Flatten into a contiguous buffer
        self.buffer = np.ascontiguousarray(data).reshape(-1)

    def num_batches(self) -> int:
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def get_batch(self, batch_idx: int) -> np.ndarray:
        if batch_idx < 0 or batch_idx >= self.num_batches():
            raise IndexError("Invalid batch index")

        start_row = batch_idx * self.batch_size
        end_row = min(start_row + self.batch_size, self.n_samples)

        start = start_row * self.n_features
        end = end_row * self.n_features

        return self.buffer[start:end].reshape(
            end_row - start_row, self.n_features
        )

    def is_zero_copy(self, batch_idx: int) -> bool:
        batch = self.get_batch(batch_idx)
        return np.shares_memory(batch, self.buffer)

    def get_batch_means(self) -> list:
        return [
            round(float(np.mean(self.get_batch(i))), 4)
            for i in range(self.num_batches())
        ]

    def write_to_buffer(self, row: int, col: int, value: float) -> None:
        if not (0 <= row < self.n_samples and 0 <= col < self.n_features):
            raise IndexError("Invalid row or column")

        self.buffer[row * self.n_features + col] = value