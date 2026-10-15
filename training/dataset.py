import numpy as np
import torch


class TokenDataset:

    def __init__(
        self,
        file_path,
        block_size=512
    ):

        self.data = np.memmap(
            file_path,
            dtype=np.uint16,
            mode="r"
        )

        self.block_size = block_size

        print(
            f"Loaded {len(self.data):,} tokens "
            f"from {file_path}"
        )

    def get_batch(
        self,
        batch_size,
        device
    ):

        max_index = (
            len(self.data)
            - self.block_size
            - 1
        )

        indexes = torch.randint(
            0,
            max_index,
            (batch_size,)
        )

        x = torch.stack([
            torch.from_numpy(
                self.data[
                    i:
                    i + self.block_size
                ].astype(np.int64)
            )
            for i in indexes
        ])

        y = torch.stack([
            torch.from_numpy(
                self.data[
                    i + 1:
                    i + self.block_size + 1
                ].astype(np.int64)
            )
            for i in indexes
        ])

        x = x.to(
            device,
            non_blocking=True
        )

        y = y.to(
            device,
            non_blocking=True
        )

        return x, y