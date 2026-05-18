from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset


class PackedTokenDataset(IterableDataset):
    """Reads pre-tokenized uint32 binary files produced by scripts/prepare_data.py.

    Workers stride through non-overlapping windows so there is no duplicate data
    across workers. Multiple paths are iterated sequentially (same as the streaming
    datasets' phase ordering).

    Args:
        paths: One or more .bin paths. Passed as a single str or a list.
        seq_len: Tokens per sample; each yield is seq_len+1 for next-token targets.
    """

    def __init__(self, paths: list[str] | str, seq_len: int) -> None:
        self._paths = [Path(paths)] if isinstance(paths, str) else [Path(p) for p in paths]
        self._seq_len = seq_len

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        chunk = self._seq_len + 1

        for path in self._paths:
            tokens = np.memmap(path, dtype=np.uint32, mode="r")
            n = len(tokens)
            start = worker_id * chunk
            for i in range(start, n - chunk + 1, num_workers * chunk):
                yield torch.from_numpy(tokens[i : i + chunk].astype(np.int64))
