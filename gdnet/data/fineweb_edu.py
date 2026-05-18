from __future__ import annotations

import tiktoken
import torch
from torch.utils.data import IterableDataset


class FineWebEduDataset(IterableDataset):
    """Streams FineWeb-Edu, filters by int_score, packs tokens into fixed-length chunks.

    All ranks should use the same seed so every rank sees identical token sequences.
    The training loop is responsible for slicing T_local per rank for sequence parallelism.

    Filtering uses the metadata fields directly (no text scan):
      - int_score: 3-5 in FineWeb-Edu; use min/max to select difficulty tier.
      - token_count: precomputed approximate count; used only as a cheap pre-filter.

    Args:
        seq_len: Output chunk length (loader yields seq_len+1 for next-token targets).
        min_int_score: Minimum int_score to accept (inclusive).
        max_int_score: Maximum int_score to accept (inclusive).
        encoding: tiktoken encoding name.
        subset: FineWeb-Edu HF config name, e.g. "sample-10BT".
        min_token_count: Skip documents with metadata token_count below this threshold.
        seed: Streaming shuffle seed.
        buffer_size: HF streaming shuffle buffer size.
    """

    def __init__(
        self,
        seq_len: int,
        min_int_score: int,
        max_int_score: int,
        encoding: str = "cl100k_base",
        subset: str = "sample-10BT",
        min_token_count: int = 128,
        seed: int = 42,
        buffer_size: int = 10_000,
    ) -> None:
        self._seq_len = seq_len
        self._min_score = min_int_score
        self._max_score = max_int_score
        self._encoding = encoding
        self._subset = subset
        self._min_token_count = min_token_count
        self._seed = seed
        self._buffer_size = buffer_size

    def __iter__(self):
        from datasets import load_dataset  # type: ignore

        enc = tiktoken.get_encoding(self._encoding)
        eos = enc.eot_token
        target_len = self._seq_len + 1

        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=self._subset,
            split="train",
            streaming=True,
        ).shuffle(seed=self._seed, buffer_size=self._buffer_size)

        buf: list[int] = []
        for sample in ds:
            if not (self._min_score <= sample["int_score"] <= self._max_score):
                continue
            if sample["token_count"] < self._min_token_count:
                continue
            buf.extend(enc.encode_ordinary(sample["text"]))
            buf.append(eos)
            while len(buf) >= target_len:
                yield torch.tensor(buf[:target_len], dtype=torch.long)  # type: ignore
                del buf[: self._seq_len]
