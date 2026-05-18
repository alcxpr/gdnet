from __future__ import annotations

import tiktoken
import torch
from torch.utils.data import IterableDataset


class NemotronDataset(IterableDataset):
    """Streams Nemotron-Pretraining-Specialized-v1.1, packs tokens into fixed-length chunks.

    Unlike FineWeb-Edu there is no precomputed token_count field, so length
    filtering uses len(text) as a cheap character-count proxy. Set min_chars=0
    to disable the pre-filter and pack everything.

    Args:
        seq_len: Output chunk length (loader yields seq_len+1 for next-token targets).
        encoding: tiktoken encoding name.
        subset: HF config name, e.g. "everything".
        min_chars: Skip documents shorter than this many characters.
        seed: Streaming shuffle seed.
        buffer_size: HF streaming shuffle buffer size.
    """

    def __init__(
        self,
        seq_len: int,
        encoding: str = "cl100k_base",
        subset: str = "everything",
        min_chars: int = 0,
        seed: int = 42,
        buffer_size: int = 10_000,
    ) -> None:
        self._seq_len = seq_len
        self._encoding = encoding
        self._subset = subset
        self._min_chars = min_chars
        self._seed = seed
        self._buffer_size = buffer_size

    def __iter__(self):
        from datasets import load_dataset  # type: ignore

        worker_info = torch.utils.data.get_worker_info()
        seed = self._seed + (worker_info.id if worker_info is not None else 0)

        enc = tiktoken.get_encoding(self._encoding)
        eos = enc.eot_token
        target_len = self._seq_len + 1

        repo = "nvidia/Nemotron-Pretraining-Specialized-v1.1"
        if self._subset == "everything":
            from datasets import (  # type: ignore
                concatenate_datasets,
                get_dataset_config_names,
            )

            configs = get_dataset_config_names(repo)
            ds = concatenate_datasets(
                [
                    load_dataset(repo, name=c, split="train", streaming=True)
                    for c in configs
                ]
            )
        else:
            ds = load_dataset(repo, name=self._subset, split="train", streaming=True)

        ds = ds.shuffle(seed=seed, buffer_size=self._buffer_size)

        buf: list[int] = []
        for sample in ds:
            text = sample["text"]
            if self._min_chars > 0 and len(text) < self._min_chars:
                continue
            buf.extend(enc.encode_ordinary(text))
            buf.append(eos)
            while len(buf) >= target_len:
                yield torch.tensor(buf[:target_len], dtype=torch.long)  # type: ignore
                del buf[: self._seq_len]
