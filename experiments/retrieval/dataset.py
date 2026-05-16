from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

N_KEYS = 16
N_VALUES = 16
N_NOISE_VOCAB = 16
CHUNK_SIZE = 16

KEY_OFFSET = 0
VALUE_OFFSET = N_KEYS
NOISE_OFFSET = N_KEYS + N_VALUES
SIGNAL_TOKEN = NOISE_OFFSET + N_NOISE_VOCAB
QUERY_TOKEN = SIGNAL_TOKEN + 1
VOCAB_SIZE = QUERY_TOKEN + 1  # 51


def _noise_chunk(size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(NOISE_OFFSET, NOISE_OFFSET + N_NOISE_VOCAB, size=size)


def _signal_chunk(
    key: int, value: int, size: int, rng: np.random.Generator
) -> np.ndarray:
    chunk = _noise_chunk(size, rng)
    chunk[0] = SIGNAL_TOKEN
    chunk[1] = KEY_OFFSET + key
    chunk[2] = VALUE_OFFSET + value
    return chunk


def _query_chunk(key: int, size: int, rng: np.random.Generator) -> np.ndarray:
    chunk = _noise_chunk(size, rng)
    chunk[0] = QUERY_TOKEN
    chunk[1] = KEY_OFFSET + key
    return chunk


class RetrievalDataset(Dataset):
    """Retrieval experiment dataset.

    Args:
        n: Number of samples.
        n_slots: Number of CAM buffer slots. Total chunks per sample = n_slots + 1.
        chunk_size: Tokens per chunk.
        condition: 1 = unique key, 2 = ambiguous key (recency), 3 = missing key.
        ambiguous_k: For condition 2, number of noise chunks between old and new signal.
        seed: RNG seed.
    """

    def __init__(
        self,
        n: int,
        n_slots: int,
        chunk_size: int = CHUNK_SIZE,
        condition: int = 1,
        ambiguous_k: int = 2,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        all_chunks = []
        all_targets = []

        for _ in range(n):
            key = int(rng.integers(0, N_KEYS))

            if condition == 1:
                value = int(rng.integers(0, N_VALUES))
                chunks = [_signal_chunk(key, value, chunk_size, rng)]
                for _ in range(n_slots - 1):
                    chunks.append(_noise_chunk(chunk_size, rng))
                chunks.append(_query_chunk(key, chunk_size, rng))
                target = value

            elif condition == 2:
                v_old = int(rng.integers(0, N_VALUES))
                v_new = int(rng.integers(0, N_VALUES - 1))
                if v_new >= v_old:
                    v_new += 1
                k = ambiguous_k
                n_trailing = n_slots - k - 2
                chunks = [_signal_chunk(key, v_old, chunk_size, rng)]
                for _ in range(k):
                    chunks.append(_noise_chunk(chunk_size, rng))
                chunks.append(_signal_chunk(key, v_new, chunk_size, rng))
                for _ in range(n_trailing):
                    chunks.append(_noise_chunk(chunk_size, rng))
                chunks.append(_query_chunk(key, chunk_size, rng))
                target = v_new

            elif condition == 3:
                chunks = []
                for _ in range(n_slots):
                    chunks.append(_noise_chunk(chunk_size, rng))
                chunks.append(_query_chunk(key, chunk_size, rng))
                target = -1  # unused in training; only R and entropy are evaluated

            else:
                raise ValueError(f"unknown condition {condition}")

            all_chunks.append(np.stack(chunks))
            all_targets.append(target)

        self.chunks = torch.tensor(np.array(all_chunks), dtype=torch.long)  # type: ignore
        self.targets = torch.tensor(all_targets, dtype=torch.long)  # type: ignore

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.chunks[idx], self.targets[idx]
