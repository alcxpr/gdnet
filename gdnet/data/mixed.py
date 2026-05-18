from __future__ import annotations

import random

import torch
from torch.utils.data import IterableDataset


class MixedDataset(IterableDataset):
    """Samples from multiple IterableDatasets according to fixed weights.

    At each iteration, picks a source with probability proportional to its
    weight, then yields one item from that source. Exhausted sources are
    restarted automatically, so the dataset is infinite.

    Args:
        datasets: Source datasets to mix.
        weights: Sampling weight for each source (need not sum to 1).
        seed: RNG seed for reproducible mixing order.
    """

    def __init__(
        self,
        datasets: list[IterableDataset],
        weights: list[float],
        seed: int = 42,
    ) -> None:
        assert len(datasets) == len(weights) and len(datasets) > 0
        total = sum(weights)
        self._datasets = datasets
        self._weights = [w / total for w in weights]
        self._seed = seed

    def __iter__(self):
        rng = random.Random(self._seed)
        iters = [iter(ds) for ds in self._datasets]
        while True:
            idx = rng.choices(range(len(iters)), weights=self._weights)[0]
            try:
                yield next(iters[idx])
            except StopIteration:
                iters[idx] = iter(self._datasets[idx])
                yield next(iters[idx])
