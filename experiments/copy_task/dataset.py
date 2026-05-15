from __future__ import annotations

import torch
from torch.utils.data import Dataset

SIGNAL_TOKENS = list(range(10))
NOISE_TOKENS = list(range(10, 50))
RECALL_TOKEN = 50
VOCAB_SIZE = 52


class CopyWithGapDataset(Dataset):
    """Signal tokens followed by a noise gap and a recall token.

    The model must reproduce the signal tokens after seeing the recall token.

    Args:
        n: Number of samples.
        seq_len: Number of signal tokens to copy.
        gap: Number of noise tokens between signal and recall.
    """

    def __init__(self, n: int, seq_len: int = 3, gap: int = 8) -> None:
        signal = torch.randint(0, len(SIGNAL_TOKENS), (n, seq_len))  # type: ignore
        noise = torch.randint(0, len(NOISE_TOKENS), (n, gap)) + len(SIGNAL_TOKENS)  # type: ignore
        recall = torch.full((n, 1), RECALL_TOKEN)  # type: ignore
        self.inputs = torch.cat([signal, noise, recall], dim=1)  # type: ignore
        self.targets = signal

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


class SelectiveCopyDataset(Dataset):
    """Signal tokens interleaved with distractors, followed by a noise gap and recall.

    Args:
        n: Number of samples.
        seq_len: Number of signal tokens to copy.
        gap: Number of noise tokens before recall.
        distractor_ratio: Noise tokens per signal token in the interleaved block.
    """

    def __init__(
        self,
        n: int,
        seq_len: int = 3,
        gap: int = 8,
        distractor_ratio: int = 1,
    ) -> None:
        signal = torch.randint(0, len(SIGNAL_TOKENS), (n, seq_len))  # type: ignore
        distractors = torch.randint(  # type: ignore
            0, len(NOISE_TOKENS), (n, seq_len * distractor_ratio)
        ) + len(SIGNAL_TOKENS)
        interleaved = []
        for i in range(seq_len):
            interleaved.append(signal[:, i : i + 1])
            interleaved.append(
                distractors[:, i * distractor_ratio : (i + 1) * distractor_ratio]
            )
        noise = torch.randint(0, len(NOISE_TOKENS), (n, gap)) + len(SIGNAL_TOKENS)  # type: ignore
        recall = torch.full((n, 1), RECALL_TOKEN)  # type: ignore
        self.inputs = torch.cat([*interleaved, noise, recall], dim=1)  # type: ignore
        self.targets = signal

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]
