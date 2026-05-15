"""Copy-with-gap ablation for GDNet positional encoding.

Establishes the receptive-field accuracy cliff as a baseline. Once RoPE /
per-step S_l PE are implemented, re-run with those variants to measure
improvement.

Usage:
    python experiments/copy_task/run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import VOCAB_SIZE, CopyWithGapDataset
from torch.utils.data import DataLoader

from gdnet.model import GDNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore

SEQ_LEN = 3
N_TRAIN = 8_000
N_VAL = 2_000
BATCH_SIZE = 128
EPOCHS = 80
LR = 1e-3
N_CYCLES = 3
N_LAYERS = 3
KERNEL_SIZE = 7
D = 64

RECEPTIVE_FIELD = N_CYCLES * KERNEL_SIZE * N_LAYERS  # 63


class GDNetCopy(nn.Module):
    """GDNet with a multi-token copy head reading from the recall token position.

    Args:
        gap: Number of noise tokens. Determines chunk_size for GDNet.
        seq_len: Number of signal tokens to predict.
    """

    def __init__(self, gap: int, seq_len: int) -> None:
        super().__init__()
        chunk_size = seq_len + gap + 1
        self.gdnet = GDNet(
            vocab_size=VOCAB_SIZE,
            d_embed=D,
            d=D,
            n_layers=N_LAYERS,
            n_cycles=N_CYCLES,
            chunk_size=chunk_size,
            kernel_size=KERNEL_SIZE,
        )
        self.heads = nn.ModuleList([nn.Linear(D, VOCAB_SIZE) for _ in range(seq_len)])
        self.seq_len = seq_len
        self._repr: torch.Tensor | None = None

        # capture norm_out output (B, T, d) to avoid re-exposing internals
        self.gdnet.norm_out.register_forward_hook(self._hook)

    def _hook(self, _, __, output: torch.Tensor) -> None:
        self._repr = output

    def forward(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        self.gdnet(tokens)
        assert self._repr is not None
        recall = self._repr[:, -1, :]  # recall token is always last
        return [h(recall) for h in self.heads]


def compute_accuracy(logits: list[torch.Tensor], targets: torch.Tensor) -> float:
    correct = sum(
        (logits[i].argmax(-1) == targets[:, i]).sum().item() for i in range(len(logits))
    )
    return correct / targets.numel()


def run_gap(gap: int) -> float:
    train_dl = DataLoader(
        CopyWithGapDataset(N_TRAIN, SEQ_LEN, gap), batch_size=BATCH_SIZE, shuffle=True
    )
    val_dl = DataLoader(CopyWithGapDataset(N_VAL, SEQ_LEN, gap), batch_size=256)

    model = GDNetCopy(gap, SEQ_LEN).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print(
        f"  gap={gap:3d}  params={sum(p.numel() for p in model.parameters()):,}"
        f"  receptive_field={RECEPTIVE_FIELD}"
    )

    final_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        for tokens, targets in train_dl:
            tokens, targets = tokens.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            logits = model(tokens)
            loss = sum(
                F.cross_entropy(logits[i], targets[:, i]) for i in range(SEQ_LEN)
            )
            loss = loss / SEQ_LEN
            loss.backward()  # type: ignore
            optimizer.step()

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                accs = []
                for tokens, targets in val_dl:
                    tokens, targets = tokens.to(DEVICE), targets.to(DEVICE)
                    accs.append(compute_accuracy(model(tokens), targets))
            final_acc = float(np.mean(accs))
            print(f"  gap={gap:3d}  epoch={epoch + 1:3d}  acc={final_acc:.3f}")
            model.train()
            if final_acc >= 0.99:
                print(f"  gap={gap:3d}  solved at epoch {epoch + 1}")
                return final_acc

    return final_acc


def main() -> None:
    gaps = [0, 8, 16, 32, 48, 64, 96, 128]
    results: dict[int, float] = {}

    print(f"Receptive field: {RECEPTIVE_FIELD} tokens")
    print(f"Expected cliff around gap ~ {RECEPTIVE_FIELD - SEQ_LEN}\n")

    for gap in gaps:
        print(f"\n--- gap={gap} ---")
        results[gap] = run_gap(gap)
        print(f"  gap={gap:3d}  FINAL acc={results[gap]:.3f}")

    gaps_sorted = sorted(results)
    accs = [results[g] for g in gaps_sorted]

    plt.figure(figsize=(10, 5))
    plt.plot(gaps_sorted, accs, marker="o")
    plt.axvline(
        RECEPTIVE_FIELD,
        color="red",
        linestyle="--",
        alpha=0.6,
        label=f"receptive field ({RECEPTIVE_FIELD})",
    )
    plt.xlabel("gap length (tokens)")
    plt.ylabel("accuracy")
    plt.title("GDNet copy-with-gap accuracy vs gap (baseline, no PE)")
    plt.ylim(0, 1.05)
    plt.axhline(1.0, color="gray", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    out = Path(__file__).parent / "copy_gap_baseline.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
