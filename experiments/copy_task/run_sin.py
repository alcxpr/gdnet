"""Sinusoidal absolute PE injection into S_l -- ordered recall ablation.

Per-step injection: S_l[t] += PE(t) before each fwd_step. Must be per-step
because a one-time injection at t=0 decays at rate (1-R)^t and is gone
within ~10 positions at R~0.5.

Usage:
    python experiments/copy_task/run_sin.py
"""
from __future__ import annotations

import math
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


def sinusoidal_pe(T: int, d: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(T, device=device).unsqueeze(1).float()
    i = torch.arange(0, d, 2, device=device).float()
    div = torch.exp(i * (-math.log(10000.0) / d))
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe  # (T, d)


class GDNetSinPE(GDNet):
    """GDNet with per-step sinusoidal PE injected into every side stream layer."""

    def one_cycle(
        self,
        fwd: torch.Tensor,
        side: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        B, T, d = fwd.shape
        pe = sinusoidal_pe(T, d, fwd.device).unsqueeze(0)  # (1, T, d)
        for i, layer in enumerate(self.layers):
            fwd, side[i] = layer.fwd_step(fwd, side[i] + pe)  # type: ignore
        for i, layer in reversed(list(enumerate(self.layers))):
            fwd, side[i] = layer.bwd_step(fwd, side[i] + pe)  # type: ignore
        return fwd, side


class GDNetCopy(nn.Module):
    def __init__(self, gap: int, use_sin: bool) -> None:
        super().__init__()
        chunk_size = SEQ_LEN + gap + 1
        cls = GDNetSinPE if use_sin else GDNet
        self.gdnet = cls(
            vocab_size=VOCAB_SIZE,
            d_embed=D,
            d=D,
            n_layers=N_LAYERS,
            n_cycles=N_CYCLES,
            chunk_size=chunk_size,
            kernel_size=KERNEL_SIZE,
        )
        self.heads = nn.ModuleList([nn.Linear(D, VOCAB_SIZE) for _ in range(SEQ_LEN)])
        self._repr: torch.Tensor | None = None
        self.gdnet.norm_out.register_forward_hook(self._hook)

    def _hook(self, _, __, output: torch.Tensor) -> None:
        self._repr = output

    def forward(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        self.gdnet(tokens)
        assert self._repr is not None
        recall = self._repr[:, -1, :]
        return [h(recall) for h in self.heads]


def compute_metrics(
    logits: list[torch.Tensor], targets: torch.Tensor
) -> tuple[float, float]:
    preds = torch.stack([logits[i].argmax(-1) for i in range(SEQ_LEN)], dim=1)
    token_correct = (preds == targets).float().mean().item()
    seq_correct = (preds == targets).all(dim=1).float().mean().item()
    return token_correct, seq_correct


def run_gap(gap: int, use_sin: bool) -> tuple[float, float]:
    train_dl = DataLoader(
        CopyWithGapDataset(N_TRAIN, SEQ_LEN, gap), batch_size=BATCH_SIZE, shuffle=True
    )
    val_dl = DataLoader(CopyWithGapDataset(N_VAL, SEQ_LEN, gap), batch_size=256)

    model = GDNetCopy(gap, use_sin).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    final_tok, final_seq = 0.0, 0.0
    for epoch in range(EPOCHS):
        model.train()
        for tokens, targets in train_dl:
            tokens, targets = tokens.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            logits = model(tokens)
            loss = sum(
                F.cross_entropy(logits[i], targets[:, i]) for i in range(SEQ_LEN)
            )
            (loss / SEQ_LEN).backward()  # type: ignore
            optimizer.step()

        if (epoch + 1) % 20 == 0:
            model.eval()
            tok_accs, seq_accs = [], []
            with torch.no_grad():
                for tokens, targets in val_dl:
                    tokens, targets = tokens.to(DEVICE), targets.to(DEVICE)
                    t, s = compute_metrics(model(tokens), targets)
                    tok_accs.append(t)
                    seq_accs.append(s)
            final_tok = float(np.mean(tok_accs))
            final_seq = float(np.mean(seq_accs))
            label = "sin" if use_sin else "base"
            print(
                f"  gap={gap:3d}  [{label}]  epoch={epoch + 1:3d}"
                f"  tok={final_tok:.3f}  seq={final_seq:.3f}"
            )
            model.train()
            if final_seq >= 0.99:
                print(f"  gap={gap:3d}  [{label}]  solved at epoch {epoch + 1}")
                return final_tok, final_seq

    return final_tok, final_seq


def main() -> None:
    gaps = [0, 32, 96, 128]
    results: dict[str, dict[int, tuple[float, float]]] = {
        "baseline": {},
        "sin_pe": {},
    }

    for gap in gaps:
        print(f"\n--- gap={gap} ---")
        results["baseline"][gap] = run_gap(gap, use_sin=False)
        results["sin_pe"][gap] = run_gap(gap, use_sin=True)
        b_tok, b_seq = results["baseline"][gap]
        s_tok, s_seq = results["sin_pe"][gap]
        print(f"  baseline  tok={b_tok:.3f}  seq={b_seq:.3f}")
        print(f"  sin_pe    tok={s_tok:.3f}  seq={s_seq:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, idx in [
        (axes[0], "token-level", 0),
        (axes[1], "sequence-level", 1),
    ]:
        for label, color in [("baseline", "steelblue"), ("sin_pe", "darkorange")]:
            xs = sorted(results[label])
            ys = [results[label][g][idx] for g in xs]
            ax.plot(xs, ys, marker="o", label=label, color=color)
        ax.axvline(RECEPTIVE_FIELD, color="red", linestyle="--", alpha=0.5,
                   label=f"RF={RECEPTIVE_FIELD}")
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("gap length (tokens)")
        ax.set_ylabel("accuracy")
        ax.set_title(f"{metric} accuracy")
        ax.set_ylim(0, 1.05)
        ax.legend()

    fig.suptitle("GDNet: baseline vs per-step sinusoidal PE in S_l -- ordered recall")
    fig.tight_layout()
    out = Path(__file__).parent / "copy_gap_sin.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
