"""RoPE on conv input ablation -- ordered recall task.

Compares baseline GDNet vs GDNet+RoPE on copy-with-gap. Both token-level
and sequence-level accuracy are measured. The gap between them is the
positional signal: a model that knows content but not order will have high
token-level and low sequence-level accuracy.

Usage:
    python experiments/copy_task/run_rope.py
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

from gdnet.kernel.gated_causal_depthwise_conv import gated_causal_depthwise_conv
from gdnet.layer import GDLayer, _sync_sn
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


def rope(x: torch.Tensor) -> torch.Tensor:
    B, T, d = x.shape
    half = d // 2
    theta = 1.0 / (
        10000.0 ** (torch.arange(0, half, device=x.device, dtype=x.dtype) / half)  # type: ignore
    )
    pos = torch.arange(T, device=x.device, dtype=x.dtype)  # type: ignore
    angles = torch.outer(pos, theta)  # type: ignore
    cos = angles.cos().unsqueeze(0)
    sin = angles.sin().unsqueeze(0)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)  # type: ignore


class GDLayerRoPE(GDLayer):
    """GDLayer with RoPE applied to the conv input on both fwd and bwd passes."""

    def fwd_step(
        self, fwd: torch.Tensor, side: torch.Tensor, return_gate: bool = False
    ) -> tuple[torch.Tensor, ...]:
        x = rope(fwd)
        fwd_t = self.conv_fwd(x)
        R = self._recovery("rf", side, fwd_t)
        _sync_sn(self.gf_W1)  # type: ignore
        fwd_new, side_new = gated_causal_depthwise_conv(
            x,
            fwd_t,
            side,
            R,
            self.conv_fwd.conv.weight.squeeze(1),
            self.gf_W1.weight,  # type: ignore
            self.gf_W1.bias,  # type: ignore
            self.gf_norm.weight,  # type: ignore
            self.gf_W2.weight,  # type: ignore
            self.gf_W2.bias,  # type: ignore
        )
        if return_gate:
            return fwd_new, side_new, self._gate("gf", fwd_t)
        return fwd_new, side_new

    def bwd_step(
        self, fwd: torch.Tensor, side: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = rope(fwd)
        fwd_t = self.conv_bwd(x)
        R = self._recovery("rb", side, fwd_t)
        _sync_sn(self.gb_W1)  # type: ignore
        return gated_causal_depthwise_conv(
            x,
            fwd_t,
            side,
            R,
            self.conv_bwd.conv.weight.squeeze(1),
            self.gb_W1.weight,  # type: ignore
            self.gb_W1.bias,  # type: ignore
            self.gb_norm.weight,  # type: ignore
            self.gb_W2.weight,  # type: ignore
            self.gb_W2.bias,  # type: ignore
        )


def make_model(gap: int, use_rope: bool) -> GDNet:
    chunk_size = SEQ_LEN + gap + 1
    model = GDNet(
        vocab_size=VOCAB_SIZE,
        d_embed=D,
        d=D,
        n_layers=N_LAYERS,
        n_cycles=N_CYCLES,
        chunk_size=chunk_size,
        kernel_size=KERNEL_SIZE,
    )
    if use_rope:
        model.layers = nn.ModuleList(
            [GDLayerRoPE(D, KERNEL_SIZE) for _ in range(N_LAYERS)]
        )
    return model


class GDNetCopy(nn.Module):
    def __init__(self, gap: int, use_rope: bool) -> None:
        super().__init__()
        self.gdnet = make_model(gap, use_rope)
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


def run_gap(gap: int, use_rope: bool) -> tuple[float, float]:
    train_dl = DataLoader(
        CopyWithGapDataset(N_TRAIN, SEQ_LEN, gap), batch_size=BATCH_SIZE, shuffle=True
    )
    val_dl = DataLoader(CopyWithGapDataset(N_VAL, SEQ_LEN, gap), batch_size=256)

    model = GDNetCopy(gap, use_rope).to(DEVICE)
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
            label = "rope" if use_rope else "base"
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
        "rope": {},
    }

    for gap in gaps:
        print(f"\n--- gap={gap} ---")
        results["baseline"][gap] = run_gap(gap, use_rope=False)
        results["rope"][gap] = run_gap(gap, use_rope=True)
        b_tok, b_seq = results["baseline"][gap]
        r_tok, r_seq = results["rope"][gap]
        print(f"  baseline  tok={b_tok:.3f}  seq={b_seq:.3f}")
        print(f"  rope      tok={r_tok:.3f}  seq={r_seq:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, idx in [
        (axes[0], "token-level", 0),
        (axes[1], "sequence-level", 1),
    ]:
        for label, color in [("baseline", "steelblue"), ("rope", "darkorange")]:
            xs = sorted(results[label])
            ys = [results[label][g][idx] for g in xs]
            ax.plot(xs, ys, marker="o", label=label, color=color)
        ax.axvline(
            RECEPTIVE_FIELD,
            color="red",
            linestyle="--",
            alpha=0.5,
            label=f"RF={RECEPTIVE_FIELD}",
        )
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("gap length (tokens)")
        ax.set_ylabel("accuracy")
        ax.set_title(f"{metric} accuracy")
        ax.set_ylim(0, 1.05)
        ax.legend()

    fig.suptitle("GDNet: baseline vs RoPE on conv input - ordered recall")
    fig.tight_layout()
    out = Path(__file__).parent / "copy_gap_rope.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
