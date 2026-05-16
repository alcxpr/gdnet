"""CAM pos_gate retrieval experiment.

Three conditions evaluated:
  C1  unique key    -- signal written once; tests content retrieval
  C2  ambiguous key -- same key at two recencies; tests pos_gate recency bias
  C3  missing key   -- key never written; tests implicit not-found via R

Baseline (ContentCAM) uses softmax over raw tag similarity.
Proposed (PosGateCAM) adds input-conditioned slot position embeddings.

Metrics per condition:
  accuracy     -- fraction of correct value predictions (C1, C2 only)
  mean R       -- recovery gate magnitude at recall position
  weight H     -- retrieval weight entropy -sum(w log w); high = uncertain

Usage:
    python experiments/retrieval/run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from dataset import N_VALUES, VOCAB_SIZE, RetrievalDataset
from model import RetrievalModel
from torch.utils.data import ConcatDataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore

N_TRAIN = 10_000
N_VAL = 2_000
BATCH_SIZE = 128
EPOCHS = 60
LR = 1e-3

N_SLOTS = 8
CHUNK_SIZE = 16
D = 32
D_SIG = 8
D_C = 8
N_LAYERS = 1
KERNEL_SIZE = 7
N_CYCLES = 2  # RF = n_cycles * kernel_size * n_layers = 14, covers chunk_size=16
AMBIGUOUS_K = 2  # noise chunks between old and new signal in C2


def _entropy(w: torch.Tensor) -> torch.Tensor:
    return -(w * (w + 1e-8).log()).sum(-1)


def _eval(
    model: RetrievalModel,
    ds: RetrievalDataset,
    has_target: bool = True,
) -> tuple[float, float, float]:
    dl = DataLoader(ds, batch_size=256)
    accs, rs, ents = [], [], []
    model.eval()
    with torch.no_grad():
        for chunks, targets in dl:
            chunks = chunks.to(DEVICE)
            logits, r_val, w_val = model(chunks)
            if has_target:
                preds = logits.argmax(-1).cpu()
                accs.append((preds == targets).float().mean().item())
            rs.append(r_val.cpu().mean().item())
            ents.append(_entropy(w_val).cpu().mean().item())
    return (
        float(np.mean(accs)) if accs else float("nan"),
        float(np.mean(rs)),
        float(np.mean(ents)),
    )


def run(use_pos_gate: bool) -> dict[str, list]:
    train_ds = ConcatDataset(
        [
            RetrievalDataset(
                N_TRAIN // 2,
                N_SLOTS,
                CHUNK_SIZE,
                condition=1,
                ambiguous_k=AMBIGUOUS_K,
                seed=0,
            ),
            RetrievalDataset(
                N_TRAIN // 2,
                N_SLOTS,
                CHUNK_SIZE,
                condition=2,
                ambiguous_k=AMBIGUOUS_K,
                seed=1,
            ),
        ]
    )
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    val_c1 = RetrievalDataset(
        N_VAL, N_SLOTS, CHUNK_SIZE, condition=1, ambiguous_k=AMBIGUOUS_K, seed=10
    )
    val_c2 = RetrievalDataset(
        N_VAL, N_SLOTS, CHUNK_SIZE, condition=2, ambiguous_k=AMBIGUOUS_K, seed=11
    )
    val_c3 = RetrievalDataset(
        N_VAL, N_SLOTS, CHUNK_SIZE, condition=3, ambiguous_k=AMBIGUOUS_K, seed=12
    )

    model = RetrievalModel(
        vocab_size=VOCAB_SIZE,
        n_values=N_VALUES,
        d=D,
        d_sig=D_SIG,
        d_c=D_C,
        n_slots=N_SLOTS,
        n_layers=N_LAYERS,
        kernel_size=KERNEL_SIZE,
        n_cycles=N_CYCLES,
        use_pos_gate=use_pos_gate,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    hist: dict[str, list] = {
        k: []
        for k in [
            "epoch",
            "c1_acc",
            "c2_acc",
            "c1_r",
            "c2_r",
            "c3_r",
            "c1_h",
            "c2_h",
            "c3_h",
        ]
    }

    label = "pos_gate" if use_pos_gate else "baseline"

    for epoch in range(EPOCHS):
        model.train()
        t0 = time.perf_counter()
        batch_times: list[float] = []
        for chunks, targets in train_dl:
            tb = time.perf_counter()
            chunks, targets = chunks.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            logits, _, _ = model(chunks)
            F.cross_entropy(logits, targets).backward()
            optimizer.step()
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            batch_times.append(time.perf_counter() - tb)
        epoch_time = time.perf_counter() - t0

        if epoch < 2:
            print(
                f"  [timing/{label}] epoch={epoch + 1}"
                f"  total={epoch_time:.2f}s"
                f"  batch: first={batch_times[0]:.3f}s"
                f"  mean={sum(batch_times) / len(batch_times):.3f}s"
                f"  max={max(batch_times):.3f}s"
            )

        if (epoch + 1) % 10 == 0:
            c1_acc, c1_r, c1_h = _eval(model, val_c1, has_target=True)
            c2_acc, c2_r, c2_h = _eval(model, val_c2, has_target=True)
            _, c3_r, c3_h = _eval(model, val_c3, has_target=False)

            print(
                f"[{label}] epoch={epoch + 1:3d}"
                f"  C1 acc={c1_acc:.3f} R={c1_r:.3f} H={c1_h:.2f}"
                f"  C2 acc={c2_acc:.3f} R={c2_r:.3f} H={c2_h:.2f}"
                f"  C3 R={c3_r:.3f} H={c3_h:.2f}"
            )

            hist["epoch"].append(epoch + 1)
            hist["c1_acc"].append(c1_acc)
            hist["c2_acc"].append(c2_acc)
            hist["c1_r"].append(c1_r)
            hist["c2_r"].append(c2_r)
            hist["c3_r"].append(c3_r)
            hist["c1_h"].append(c1_h)
            hist["c2_h"].append(c2_h)
            hist["c3_h"].append(c3_h)

            model.train()

    return hist


def main() -> None:
    results = {}
    for use_pos_gate in [False, True]:
        label = "pos_gate" if use_pos_gate else "baseline"
        print(f"\n--- {label} ---")
        results[label] = run(use_pos_gate)

    epochs = results["baseline"]["epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    def plot(ax: plt.Axes, keys: list[str], title: str, ylabel: str) -> None:  # type: ignore
        styles = {"baseline": ("steelblue", "-"), "pos_gate": ("darkorange", "--")}
        linestyles = ["-", "--", ":"]
        for label, (color, ls) in styles.items():
            for i, key in enumerate(keys):
                ax.plot(
                    epochs,
                    results[label][key],
                    color=color,
                    linestyle=linestyles[i],
                    marker="o",
                    markersize=4,
                    label=f"{label} {key}",
                )
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plot(axes[0, 0], ["c1_acc"], "C1 accuracy (unique key)", "accuracy")
    plot(axes[0, 1], ["c2_acc"], "C2 accuracy (ambiguous key -- recency)", "accuracy")
    plot(axes[1, 0], ["c1_r", "c2_r", "c3_r"], "mean R at recall position", "mean R")
    plot(axes[1, 1], ["c1_h", "c2_h", "c3_h"], "retrieval weight entropy", "entropy")

    # add max-entropy reference line (uniform over n_slots)
    import math

    axes[1, 1].axhline(
        math.log(N_SLOTS),
        color="gray",
        linestyle=":",
        alpha=0.5,
        label=f"uniform H={math.log(N_SLOTS):.2f}",
    )
    axes[1, 1].legend(fontsize=7)

    fig.suptitle(
        f"CAM retrieval experiment  |  n_slots={N_SLOTS}  d_sig={D_SIG}  ambiguous_k={AMBIGUOUS_K}"
    )
    fig.tight_layout()
    out = Path(__file__).parent / "retrieval_results.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
