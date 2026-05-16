"""Replication of the pos_gate retrieval experiment using gdnet.memory.Memory.

Tests C1/C2/C3 accuracy with the production Memory class, which writes chunks
sequentially through memory.write() (val detached) and reads via fused_mem_read.

The original experiment (run.py / cam.py) built buffer_tags and buffer_vals inline
in one batched forward pass with no detach. This script tests whether the val detach
in memory.write() affects C2 recency disambiguation.

Usage:
    python experiments/retrieval/run_memory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import N_VALUES, VOCAB_SIZE, RetrievalDataset
from model import ChunkEncoder
from torch.utils.data import ConcatDataset, DataLoader

from gdnet.memory import Memory

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
N_CYCLES = 2
AMBIGUOUS_K = 2


def _entropy(w: torch.Tensor) -> torch.Tensor:
    return -(w * (w + 1e-8).log()).sum(-1)


class SequentialMemoryModel(nn.Module):
    """Wraps Memory with a sequential chunk encoder for the retrieval task.

    Write chunks are processed one at a time through memory.write() (val detached).
    The query chunk is processed through memory.read() via fused_mem_read.
    """

    def __init__(
        self, vocab_size: int, n_values: int, d: int, d_sig: int, d_c: int, n_slots: int
    ) -> None:
        super().__init__()
        self.n_slots = n_slots
        self.d_sig = d_sig
        self.d_c = d_c
        self.embed = nn.Embedding(vocab_size, d)
        self.encoder = ChunkEncoder(d, N_LAYERS, KERNEL_SIZE, N_CYCLES)
        self.cam = Memory(d, d_c, d_sig, n_slots, chunk_size=CHUNK_SIZE)
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, n_values)

    def forward(self, chunks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, n_chunks, T = chunks.shape
        n_write = n_chunks - 1

        buffer_tags = torch.zeros(B, self.n_slots, self.d_sig, device=chunks.device)  # type: ignore
        buffer_vals = torch.zeros(B, self.n_slots, self.d_c, device=chunks.device)  # type: ignore

        for i in range(n_write):
            x = self.embed(chunks[:, i, :])
            fwd, side = self.encoder(x)
            buffer_tags, buffer_vals = self.cam.write(
                fwd[:, -1, :],
                side[0].mean(dim=1),
                buffer_tags,
                buffer_vals,  # type: ignore
            )

        x = self.embed(chunks[:, -1, :])
        fwd, side = self.encoder(x)
        fwd_new, w = self.cam.read(fwd, side[0], buffer_tags, buffer_vals)

        logits = self.head(self.norm(fwd_new[:, -1, :]))
        return logits, w


def _eval(
    model: SequentialMemoryModel, ds: RetrievalDataset, has_target: bool = True
) -> tuple[float, float]:
    dl = DataLoader(ds, batch_size=256)
    accs, ents = [], []
    model.eval()
    with torch.no_grad():
        for chunks, targets in dl:
            chunks = chunks.to(DEVICE)
            logits, w = model(chunks)
            if has_target:
                preds = logits.argmax(-1).cpu()
                accs.append((preds == targets).float().mean().item())
            ents.append(_entropy(w).cpu().mean().item())
    return (
        float(np.mean(accs)) if accs else float("nan"),
        float(np.mean(ents)),
    )


def run() -> None:
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

    model = SequentialMemoryModel(
        vocab_size=VOCAB_SIZE,
        n_values=N_VALUES,
        d=D,
        d_sig=D_SIG,
        d_c=D_C,
        n_slots=N_SLOTS,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        t0 = time.perf_counter()
        for chunks, targets in train_dl:
            chunks, targets = chunks.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(chunks)
            F.cross_entropy(logits, targets).backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            c1_acc, c1_h = _eval(model, val_c1, has_target=True)
            c2_acc, c2_h = _eval(model, val_c2, has_target=True)
            _, c3_h = _eval(model, val_c3, has_target=False)
            epoch_time = time.perf_counter() - t0
            print(
                f"epoch={epoch + 1:3d} ({epoch_time:.1f}s)"
                f"  C1 acc={c1_acc:.3f} H={c1_h:.2f}"
                f"  C2 acc={c2_acc:.3f} H={c2_h:.2f}"
                f"  C3 H={c3_h:.2f}"
            )
            model.train()


if __name__ == "__main__":
    run()
