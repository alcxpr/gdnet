"""Retrieval v2 - entropy-conditioned recovery gate to fix C3."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from gdnet.layer import GDLayer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

N_KEYS = 16
N_VALUES = 16
N_NOISE_VOCAB = 16
KEY_OFFSET = 0
VALUE_OFFSET = N_KEYS
NOISE_OFFSET = N_KEYS + N_VALUES
SIGNAL_TOKEN = NOISE_OFFSET + N_NOISE_VOCAB
QUERY_TOKEN = SIGNAL_TOKEN + 1
VOCAB_SIZE = QUERY_TOKEN + 1


def _noise_chunk(size, rng):
    return rng.integers(NOISE_OFFSET, NOISE_OFFSET + N_NOISE_VOCAB, size=size)


def _signal_chunk(key, value, size, rng):
    chunk = _noise_chunk(size, rng)
    chunk[0] = SIGNAL_TOKEN
    chunk[1] = KEY_OFFSET + key
    chunk[2] = VALUE_OFFSET + value
    return chunk


def _query_chunk(key, size, rng):
    chunk = _noise_chunk(size, rng)
    chunk[0] = QUERY_TOKEN
    chunk[1] = KEY_OFFSET + key
    return chunk


class RetrievalDataset(Dataset):
    def __init__(
        self, n, n_slots, chunk_size=CHUNK_SIZE, condition=1, ambiguous_k=2, seed=0
    ):
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
                target = -1

            else:
                raise ValueError(f"unknown condition {condition}")

            all_chunks.append(np.stack(chunks))
            all_targets.append(target)

        self.chunks = torch.tensor(np.array(all_chunks), dtype=torch.long)
        self.targets = torch.tensor(all_targets, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.chunks[idx], self.targets[idx]


class CAM(nn.Module):
    def __init__(self, d, d_sig, d_c, n_slots):
        super().__init__()
        self.n_slots = n_slots
        self.d_sig = d_sig
        self.d_c = d_c
        self._max_h = math.log(n_slots)

        self.W_tag = nn.Linear(d, d_sig, bias=False)
        self.W_pos = nn.Linear(d_sig, d_sig, bias=False)
        nn.init.zeros_(self.W_pos.weight)
        self.W_slot = nn.Embedding(n_slots, d_sig)
        decay = 0.85
        with torch.no_grad():
            for s in range(n_slots):
                self.W_slot.weight[s] *= decay**s
        self.rho = nn.Parameter(torch.zeros(1))

        self.W_c = nn.Linear(d, d_c, bias=False)
        self.W_up = nn.Linear(d_c, d, bias=False)

        self.W_r1 = nn.utils.spectral_norm(nn.Linear(d * 3 + 1, d))
        self.W_r2 = nn.Linear(d, d)
        self.r_norm = nn.RMSNorm(d)
        nn.init.normal_(self.W_r2.weight, std=0.01)
        nn.init.constant_(self.W_r2.bias, -2.0)

    def read(self, fwd, side, buffer_tags, buffer_vals):
        B, T, _ = fwd.shape

        q = self.W_tag(fwd[:, -1, :])
        gamma = F.sigmoid(self.W_pos(q))
        alpha = torch.exp(self.rho)
        slot_ids = torch.arange(self.n_slots, device=fwd.device)
        e = self.W_slot(slot_ids)
        sim_content = torch.einsum("bd,bsd->bs", q, buffer_tags)
        sim_pos = torch.einsum("bd,sd->bs", gamma, e)
        w = F.softmax(sim_content + alpha * sim_pos, dim=-1)

        h = -(w * (w + 1e-8).log()).sum(-1) / self._max_h
        h_seq = h.unsqueeze(1).unsqueeze(2).expand(B, T, 1)

        retrieved_c = torch.einsum("bs,bsd->bd", w, buffer_vals)
        retrieved_e = self.W_up(retrieved_c).unsqueeze(1).expand_as(fwd)

        gate_input = torch.cat([side, fwd, retrieved_e, h_seq], dim=-1)
        R = F.sigmoid(self.r_norm(self.W_r2(F.silu(self.W_r1(gate_input)))))

        return fwd * (1 - R) + retrieved_e * R, R, w


class ChunkEncoder(nn.Module):
    def __init__(self, d, n_layers, kernel_size, n_cycles):
        super().__init__()
        self.layers = nn.ModuleList([GDLayer(d, kernel_size) for _ in range(n_layers)])
        self.n_cycles = n_cycles

    def forward(self, x):
        side = [torch.zeros_like(x) for _ in self.layers]
        for _ in range(self.n_cycles):
            for i, layer in enumerate(self.layers):
                x, side[i] = layer.fwd_step(x, side[i])  # type: ignore
            for i, layer in reversed(list(enumerate(self.layers))):
                x, side[i] = layer.bwd_step(x, side[i])  # type: ignore
        return x, side


class RetrievalModel(nn.Module):
    def __init__(self, vocab_size, n_values, d, d_sig, d_c, n_slots):
        super().__init__()
        self.n_slots = n_slots
        self.d_sig = d_sig
        self.d_c = d_c
        self.embed = nn.Embedding(vocab_size, d)
        self.encoder = ChunkEncoder(d, N_LAYERS, KERNEL_SIZE, N_CYCLES)
        self.cam = CAM(d, d_sig, d_c, n_slots)
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, n_values)

    def forward(self, chunks):
        B, n_chunks, T = chunks.shape
        n_write = n_chunks - 1

        x_write = self.embed(chunks[:, :n_write].reshape(B * n_write, T))
        fwd_write, side_write = self.encoder(x_write)

        tag_inp = fwd_write[:, -1, :].reshape(B, n_write, -1).flip(dims=[1])
        val_inp = side_write[0].mean(dim=1).reshape(B, n_write, -1).flip(dims=[1])
        buffer_tags = self.cam.W_tag(tag_inp)
        buffer_vals = self.cam.W_c(val_inp)

        x = self.embed(chunks[:, -1])
        fwd, side = self.encoder(x)
        fwd_new, R, w = self.cam.read(fwd, side[0], buffer_tags, buffer_vals)

        logits = self.head(self.norm(fwd_new[:, -1, :]))
        r_scalar = R[:, -1, :].mean(-1)
        return logits, r_scalar, w


def _entropy(w):
    return -(w * (w + 1e-8).log()).sum(-1)


def _eval(model, ds, has_target=True):
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


def run():
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
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    hist = {
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

    for epoch in range(EPOCHS):
        model.train()
        t0 = time.perf_counter()
        for chunks, targets in train_dl:
            chunks, targets = chunks.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            logits, _, _ = model(chunks)
            F.cross_entropy(logits, targets).backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            c1_acc, c1_r, c1_h = _eval(model, val_c1, has_target=True)
            c2_acc, c2_r, c2_h = _eval(model, val_c2, has_target=True)
            _, c3_r, c3_h = _eval(model, val_c3, has_target=False)
            epoch_time = time.perf_counter() - t0

            print(
                f"epoch={epoch + 1:3d} ({epoch_time:.1f}s)"
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


def plot(hist):
    epochs = hist["epoch"]
    max_h = math.log(N_SLOTS)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    ax.plot(epochs, hist["c1_acc"], marker="o", label="C1")
    ax.plot(epochs, hist["c2_acc"], marker="o", label="C2")
    ax.set_title("Accuracy")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, hist["c1_r"], marker="o", label="C1")
    ax.plot(epochs, hist["c2_r"], marker="o", label="C2")
    ax.plot(epochs, hist["c3_r"], marker="o", linestyle="--", label="C3")
    ax.set_title("Mean R at recall")
    ax.set_xlabel("epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(epochs, hist["c1_h"], marker="o", label="C1")
    ax.plot(epochs, hist["c2_h"], marker="o", label="C2")
    ax.plot(epochs, hist["c3_h"], marker="o", linestyle="--", label="C3")
    ax.axhline(
        max_h, color="gray", linestyle=":", alpha=0.5, label=f"uniform H={max_h:.2f}"
    )
    ax.set_title("Retrieval weight entropy")
    ax.set_xlabel("epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Retrieval v2 | n_slots={N_SLOTS} d_sig={D_SIG} ambiguous_k={AMBIGUOUS_K}"
    )
    fig.tight_layout()
    out = Path(__file__).parent / "results.png"
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    hist = run()
    plot(hist)
