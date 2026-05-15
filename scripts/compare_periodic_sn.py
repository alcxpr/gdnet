import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from contextlib import nullcontext

import matplotlib.pyplot as plt
import torch

from gdnet.layer import GDLayer, freeze_sn_iteration

B, T, d, k = 2, 128, 512, 7
N_STEPS = 500
LR = 1e-4
SN_PREFIXES = ["gf", "gb", "rf", "rb"]


def make_rng():
    return torch.Generator(device="cuda").manual_seed(42)  # type: ignore


def read_sigma(layer: GDLayer, prefix: str) -> float:
    m = getattr(layer, f"{prefix}_W1")
    return torch.dot(m.weight_u, m.weight_orig @ m.weight_v).item()


def run(update_every: int, label: str):
    torch.manual_seed(0)
    layer = GDLayer(d, k).cuda().float()
    optim = torch.optim.AdamW(layer.parameters(), lr=LR)
    rng = make_rng()

    losses, times = [], []
    sigmas = {p: [] for p in SN_PREFIXES}

    for step in range(N_STEPS):
        fwd_in  = torch.randn(B, T, d, device="cuda", generator=rng, requires_grad=True)
        side_in = torch.randn(B, T, d, device="cuda", generator=rng, requires_grad=True)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        ctx = nullcontext() if step % update_every == 0 else freeze_sn_iteration(layer)
        with ctx:
            f2, s2 = layer.fwd_step(fwd_in, side_in)
            loss = f2.sum() + s2.sum()
            optim.zero_grad()
            loss.backward()
            optim.step()

        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        losses.append(loss.item())
        for p in SN_PREFIXES:
            sigmas[p].append(read_sigma(layer, p))

    avg_ms = sum(times[10:]) / len(times[10:]) * 1000
    sigma_finals = {p: sigmas[p][-1] for p in SN_PREFIXES}
    print(f"{label:20s}  {avg_ms:.3f} ms/iter  sigma_final={sigma_finals}")
    del layer, optim
    torch.cuda.empty_cache()
    return losses, times, sigmas


configs = [
    (1,  "baseline (K=1)"),
    (5,  "periodic K=5"),
    (10, "periodic K=10"),
    (50, "periodic K=50"),
]

results = {}
for K, label in configs:
    results[label] = run(K, label)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
ax_loss, ax_time = axes[0, 0], axes[0, 1]
sigma_axes = {p: axes[1, i] for i, p in enumerate(SN_PREFIXES[:3])}
sigma_axes[SN_PREFIXES[3]] = axes[0, 2]

for label, (losses, times, sigmas) in results.items():
    ax_loss.plot(losses, label=label, alpha=0.8)
    ax_time.plot(times[10:], label=label, alpha=0.6)
    for p, ax in sigma_axes.items():
        ax.plot(sigmas[p], label=label, alpha=0.8)

ax_loss.set_title("Loss")
ax_loss.set_xlabel("Step")
ax_loss.legend(fontsize=8)

ax_time.set_title("Per-step time (s)")
ax_time.set_xlabel("Step")
ax_time.legend(fontsize=8)

for p, ax in sigma_axes.items():
    ax.set_title(f"sigma({p}_W1)")
    ax.set_xlabel("Step")
    ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=8)

fig.tight_layout()
out = Path(__file__).parent / "compare_periodic_sn.png"
fig.savefig(out, dpi=150)
print(f"Saved -> {out}")
