import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from gdnet.layer import GDLayer

B, T, d, k = 2, 128, 512, 7
N_STEPS = 500
LR = 1e-4


def make_rng():
    return torch.Generator(device="cuda").manual_seed(42)  # type: ignore


losses_full, losses_partial = [], []
times_full, times_partial = [], []

torch.manual_seed(0)
layer_full = GDLayer(d, k).cuda().float()
optim_full = torch.optim.AdamW(layer_full.parameters(), lr=LR)
rng = make_rng()

for _ in range(N_STEPS):
    fwd_in = torch.randn(B, T, d, device="cuda", generator=rng, requires_grad=True)
    side_in = torch.randn(B, T, d, device="cuda", generator=rng, requires_grad=True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    f2, s2 = layer_full.fwd_step(fwd_in, side_in)
    loss = f2.sum() + s2.sum()
    optim_full.zero_grad()
    loss.backward()
    optim_full.step()
    torch.cuda.synchronize()
    times_full.append(time.perf_counter() - t0)
    losses_full.append(loss.item())

del layer_full, optim_full
torch.cuda.empty_cache()

torch.manual_seed(0)
layer_partial = GDLayer(d, k).cuda().float()
for prefix in ["gb", "rb"]:
    nn.utils.remove_spectral_norm(getattr(layer_partial, f"{prefix}_W1"))
optim_partial = torch.optim.AdamW(layer_partial.parameters(), lr=LR)
rng = make_rng()

for _ in range(N_STEPS):
    fwd_in = torch.randn(B, T, d, device="cuda", generator=rng, requires_grad=True)
    side_in = torch.randn(B, T, d, device="cuda", generator=rng, requires_grad=True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    f2, s2 = layer_partial.fwd_step(fwd_in, side_in)
    loss = f2.sum() + s2.sum()
    optim_partial.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(layer_partial.parameters(), max_norm=1.0)
    optim_partial.step()
    torch.cuda.synchronize()
    times_partial.append(time.perf_counter() - t0)
    losses_partial.append(loss.item())

del layer_partial, optim_partial
torch.cuda.empty_cache()

avg_full = sum(times_full[10:]) / len(times_full[10:]) * 1000
avg_partial = sum(times_partial[10:]) / len(times_partial[10:]) * 1000
print(f"Full SN:    {avg_full:.3f} ms/iter")
print(f"Partial SN: {avg_partial:.3f} ms/iter")
print(f"Speedup:    {avg_full / avg_partial:.2f}x")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(losses_full, label="full SN", alpha=0.8)
ax1.plot(losses_partial, label="partial SN + clip", alpha=0.8)
ax1.set_xlabel("Step")
ax1.set_ylabel("Loss")
ax1.set_title("Loss: full SN vs partial SN")
ax1.legend()

ax2.plot(times_full[10:], label="full SN", alpha=0.8)
ax2.plot(times_partial[10:], label="partial SN + clip", alpha=0.8)
ax2.set_xlabel("Step")
ax2.set_ylabel("Time (s)")
ax2.set_title(f"Per-step time  |  full={avg_full:.1f}ms  partial={avg_partial:.1f}ms")
ax2.legend()

fig.tight_layout()
out = Path(__file__).parent / "compare_partial_sn.png"
fig.savefig(out, dpi=150)
print(f"Saved -> {out}")
