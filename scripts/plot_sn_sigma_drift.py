import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import torch

from gdnet.layer import GDLayer

B, T, d, k = 4, 512, 512, 7
N_STEPS = 500

torch.manual_seed(0)
layer = GDLayer(d, k).cuda().float()
optim = torch.optim.AdamW(layer.parameters(), lr=1e-4)

PREFIXES = ["gf", "gb", "rf", "rb"]


def sigma(module) -> float:
    u = module.weight_u
    v = module.weight_v
    W = module.weight_orig
    return torch.dot(u, W @ v).item()  # type: ignore


history = {p: [] for p in PREFIXES}

for step in range(N_STEPS):
    fwd = torch.randn(B, T, d, device="cuda", requires_grad=True)
    side = torch.randn(B, T, d, device="cuda", requires_grad=True)

    fwd2, side2 = layer.fwd_step(fwd, side)
    loss = fwd2.sum() + side2.sum()
    optim.zero_grad()
    loss.backward()
    optim.step()

    for p in PREFIXES:
        history[p].append(sigma(getattr(layer, f"{p}_W1")))

fig, ax = plt.subplots(figsize=(10, 5))
for p in PREFIXES:
    ax.plot(history[p], label=p)
ax.set_xlabel("Step")
ax.set_ylabel("sigma(W1)")
ax.set_title("Spectral norm sigma drift over 500 steps")
ax.legend()
fig.tight_layout()
out = Path(__file__).parent / "sn_sigma_drift.png"
fig.savefig(out, dpi=150)
print(f"Saved -> {out}")
