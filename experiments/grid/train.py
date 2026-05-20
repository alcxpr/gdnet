"""
Synthetic 4x4 grid experiment: does contrastive MoE routing specialize?

Setup:
  - 16 positions on a 4x4 grid, each embedded as a learned d-dim vector
  - 4 transition types: up / down / left / right (with wrap)
  - Each training sample is (pos, transition_type) -> (next_pos)
  - z_t = embed(pos), z_t1 = embed(next_pos).detach()

We train two TransitionOperators variants side-by-side:
  - MSE: original objective (F.mse_loss)
  - CTR: contrastive + load-balance (InfoNCE + entropy penalty)

After training we measure:
  1. Loss curves
  2. Routing specialization: for each transition type, which operator wins most?
     Ideal: each transition type -> a distinct dominant operator
  3. Routing entropy per transition type (lower = more specialized)
  4. Balance: fraction of samples routed to each operator

Run:
  uv run python experiments/grid/train.py
  uv run python experiments/grid/train.py --no-plot   (for headless)
"""

import argparse
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GRID = 4
N_POS = GRID * GRID
D = 64
N_OPS = 8
STEPS = 2000
BATCH = 64
LR = 3e-3
BALANCE_COEF = 0.01
SEED = 0


def _pos(row: int, col: int) -> int:
    return row * GRID + col


TRANSITIONS = {
    "up":    [(_pos(r, c), _pos((r - 1) % GRID, c)) for r in range(GRID) for c in range(GRID)],
    "down":  [(_pos(r, c), _pos((r + 1) % GRID, c)) for r in range(GRID) for c in range(GRID)],
    "left":  [(_pos(r, c), _pos(r, (c - 1) % GRID)) for r in range(GRID) for c in range(GRID)],
    "right": [(_pos(r, c), _pos(r, (c + 1) % GRID)) for r in range(GRID) for c in range(GRID)],
}
TRANS_NAMES = list(TRANSITIONS.keys())
N_TRANS = len(TRANS_NAMES)

ALL_PAIRS: list[tuple[int, int, int]] = []
for tid, name in enumerate(TRANS_NAMES):
    for src, dst in TRANSITIONS[name]:
        ALL_PAIRS.append((src, dst, tid))

ALL_PAIRS_T = torch.tensor(ALL_PAIRS, dtype=torch.long)


class MSEOps(nn.Module):
    def __init__(self, d: int, n_ops: int):
        super().__init__()
        self.W = nn.Parameter(
            torch.stack([torch.eye(d) + torch.randn(d, d) * 0.01 for _ in range(n_ops)])
        )
        self.router = nn.Linear(d, n_ops, bias=False)
        self.log_tau = nn.Parameter(torch.tensor(0.0))
        self.n_ops = n_ops

    def loss(self, z_t: torch.Tensor, z_t1: torch.Tensor) -> torch.Tensor:
        tau = self.log_tau.exp().clamp(min=0.1)
        w = F.softmax(self.router(z_t) / tau, dim=-1)
        preds = torch.einsum("oij,bj->boi", self.W, z_t)
        z_pred = (w.unsqueeze(-1) * preds).sum(dim=1)
        return F.mse_loss(z_pred, z_t1)

    @torch.no_grad()
    def routing_weights(self, z_t: torch.Tensor) -> torch.Tensor:
        tau = self.log_tau.exp().clamp(min=0.1)
        return F.softmax(self.router(z_t) / tau, dim=-1)


class CTROps(nn.Module):
    def __init__(self, d: int, n_ops: int, balance_coef: float = BALANCE_COEF):
        super().__init__()
        self.W = nn.Parameter(
            torch.stack([torch.eye(d) + torch.randn(d, d) * 0.01 for _ in range(n_ops)])
        )
        self.router = nn.Linear(d, n_ops, bias=False)
        self.log_tau = nn.Parameter(torch.tensor(0.0))
        self.log_temp = nn.Parameter(torch.tensor(-2.66))
        self.n_ops = n_ops
        self.balance_coef = balance_coef

    def loss(self, z_t: torch.Tensor, z_t1: torch.Tensor) -> torch.Tensor:
        tau = self.log_tau.exp().clamp(min=0.1)
        w = F.softmax(self.router(z_t) / tau, dim=-1)
        preds = torch.einsum("oij,bj->boi", self.W, z_t)
        z_pred = (w.unsqueeze(-1) * preds).sum(dim=1)

        temp = self.log_temp.exp().clamp(min=0.01)
        z_pred_n = F.normalize(z_pred, dim=-1)
        z_t1_n = F.normalize(z_t1, dim=-1)
        logits = z_pred_n @ z_t1_n.t() / temp
        labels = torch.arange(z_t.shape[0], device=z_t.device)
        recon = F.cross_entropy(logits, labels)

        mean_w = w.mean(dim=0)
        balance = (mean_w * mean_w.log()).sum()
        return recon + self.balance_coef * balance

    @torch.no_grad()
    def routing_weights(self, z_t: torch.Tensor) -> torch.Tensor:
        tau = self.log_tau.exp().clamp(min=0.1)
        return F.softmax(self.router(z_t) / tau, dim=-1)


@dataclass
class RunResult:
    name: str
    losses: list[float] = field(default_factory=list)
    routing_matrix: torch.Tensor = field(default_factory=lambda: torch.zeros(1))
    balance: torch.Tensor = field(default_factory=lambda: torch.zeros(1))


def sample_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = torch.randint(len(ALL_PAIRS_T), (batch_size,))
    rows = ALL_PAIRS_T[idx]
    return rows[:, 0], rows[:, 1], rows[:, 2]


def train(model: nn.Module, embed: nn.Embedding, steps: int, batch: int, lr: float, name: str) -> RunResult:
    result = RunResult(name=name)
    opt = torch.optim.Adam(list(model.parameters()) + list(embed.parameters()), lr=lr)

    for step in range(steps):
        src_ids, dst_ids, _ = sample_batch(batch)
        src_ids, dst_ids = src_ids.to(DEVICE), dst_ids.to(DEVICE)

        z_t = embed(src_ids)
        z_t1 = embed(dst_ids).detach()

        loss = model.loss(z_t, z_t1)
        opt.zero_grad()
        loss.backward()
        opt.step()

        result.losses.append(loss.item())

    result.routing_matrix, result.balance = eval_routing(model, embed)
    return result


@torch.no_grad()
def eval_routing(model: nn.Module, embed: nn.Embedding) -> tuple[torch.Tensor, torch.Tensor]:
    routing = torch.zeros(N_TRANS, model.n_ops)
    for tid, name in enumerate(TRANS_NAMES):
        pairs = TRANSITIONS[name]
        src_ids = torch.tensor([p for p, _ in pairs], device=DEVICE)
        z_t = embed(src_ids)
        w = model.routing_weights(z_t).cpu()
        routing[tid] = w.mean(dim=0)

    all_src = torch.arange(N_POS, device=DEVICE)
    z_all = embed(all_src)
    balance = model.routing_weights(z_all).cpu().mean(dim=0)
    return routing, balance


def print_results(result: RunResult) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {result.name}")
    print(f"{'=' * 60}")

    window = 200
    losses = result.losses
    early = sum(losses[:window]) / window
    late = sum(losses[-window:]) / window
    print(f"  loss  early={early:.4f}  late={late:.4f}")

    print(f"\n  routing matrix (transition x operator), mean weight:")
    print(f"  {'':8s}", end="")
    for o in range(result.routing_matrix.shape[1]):
        print(f"  op{o}", end="")
    print()
    for tid, name in enumerate(TRANS_NAMES):
        row = result.routing_matrix[tid]
        dominant = row.argmax().item()
        print(f"  {name:8s}", end="")
        for o in range(len(row)):
            marker = "*" if o == dominant else " "
            print(f" {row[o].item():.2f}{marker}", end="")
        w = row
        ent = -(w * w.clamp(min=1e-9).log()).sum().item()
        print(f"  entropy={ent:.3f}")

    dominant_ops = result.routing_matrix.argmax(dim=1).tolist()
    n_unique = len(set(dominant_ops))
    print(f"\n  dominant ops: {[TRANS_NAMES[i]+'->'+'op'+str(dominant_ops[i]) for i in range(N_TRANS)]}")
    print(f"  unique dominant ops: {n_unique}/{N_TRANS}  ({'specialized' if n_unique == N_TRANS else 'collapsed'})")

    print(f"\n  operator load balance (fraction of all positions):")
    for o, frac in enumerate(result.balance.tolist()):
        bar = "#" * int(frac * 40)
        print(f"  op{o}  {frac:.3f}  {bar}")


def plot_results(results: list[RunResult]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    window = 50
    for r in results:
        smoothed = [sum(r.losses[max(0, i - window):i + 1]) / min(i + 1, window) for i in range(len(r.losses))]
        ax.plot(smoothed, label=r.name)
    ax.set_title("Loss (smoothed)")
    ax.set_xlabel("step")
    ax.legend()

    for ax, r in zip(axes[1:], results):
        mat = r.routing_matrix.numpy()
        im = ax.imshow(mat, aspect="auto", vmin=0, vmax=mat.max())
        ax.set_title(f"{r.name}: routing matrix")
        ax.set_yticks(range(N_TRANS))
        ax.set_yticklabels(TRANS_NAMES)
        ax.set_xlabel("operator")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    out = "experiments/grid/results.png"
    plt.savefig(out, dpi=120)
    print(f"\nPlot saved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--d", type=int, default=D)
    parser.add_argument("--n-ops", type=int, default=N_OPS)
    args = parser.parse_args()

    torch.manual_seed(SEED)

    embed_mse = nn.Embedding(N_POS, args.d).to(DEVICE)
    embed_ctr = nn.Embedding(N_POS, args.d).to(DEVICE)
    nn.init.normal_(embed_mse.weight, std=0.1)
    embed_ctr.weight.data.copy_(embed_mse.weight.data)

    mse_ops = MSEOps(args.d, args.n_ops).to(DEVICE)
    ctr_ops = CTROps(args.d, args.n_ops).to(DEVICE)

    print(f"Training on {DEVICE}, d={args.d}, n_ops={args.n_ops}, steps={args.steps}, batch={args.batch}")
    print(f"Grid: {GRID}x{GRID}, {N_TRANS} transition types, {N_POS} positions")

    print("\n[1/2] MSE objective...")
    r_mse = train(mse_ops, embed_mse, args.steps, args.batch, LR, "MSE")

    print("[2/2] Contrastive + balance objective...")
    r_ctr = train(ctr_ops, embed_ctr, args.steps, args.batch, LR, "CTR")

    print_results(r_mse)
    print_results(r_ctr)

    if not args.no_plot:
        plot_results([r_mse, r_ctr])


if __name__ == "__main__":
    main()
