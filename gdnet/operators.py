import torch
import torch.nn as nn
import torch.nn.functional as F


class TransitionOperators(nn.Module):
    r"""Learned transition operators over chunk representations.

    A bank of n_ops linear operators W_a in R^(d x d). A soft router assigns
    weights to operators based on z_t. The loss is InfoNCE: the routed
    prediction must identify the true z_t1 from all other z_t1s in the batch.
    A load-balancing entropy penalty keeps routing roughly uniform.

    Args:
        d: Representation dimension.
        n_ops: Number of transition operators.
        balance_coef: Weight on the load-balancing entropy penalty.
    """

    def __init__(self, d: int, n_ops: int = 8, balance_coef: float = 0.01):
        super().__init__()
        self.W = nn.Parameter(
            torch.stack([torch.eye(d) + torch.randn(d, d) * 0.01 for _ in range(n_ops)])  # type: ignore
        )
        self.router = nn.Linear(d, n_ops, bias=False)
        self.log_tau = nn.Parameter(torch.tensor(0.0))  # type: ignore
        self.log_temp = nn.Parameter(torch.tensor(-2.66))  # type: ignore
        self.n_ops = n_ops
        self.d = d
        self.balance_coef = balance_coef

    def loss(self, z_t: torch.Tensor, z_t1: torch.Tensor) -> torch.Tensor:
        tau = self.log_tau.exp().clamp(min=0.1)
        w = F.softmax(self.router(z_t) / tau, dim=-1)
        preds = torch.einsum("oij,bj->boi", self.W, z_t)  # type: ignore
        z_pred = (w.unsqueeze(-1) * preds).sum(dim=1)

        temp = self.log_temp.exp().clamp(min=0.01)
        logits = F.normalize(z_pred, dim=-1) @ F.normalize(z_t1, dim=-1).t() / temp
        labels = torch.arange(z_t.shape[0], device=z_t.device)  # type: ignore
        recon = F.cross_entropy(logits, labels)

        mean_w = w.mean(dim=0)
        balance = (mean_w * mean_w.log()).sum()
        return recon + self.balance_coef * balance

    @torch.no_grad()
    def winning_operator(self, z_t: torch.Tensor) -> torch.Tensor:
        tau = self.log_tau.exp().clamp(min=0.1)
        return F.softmax(self.router(z_t) / tau, dim=-1).argmax(dim=-1)
