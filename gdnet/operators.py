import torch
import torch.nn as nn
import torch.nn.functional as F


class TransitionOperators(nn.Module):
    r"""Learned transition operators over chunk representations.

    A bank of n_ops linear operators W_a in R^(d x d)
    that predict the next chunk representation from the current one. A soft router
    assigns fractional weights to all operators, preventing winner-take-all collapse.
    The router temperature tau is learned and starts at 1.0.

    Args:
        d: Representation dimension.
        n_ops: Number of transition operators.
    """

    def __init__(self, d: int, n_ops: int = 8):
        super().__init__()
        self.W = nn.Parameter(
            torch.stack([torch.eye(d) + torch.randn(d, d) * 0.01 for _ in range(n_ops)])  # type: ignore[reportPrivateImportUsage]
        )
        self.router = nn.Linear(d, n_ops, bias=False)
        self.log_tau = nn.Parameter(torch.tensor(0.0))  # type: ignore[reportPrivateImportUsage]
        self.n_ops = n_ops
        self.d = d

    def loss(self, z_t: torch.Tensor, z_t1: torch.Tensor) -> torch.Tensor:
        """Compute the capability loss between consecutive chunk representations.

        Args:
            z_t: Current chunk representation `(B, d)`.
            z_t1: Next chunk representation `(B, d)`, detached.

        Returns:
            Scalar MSE loss between the predicted and actual next representation.
        """
        tau = self.log_tau.exp().clamp(min=0.1)
        w = F.softmax(self.router(z_t) / tau, dim=-1)
        preds = torch.einsum("oij,bj->boi", self.W, z_t)  # type: ignore[reportPrivateImportUsage]
        z_pred = (w.unsqueeze(-1) * preds).sum(dim=1)
        return F.mse_loss(z_pred, z_t1)

    @torch.no_grad
    def winning_operator(self, z_t: torch.Tensor) -> torch.Tensor:  # type: ignore[reportPrivateImportUsage]
        """Return the index of the operator with the highest router weight.

        Args:
            z_t: Current chunk representation `(B, d)`.

        Returns:
            Operator indices `(B,)`.
        """
        tau = self.log_tau.exp().clamp(min=0.1)
        w = F.softmax(self.router(z_t) / tau, dim=-1)
        return w.argmax(dim=-1)
