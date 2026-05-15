import torch
import torch.nn as nn
import torch.nn.functional as F

from kernel.gated_causal_depthwise_conv import gated_causal_depthwise_conv


def _sync_sn(m: nn.Module) -> None:
    for hook in m._forward_pre_hooks.values():
        hook(m, None)


class CausalDepthWiseConv1d(nn.Module):
    r"""Causal depthwise convolution over sequences.

    Each position attends only to itself and the preceding :math:`size - 1` positions.
    Depthwise groups keep the cost at :math:`O(d \cdot size)` per layer rather than :math:`O(d^2 \cdot size)`.

    Args:
        d: Channel dimension.
        size: Kernel size; controls the local receptive field per pass.
    """

    def __init__(self, d: int, size: int = 7):
        super().__init__()
        self.size = size
        self.conv = nn.Conv1d(
            d, d, kernel_size=size, padding=size - 1, groups=d, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x: `(B, T, d)`

        Returns:
            `(B, T, d)` with causal masking applied.
        """
        x = x.transpose(1, 2)
        x = self.conv(x)[:, :, : -(self.size - 1)]
        return x.transpose(1, 2)


class GDLayer(nn.Module):
    r"""A single Gated Dissipative layer.

    Args:
        d: Hidden dimensiion.
        size: Kernel size for the causal depthwise convolution carrier.
    """

    def __init__(self, d: int, size: int = 7):
        super().__init__()
        self.conv_fwd = CausalDepthWiseConv1d(d, size)
        self.conv_bwd = CausalDepthWiseConv1d(d, size)

        for prefix in ["gf", "gb"]:
            W1 = nn.utils.spectral_norm(nn.Linear(d, d))
            W2 = nn.Linear(d, d)
            norm = nn.RMSNorm(d)
            nn.init.normal_(W2.weight, std=0.01)
            nn.init.constant_(W2.bias, 2.0)  # The gates need to start open ~0.95.
            setattr(self, f"{prefix}_W1", W1)
            setattr(self, f"{prefix}_W2", W2)
            setattr(self, f"{prefix}_norm", norm)

        for prefix in ["rf", "rb"]:
            W1 = nn.utils.spectral_norm(nn.Linear(d * 2, d))
            W2 = nn.Linear(d, d)
            norm = nn.RMSNorm(d)
            nn.init.normal_(W2.weight, std=0.01)
            nn.init.constant_(W2.bias, -2.0)  # The recovery starts closed ~0.12.
            setattr(self, f"{prefix}_W1", W1)
            setattr(self, f"{prefix}_W2", W2)
            setattr(self, f"{prefix}_norm", norm)

    def _gate(self, prefix: str, x: torch.Tensor) -> torch.Tensor:
        W1   = getattr(self, f"{prefix}_W1")
        W2   = getattr(self, f"{prefix}_W2")
        norm = getattr(self, f"{prefix}_norm")
        return F.sigmoid(W2(norm(F.silu(W1(x)))))

    def _recovery(
        self, prefix: str, side: torch.Tensor, fwd: torch.Tensor
    ) -> torch.Tensor:
        """Compute recovery gate values.

        Conditioned on both `side` and `fwd` so that a trigger in `fwd` can
        activate recall of content stored in `side` even when the two are
        geometrically orthogonal.

        Args:
            prefix: Parameter prefix, one of `gf` or `gb`.
            side: Side stream `(B, T, d)`.
            fwd: Forward stream `(B, T, d)`.

        Returns:
            Recovery gate values in `(0, 1)^d`, shape `(B, T, d)`.
        """
        W1 = getattr(self, f"{prefix}_W1")
        W2 = getattr(self, f"{prefix}_W2")
        norm = getattr(self, f"{prefix}_norm")
        return F.sigmoid(norm(W2(F.silu(W1(torch.cat([side, fwd], dim=-1))))))  # type: ignore[reportPrivateImportUsage]

    def fwd_step(
        self, fwd: torch.Tensor, side: torch.Tensor, return_gate: bool = False
    ) -> tuple[torch.Tensor, ...]:
        fwd_t: torch.Tensor = self.conv_fwd(fwd)
        R = self._recovery("rf", side, fwd_t)
        _sync_sn(self.gf_W1)  # type: ignore
        fwd_new, side_new = gated_causal_depthwise_conv(
            fwd, fwd_t, side, R,
            self.conv_fwd.conv.weight.squeeze(1).float(),
            self.gf_W1.weight.float(), self.gf_W1.bias.float(),  # type: ignore
            self.gf_norm.weight.float(),  # type: ignore
            self.gf_W2.weight.float(), self.gf_W2.bias.float(),  # type: ignore
        )
        if return_gate:
            return fwd_new, side_new, self._gate("gf", fwd_t)
        return fwd_new, side_new

    def bwd_step(
        self, fwd: torch.Tensor, side: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fwd_t: torch.Tensor = self.conv_bwd(fwd)
        R = self._recovery("rb", side, fwd_t)
        _sync_sn(self.gb_W1)  # type: ignore
        return gated_causal_depthwise_conv(
            fwd, fwd_t, side, R,
            self.conv_bwd.conv.weight.squeeze(1).float(),
            self.gb_W1.weight.float(), self.gb_W1.bias.float(),  # type: ignore
            self.gb_norm.weight.float(),  # type: ignore
            self.gb_W2.weight.float(), self.gb_W2.bias.float(),  # type: ignore
        )
