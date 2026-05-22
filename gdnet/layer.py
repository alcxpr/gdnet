import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import triton

from .kernel.gated_causal_depthwise_conv import (
    CausalDWConvFunction,
    gated_output,
)
from .utils.sp import FusedHaloConvSP


class CausalDepthWiseConv1d(nn.Module):
    r"""Causal depthwise convolution over sequences.

    Each position attends only to itself and the preceding size - 1 positions.
    Depthwise groups keep the cost at O(d . size) per layer rather than O(d^2 . size).

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
            W1 = nn.Linear(d, d)
            nn.init.orthogonal_(W1.weight)
            W2 = nn.Linear(d, d)
            norm = nn.RMSNorm(d)
            nn.init.normal_(W2.weight, std=0.01)
            nn.init.constant_(W2.bias, 2.0)
            setattr(self, f"{prefix}_W1", W1)
            setattr(self, f"{prefix}_W2", W2)
            setattr(self, f"{prefix}_norm", norm)

        for prefix in ["rf", "rb"]:
            W1 = nn.Linear(d * 2, d)
            nn.init.orthogonal_(W1.weight)
            W2 = nn.Linear(d, d)
            norm = nn.RMSNorm(d)
            nn.init.normal_(W2.weight, std=0.01)
            nn.init.constant_(W2.bias, -2.0)
            setattr(self, f"{prefix}_W1", W1)
            setattr(self, f"{prefix}_W2", W2)
            setattr(self, f"{prefix}_norm", norm)

    def _gate(self, prefix: str, x: torch.Tensor) -> torch.Tensor:
        W1 = getattr(self, f"{prefix}_W1")
        W2 = getattr(self, f"{prefix}_W2")
        norm = getattr(self, f"{prefix}_norm")
        return F.sigmoid(W2(norm(F.silu(W1(x)))))

    def _recovery(
        self, prefix: str, side: torch.Tensor, fwd: torch.Tensor
    ) -> torch.Tensor:
        W1 = getattr(self, f"{prefix}_W1")
        W2 = getattr(self, f"{prefix}_W2")
        norm = getattr(self, f"{prefix}_norm")
        return F.sigmoid(norm(W2(F.silu(W1(torch.cat([side, fwd], dim=-1))))))  # type: ignore

    def _conv_flat(
        self,
        fwd: torch.Tensor,
        conv_module: CausalDepthWiseConv1d,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute causal conv via Triton, return (conv_3d, conv_flat).

        conv_3d is (B, T, d) float32; conv_flat is (B*T, d) float32.
        Both share the same storage - use conv_3d for R, conv_flat for gated_output.
        """
        B, T, d = fwd.shape
        k = conv_module.size
        BLOCK_T = min(triton.next_power_of_2(T), 64)
        W_conv = conv_module.conv.weight.squeeze(1)
        conv_3d = CausalDWConvFunction.apply(fwd, W_conv, T, k, BLOCK_T)
        return conv_3d, conv_3d.view(B * T, d)

    def _conv_flat_sp(
        self,
        fwd: torch.Tensor,
        conv_module: CausalDepthWiseConv1d,
        sp_group: dist.ProcessGroup,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, d = fwd.shape
        k = conv_module.size
        BLOCK_T = min(triton.next_power_of_2(T), 64)
        W_conv = conv_module.conv.weight.squeeze(1)
        conv_3d = FusedHaloConvSP.apply(fwd, W_conv, T, k, BLOCK_T, sp_group, id(conv_module))
        return conv_3d, conv_3d.view(B * T, d)

    def fwd_step(
        self, fwd: torch.Tensor, side: torch.Tensor, return_gate: bool = False
    ) -> tuple[torch.Tensor, ...]:
        B, T, d = fwd.shape
        conv_3d, conv_flat = self._conv_flat(fwd, self.conv_fwd)
        R = self._recovery("rf", side, conv_3d)
        fwd_out, side_out = gated_output(
            conv_flat,
            side,
            R,
            self.gf_W1.weight,  # type: ignore
            self.gf_W1.bias,  # type: ignore
            self.gf_norm.weight,  # type: ignore
            self.gf_W2.weight,  # type: ignore
            self.gf_W2.bias,  # type: ignore
        )
        if return_gate:
            return (
                fwd_out.view(B, T, d),
                side_out.view(B, T, d),
                self._gate("gf", conv_3d),
            )
        return fwd_out.view(B, T, d), side_out.view(B, T, d)

    def fwd_step_sp(
        self,
        fwd: torch.Tensor,
        side: torch.Tensor,
        sp_group: dist.ProcessGroup,
        return_gate: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        B, T, d = fwd.shape
        conv_3d, conv_flat = self._conv_flat_sp(fwd, self.conv_fwd, sp_group)
        R = self._recovery("rf", side, conv_3d)
        fwd_out, side_out = gated_output(
            conv_flat,
            side,
            R,
            self.gf_W1.weight,  # type: ignore
            self.gf_W1.bias,  # type: ignore
            self.gf_norm.weight,  # type: ignore
            self.gf_W2.weight,  # type: ignore
            self.gf_W2.bias,  # type: ignore
        )
        if return_gate:
            return (
                fwd_out.view(B, T, d),
                side_out.view(B, T, d),
                self._gate("gf", conv_3d),
            )
        return fwd_out.view(B, T, d), side_out.view(B, T, d)

    def bwd_step(
        self,
        fwd: torch.Tensor,
        side: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, d = fwd.shape
        conv_3d, conv_flat = self._conv_flat(fwd, self.conv_bwd)
        R = self._recovery("rb", side, conv_3d)
        fwd_out, side_out = gated_output(
            conv_flat,
            side,
            R,
            self.gb_W1.weight,  # type: ignore
            self.gb_W1.bias,  # type: ignore
            self.gb_norm.weight,  # type: ignore
            self.gb_W2.weight,  # type: ignore
            self.gb_W2.bias,  # type: ignore
        )
        return fwd_out.view(B, T, d), side_out.view(B, T, d)

    def bwd_step_sp(
        self,
        fwd: torch.Tensor,
        side: torch.Tensor,
        sp_group: dist.ProcessGroup,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, d = fwd.shape
        conv_3d, conv_flat = self._conv_flat_sp(fwd, self.conv_bwd, sp_group)
        R = self._recovery("rb", side, conv_3d)
        fwd_out, side_out = gated_output(
            conv_flat,
            side,
            R,
            self.gb_W1.weight,  # type: ignore
            self.gb_W1.bias,  # type: ignore
            self.gb_norm.weight,  # type: ignore
            self.gb_W2.weight,  # type: ignore
            self.gb_W2.bias,  # type: ignore
        )
        return fwd_out.view(B, T, d), side_out.view(B, T, d)
