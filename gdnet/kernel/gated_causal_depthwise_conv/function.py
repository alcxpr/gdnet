from __future__ import annotations

import torch
import torch.nn.functional as F
import triton

from .conv import MAX_K, CausalDWConvFunction
from .gate_norm import gate_stream_update_fwd, gate_w2_bwd, rmsnorm_bwd, rmsnorm_fwd


class GatedCausalDepthwiseConvFunction(torch.autograd.Function):
    """Gating + RMSNorm path over a pre-computed conv output.

    Takes conv_flat (already computed by CausalDWConvFunction) so the conv is
    performed exactly once and its gradient flows back through that Function.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")  # type: ignore
    def forward(
        ctx,
        conv_flat: torch.Tensor,
        side_flat: torch.Tensor,
        R_flat: torch.Tensor,
        W1: torch.Tensor,
        b1: torch.Tensor,
        W_norm: torch.Tensor,
        W2: torch.Tensor,
        b2: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d = conv_flat.shape[1]
        BLOCK_D = d

        H = F.silu(F.linear(conv_flat, W1, b1))
        H_NORM, RSTD = rmsnorm_fwd(H, W_norm, eps, BLOCK_D)
        g_pre = F.linear(H_NORM, W2, b2)
        fwd_out, side_out = gate_stream_update_fwd(
            g_pre, conv_flat, side_flat, R_flat, BLOCK_D
        )

        ctx.save_for_backward(
            conv_flat, H, g_pre, side_flat, R_flat, W_norm, W2, W1, b1, RSTD
        )
        ctx.BLOCK_D = BLOCK_D
        return fwd_out, side_out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")  # type: ignore
    def backward(  # type: ignore
        ctx,
        d_fwd_out: torch.Tensor,
        d_side_out: torch.Tensor,
    ) -> tuple:
        conv_flat, H, g_pre, side_flat, R_flat, W_norm, W2, W1, b1, RSTD = (
            ctx.saved_tensors
        )
        BLOCK_D = ctx.BLOCK_D
        n_rows, d = conv_flat.shape

        d_fwd_f = d_fwd_out.reshape(n_rows, d)  # type: ignore
        d_side_f = d_side_out.reshape(n_rows, d)  # type: ignore

        d_h_norm, d_conv, d_side, d_R, dW2, db2, dW_norm = gate_w2_bwd(
            d_fwd_f,
            d_side_f,
            g_pre,
            conv_flat,
            side_flat,
            R_flat,
            H,
            RSTD,
            W_norm,
            W2,
            BLOCK_D,
        )
        dH = rmsnorm_bwd(d_h_norm, H, RSTD, W_norm, BLOCK_D)

        conv_req = conv_flat.detach().requires_grad_(True)
        W1_req = W1.detach().requires_grad_(True)
        b1_req = b1.detach().requires_grad_(True)
        with torch.enable_grad():
            F.silu(F.linear(conv_req, W1_req, b1_req)).backward(dH)
        d_conv_total = d_conv + conv_req.grad  # type: ignore
        dW1 = W1_req.grad
        db1 = b1_req.grad

        return d_conv_total, d_side, d_R, dW1, db1, dW_norm, dW2, db2, None


def gated_output(
    conv_flat: torch.Tensor,
    side: torch.Tensor,
    R: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W_norm: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply gating over a pre-computed conv output.

    Args:
        conv_flat: Conv output (n_rows, d), float32.
        side: Side stream (n_rows, d) or broadcastable; any dtype.
        R: Recovery gate (same shape as side); any dtype.
        W1: Gate MLP first-layer weight.
        b1: Gate MLP first-layer bias.
        W_norm: RMSNorm scale.
        W2: Gate MLP second-layer weight.
        b2: Gate MLP second-layer bias.
        eps: RMSNorm epsilon.

    Returns:
        (fwd_out, side_out), both (n_rows, d) float32.
    """
    n_rows, d = conv_flat.shape
    dtype = conv_flat.dtype
    side_flat = side.to(dtype).reshape(n_rows, d)
    R_flat = R.to(dtype).reshape(n_rows, d)
    return GatedCausalDepthwiseConvFunction.apply(  # type: ignore
        conv_flat,
        side_flat,
        R_flat,
        W1.to(dtype),
        b1.to(dtype),
        W_norm.to(dtype),
        W2.to(dtype),
        b2.to(dtype),
        eps,
    )


def gated_causal_depthwise_conv(
    x: torch.Tensor,
    side: torch.Tensor,
    R: torch.Tensor,
    W_conv: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W_norm: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal depthwise conv + gated stream update.

    Args:
        x: Input (B, T, d).
        side: Side stream (B, T, d).
        R: Recovery gate (B, T, d).
        W_conv: Depthwise conv weights (d, k).
        W1: Gate MLP first-layer weight.
        b1: Gate MLP first-layer bias.
        W_norm: RMSNorm scale.
        W2: Gate MLP second-layer weight.
        b2: Gate MLP second-layer bias.
        eps: RMSNorm epsilon.

    Returns:
        (fwd_out, side_out), both (B, T, d) in x.dtype.
    """
    B, T, d = x.shape
    k = W_conv.shape[1]
    assert k <= MAX_K, f"kernel size {k} exceeds MAX_K={MAX_K}"
    assert d == triton.next_power_of_2(d), f"d={d} must be a power of 2"

    BLOCK_T = min(triton.next_power_of_2(T), 64)
    x_dt = x.permute(0, 2, 1).contiguous()
    conv_out_dt = CausalDWConvFunction.apply(x_dt, W_conv, T, k, BLOCK_T)
    conv_flat = conv_out_dt.permute(0, 2, 1).contiguous().view(B * T, d)  # type: ignore

    fwd_out, side_out = gated_output(conv_flat, side, R, W1, b1, W_norm, W2, b2, eps)
    return fwd_out.view(B, T, d).to(x.dtype), side_out.view(B, T, d).to(x.dtype)
