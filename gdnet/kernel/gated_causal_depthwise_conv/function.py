from __future__ import annotations

import torch
import torch.nn.functional as F
import triton

from .conv import MAX_K, causal_dwconv_bwd, causal_dwconv_fwd
from .gate_norm import gate_stream_update_fwd, gate_w2_bwd, rmsnorm_bwd, rmsnorm_fwd


class GatedCausalDepthwiseConvFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # type: ignore
    def forward(
        ctx,
        x: torch.Tensor,
        fwd: torch.Tensor,
        side: torch.Tensor,
        R: torch.Tensor,
        W_conv: torch.Tensor,
        W1: torch.Tensor,
        b1: torch.Tensor,
        W_norm: torch.Tensor,
        W2: torch.Tensor,
        b2: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = x.dtype
        B, T, d = x.shape
        k = W_conv.shape[1]
        n_rows = B * T

        assert k <= MAX_K, f"kernel size {k} exceeds MAX_K={MAX_K}"
        assert d == triton.next_power_of_2(d), f"d={d} must be a power of 2"

        BLOCK_T = min(triton.next_power_of_2(T), 64)
        BLOCK_D = d

        x_dt = x.float().contiguous().permute(0, 2, 1).contiguous()
        side_flat = side.float().contiguous().view(n_rows, d)
        R_flat = R.float().contiguous().view(n_rows, d)
        W_conv, W1, b1, W_norm, W2, b2 = (
            W_conv.float(),
            W1.float(),
            b1.float(),
            W_norm.float(),
            W2.float(),
            b2.float(),
        )

        conv_out_dt = causal_dwconv_fwd(x_dt, W_conv, T, k, BLOCK_T)  # type: ignore
        conv_flat = conv_out_dt.permute(0, 2, 1).contiguous().view(n_rows, d)

        H = F.silu(F.linear(conv_flat, W1, b1))
        H_NORM, RSTD = rmsnorm_fwd(H, W_norm, eps, BLOCK_D)
        g_pre = F.linear(H_NORM, W2, b2)

        fwd_out, side_out = gate_stream_update_fwd(
            g_pre, conv_flat, side_flat, R_flat, BLOCK_D
        )

        ctx.save_for_backward(
            x_dt,
            conv_flat,
            H,
            g_pre,
            side_flat,
            R_flat,
            W_conv,
            W1,
            b1,
            W_norm,
            W2,
            RSTD,
        )
        ctx.B, ctx.T, ctx.d, ctx.k, ctx.eps = B, T, d, k, eps
        ctx.BLOCK_D, ctx.BLOCK_T = BLOCK_D, BLOCK_T
        ctx.dtype = dtype

        return fwd_out.view(B, T, d).to(dtype), side_out.view(B, T, d).to(dtype)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")  # type: ignore
    def backward(  # type: ignore
        ctx,
        d_fwd_out: torch.Tensor,
        d_side_out: torch.Tensor,
    ) -> tuple:
        (
            x_dt,
            conv_flat,
            H,
            g_pre,
            side_flat,
            R_flat,
            W_conv,
            W1,
            b1,
            W_norm,
            W2,
            RSTD,
        ) = ctx.saved_tensors
        B, T, d, k = ctx.B, ctx.T, ctx.d, ctx.k
        n_rows = B * T
        BLOCK_D, BLOCK_T = ctx.BLOCK_D, ctx.BLOCK_T

        d_fwd_f = d_fwd_out.contiguous().view(n_rows, d).to(torch.float32)  # type: ignore
        d_side_f = d_side_out.contiguous().view(n_rows, d).to(torch.float32)  # type: ignore

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
        d_conv_total = d_conv + conv_req.grad
        dW1 = W1_req.grad
        db1 = b1_req.grad

        d_conv_dt = d_conv_total.view(B, T, d).permute(0, 2, 1).contiguous()
        dX_dt, dW_conv = causal_dwconv_bwd(d_conv_dt, x_dt, W_conv, T, k, BLOCK_T)

        dtype = ctx.dtype
        return (
            dX_dt.permute(0, 2, 1).contiguous().to(dtype),
            None,
            d_side.view(B, T, d).to(dtype),
            d_R.view(B, T, d).to(dtype),
            dW_conv,
            dW1,
            db1,
            dW_norm,
            dW2,
            db2,
            None,
        )


def gated_causal_depthwise_conv(
    x: torch.Tensor,
    fwd: torch.Tensor,
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
    return GatedCausalDepthwiseConvFunction.apply(
        x, fwd, side, R, W_conv, W1, b1, W_norm, W2, b2, eps
    )
