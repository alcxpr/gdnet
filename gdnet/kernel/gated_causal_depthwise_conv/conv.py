from __future__ import annotations

import torch
import triton
import triton.language as tl

MAX_K = 16


@triton.jit
def _causal_dwconv_fwd_kernel(
    X_ptr,
    HALO_ptr,
    W_ptr,
    OUT_ptr,
    T,
    k,
    stride_xb,
    stride_xt,
    stride_xd,
    stride_hb,
    stride_ht,
    stride_hd,
    stride_ob,
    stride_ot,
    stride_od,
    BLOCK_T: tl.constexpr,
    MAX_K: tl.constexpr,
    USE_HALO: tl.constexpr,
):
    b = tl.program_id(0)
    ch = tl.program_id(1)

    ki_range = tl.arange(0, MAX_K)
    w = tl.load(W_ptr + ch * k + ki_range, mask=ki_range < k, other=0.0)

    for t_base in range(0, T, BLOCK_T):
        t = t_base + tl.arange(0, BLOCK_T)
        t_mask = t < T
        t_src = t[None, :] - (k - 1) + ki_range[:, None]  # (MAX_K, BLOCK_T)

        if USE_HALO:
            halo_src = t_src + (k - 1)
            is_halo = (t_src < 0) & t_mask[None, :] & (ki_range < k)[:, None]
            is_local = (t_src >= 0) & t_mask[None, :] & (ki_range < k)[:, None]
            halo_vals = tl.load(
                HALO_ptr + b * stride_hb + halo_src * stride_ht + ch * stride_hd,
                mask=is_halo,
                other=0.0,
            ).to(tl.float32)
            x_vals = tl.load(
                X_ptr + b * stride_xb + t_src * stride_xt + ch * stride_xd,
                mask=is_local,
                other=0.0,
            ).to(tl.float32)
            vals = halo_vals + x_vals  # masks are disjoint, sum == select
        else:
            valid = t_mask[None, :] & (t_src >= 0) & (ki_range < k)[:, None]
            vals = tl.load(
                X_ptr + b * stride_xb + t_src * stride_xt + ch * stride_xd,
                mask=valid,
                other=0.0,
            ).to(tl.float32)

        tl.store(
            OUT_ptr + b * stride_ob + t * stride_ot + ch * stride_od,
            tl.sum(vals * w[:, None], axis=0),
            mask=t_mask,
        )


@triton.jit
def _causal_dwconv_bwd_kernel(
    dOUT_ptr,
    X_ptr,
    HALO_ptr,
    W_ptr,
    dX_ptr,
    dHALO_ptr,
    dW_ptr,
    T,
    k,
    stride_b,
    stride_t,
    stride_d,
    stride_hb,
    stride_ht,
    stride_hd,
    BLOCK_T: tl.constexpr,
    MAX_K: tl.constexpr,
    USE_HALO: tl.constexpr,
):
    b = tl.program_id(0)
    ch = tl.program_id(1)

    ki_range = tl.arange(0, MAX_K)
    w = tl.load(W_ptr + ch * k + ki_range, mask=ki_range < k, other=0.0)

    dw_accs = tl.zeros((MAX_K,), dtype=tl.float32)
    for t_base in range(0, T, BLOCK_T):
        t = t_base + tl.arange(0, BLOCK_T)
        t_mask = t < T

        # dX: 2D load covers [t_base, t_base+BLOCK_T+k-2] of dOUT.
        # The ki=k-1 row is exactly dOUT[t], so the dW load below is an L1 hit.
        t_prime = t[None, :] + (k - 1) - ki_range[:, None]
        valid_tp = (
            t_mask[None, :] & (t_prime >= 0) & (t_prime < T) & (ki_range < k)[:, None]
        )
        d_outs = tl.load(
            dOUT_ptr + b * stride_b + t_prime * stride_t + ch * stride_d,
            mask=valid_tp,
            other=0.0,
        )
        tl.store(
            dX_ptr + b * stride_b + t * stride_t + ch * stride_d,
            tl.sum(d_outs * w[:, None], axis=0),
            mask=t_mask,
        )

        # dW: dOUT[t] is the ki=k-1 row already in L1 from the dX load above.
        d_out = tl.load(
            dOUT_ptr + b * stride_b + t * stride_t + ch * stride_d,
            mask=t_mask,
            other=0.0,
        ).to(tl.float32)
        t_src = t[None, :] + ki_range[:, None] - (k - 1)  # (MAX_K, BLOCK_T)

        if USE_HALO:
            halo_src = t_src + (k - 1)
            is_halo = (t_src < 0) & t_mask[None, :] & (ki_range < k)[:, None]
            is_local = (t_src >= 0) & t_mask[None, :] & (ki_range < k)[:, None]
            halo_x = tl.load(
                HALO_ptr + b * stride_hb + halo_src * stride_ht + ch * stride_hd,
                mask=is_halo,
                other=0.0,
            ).to(tl.float32)
            local_x = tl.load(
                X_ptr + b * stride_b + t_src * stride_t + ch * stride_d,
                mask=is_local,
                other=0.0,
            ).to(tl.float32)
            x_vals = halo_x + local_x
        else:
            valid = t_mask[None, :] & (t_src >= 0) & (t_src < T) & (ki_range < k)[:, None]
            x_vals = tl.load(
                X_ptr + b * stride_b + t_src * stride_t + ch * stride_d,
                mask=valid,
                other=0.0,
            ).to(tl.float32)

        dw_accs = dw_accs + tl.sum(d_out[None, :] * x_vals, axis=1)

    # dHALO[t_h] = sum_{t'=0}^{t_h} d_out[t'] * w[t_h - t']
    # Only the first k-1 output positions feed into halo grads.
    if USE_HALO:
        t_prime_h = ki_range  # shape (MAX_K,), mask to [0, k-2]
        t_prime_h_mask = t_prime_h < tl.minimum(k - 1, T)
        d_out_h = tl.load(
            dOUT_ptr + b * stride_b + t_prime_h * stride_t + ch * stride_d,
            mask=t_prime_h_mask,
            other=0.0,
        ).to(tl.float32)  # (MAX_K,)

        t_h = ki_range  # halo positions [0, k-2]
        t_h_mask = t_h < (k - 1)
        ki_2d = t_h[:, None] - t_prime_h[None, :]  # (MAX_K, MAX_K)
        valid_2d = (
            t_h_mask[:, None] & t_prime_h_mask[None, :] & (ki_2d >= 0) & (ki_2d < k)
        )
        w_2d = tl.load(W_ptr + ch * k + ki_2d, mask=valid_2d, other=0.0)
        dh_accs = tl.sum(d_out_h[None, :] * w_2d, axis=1)  # (MAX_K,)
        tl.store(
            dHALO_ptr + b * stride_hb + t_h * stride_ht + ch * stride_hd,
            dh_accs,
            mask=t_h_mask,
        )

    tl.atomic_add(dW_ptr + ch * k + ki_range, dw_accs, mask=ki_range < k)


def causal_dwconv_fwd(
    x: torch.Tensor,
    W_conv: torch.Tensor,
    T: int,
    k: int,
    BLOCK_T: int,
) -> torch.Tensor:
    """x: (B, T, d); returns (B, T, d)."""
    B, _, d = x.shape
    out = torch.empty(B, T, d, dtype=x.dtype, device=x.device)  # type: ignore
    _causal_dwconv_fwd_kernel[(B, d)](
        x,
        x,  # dummy HALO_ptr, never read when USE_HALO=False
        W_conv,
        out,
        T,
        k,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        0,
        0,
        0,  # dummy halo strides
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_T=BLOCK_T,  # type: ignore
        MAX_K=MAX_K,  # type: ignore
        USE_HALO=False,  # type: ignore
    )
    return out


def causal_dwconv_fwd_sp(
    x: torch.Tensor,
    halo: torch.Tensor,
    W_conv: torch.Tensor,
    T: int,
    k: int,
    BLOCK_T: int,
) -> torch.Tensor:
    """x: (B, T, d); halo: (B, k-1, d) from left neighbour; returns (B, T, d)."""
    B, _, d = x.shape
    out = torch.empty(B, T, d, dtype=x.dtype, device=x.device)  # type: ignore
    _causal_dwconv_fwd_kernel[(B, d)](
        x,
        halo,
        W_conv,
        out,
        T,
        k,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        halo.stride(0),
        halo.stride(1),
        halo.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_T=BLOCK_T,  # type: ignore
        MAX_K=MAX_K,  # type: ignore
        USE_HALO=True,  # type: ignore
    )
    return out


def causal_dwconv_bwd(
    d_conv: torch.Tensor,
    x: torch.Tensor,
    W_conv: torch.Tensor,
    T: int,
    k: int,
    BLOCK_T: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """d_conv, x: (B, T, d); returns dX (B, T, d) float32, dW_conv."""
    B, _, d = x.shape
    dX = torch.zeros(B, T, d, dtype=torch.float32, device=x.device)  # type: ignore
    dW_conv = torch.zeros_like(W_conv)  # type: ignore
    _causal_dwconv_bwd_kernel[(B, d)](
        d_conv,
        x,
        x,  # dummy HALO_ptr
        W_conv,
        dX,
        dX,  # dummy dHALO_ptr, never written when USE_HALO=False
        dW_conv,
        T,
        k,
        d_conv.stride(0),
        d_conv.stride(1),
        d_conv.stride(2),
        0,
        0,
        0,  # dummy halo strides
        BLOCK_T=BLOCK_T,  # type: ignore
        MAX_K=MAX_K,  # type: ignore
        USE_HALO=False,  # type: ignore
    )
    return dX, dW_conv


def causal_dwconv_bwd_sp(
    d_conv: torch.Tensor,
    x: torch.Tensor,
    halo: torch.Tensor,
    W_conv: torch.Tensor,
    T: int,
    k: int,
    BLOCK_T: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """d_conv, x: (B, T, d); halo: (B, k-1, d); returns dX (B, T, d), dHalo (B, k-1, d), dW_conv."""
    B, _, d = x.shape
    dX = torch.zeros(B, T, d, dtype=torch.float32, device=x.device)  # type: ignore
    dHalo = torch.zeros(B, k - 1, d, dtype=torch.float32, device=x.device)  # type: ignore
    dW_conv = torch.zeros_like(W_conv)  # type: ignore
    _causal_dwconv_bwd_kernel[(B, d)](
        d_conv,
        x,
        halo,
        W_conv,
        dX,
        dHalo,
        dW_conv,
        T,
        k,
        d_conv.stride(0),
        d_conv.stride(1),
        d_conv.stride(2),
        halo.stride(0),
        halo.stride(1),
        halo.stride(2),
        BLOCK_T=BLOCK_T,  # type: ignore
        MAX_K=MAX_K,  # type: ignore
        USE_HALO=True,  # type: ignore
    )
    return dX, dHalo, dW_conv


class CausalDWConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, W_conv: torch.Tensor, T: int, k: int, BLOCK_T: int
    ) -> torch.Tensor:
        out = causal_dwconv_fwd(x, W_conv, T, k, BLOCK_T)
        ctx.save_for_backward(x, W_conv)
        ctx.T, ctx.k, ctx.BLOCK_T = T, k, BLOCK_T
        return out

    @staticmethod
    def backward(ctx, d_conv: torch.Tensor) -> tuple:  # type: ignore
        x, W_conv = ctx.saved_tensors
        dX, dW_conv = causal_dwconv_bwd(
            d_conv.contiguous(), x, W_conv, ctx.T, ctx.k, ctx.BLOCK_T
        )
        return dX, dW_conv, None, None, None


class CausalDWConvFunctionSP(torch.autograd.Function):
    """SP variant: takes halo from left neighbour rank, propagates grad back via backward."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        halo: torch.Tensor,
        W_conv: torch.Tensor,
        T: int,
        k: int,
        BLOCK_T: int,
    ) -> torch.Tensor:
        out = causal_dwconv_fwd_sp(x, halo, W_conv, T, k, BLOCK_T)
        ctx.save_for_backward(x, halo, W_conv)
        ctx.T, ctx.k, ctx.BLOCK_T = T, k, BLOCK_T
        return out

    @staticmethod
    def backward(ctx, d_conv: torch.Tensor) -> tuple:  # type: ignore
        x, halo, W_conv = ctx.saved_tensors
        dX, dHalo, dW_conv = causal_dwconv_bwd_sp(
            d_conv.contiguous(), x, halo, W_conv, ctx.T, ctx.k, ctx.BLOCK_T
        )
        return dX, dHalo, dW_conv, None, None, None
