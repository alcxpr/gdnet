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

    for t_base in range(0, T, BLOCK_T):
        t = t_base + tl.arange(0, BLOCK_T)
        t_mask = t < T
        t_prime = t[None, :] + (k - 1) - ki_range[:, None]
        valid = (
            t_mask[None, :] & (t_prime >= 0) & (t_prime < T) & (ki_range < k)[:, None]
        )
        d_outs = tl.load(
            dOUT_ptr + b * stride_b + t_prime * stride_t + ch * stride_d,
            mask=valid,
            other=0.0,
        )
        tl.store(
            dX_ptr + b * stride_b + t * stride_t + ch * stride_d,
            tl.sum(d_outs * w[:, None], axis=0),
            mask=t_mask,
        )

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

    dw_accs = tl.zeros((MAX_K,), dtype=tl.float32)
    for t_base in range(0, T, BLOCK_T):
        t = t_base + tl.arange(0, BLOCK_T)
        t_mask = t < T
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
            valid = t_mask[None, :] & (t_src >= 0) & (ki_range < k)[:, None]
            x_vals = tl.load(
                X_ptr + b * stride_b + t_src * stride_t + ch * stride_d,
                mask=valid,
                other=0.0,
            ).to(tl.float32)

        dw_accs = dw_accs + tl.sum(d_out[None, :] * x_vals, axis=1)

    tl.atomic_add(dW_ptr + ch * k + ki_range, dw_accs, mask=ki_range < k)


def causal_dwconv_fwd(
    x_dt: torch.Tensor,
    W_conv: torch.Tensor,
    T: int,
    k: int,
    BLOCK_T: int,
) -> torch.Tensor:
    B, d, _ = x_dt.shape
    out = torch.empty(B, d, T, dtype=torch.float32, device=x_dt.device)  # type: ignore
    _causal_dwconv_fwd_kernel[(B, d)](
        x_dt,
        x_dt,  # dummy, never read when USE_HALO=False
        W_conv,
        out,
        T,
        k,
        x_dt.stride(0),
        x_dt.stride(2),
        x_dt.stride(1),
        0, 0, 0,  # dummy halo strides
        out.stride(0),
        out.stride(2),
        out.stride(1),
        BLOCK_T=BLOCK_T,  # type: ignore
        MAX_K=MAX_K,  # type: ignore
        USE_HALO=False,  # type: ignore
    )
    return out


def causal_dwconv_fwd_sp(
    x_dt: torch.Tensor,
    halo_dt: torch.Tensor,
    W_conv: torch.Tensor,
    T: int,
    k: int,
    BLOCK_T: int,
) -> torch.Tensor:
    """SP variant: halo_dt is (B, d, k-1) from the left neighbour rank.

    Args:
        x_dt: Local input in channel-first layout (B, d, T_local).
        halo_dt: Left context received via SPHaloExchange (B, d, k-1).
        W_conv: Depthwise conv weights (d, k).
        T: Local sequence length.
        k: Kernel size.
        BLOCK_T: Triton tile size along T.

    Returns:
        Conv output (B, d, T).
    """
    B, d, _ = x_dt.shape
    out = torch.empty(B, d, T, dtype=torch.float32, device=x_dt.device)  # type: ignore
    _causal_dwconv_fwd_kernel[(B, d)](
        x_dt,
        halo_dt,
        W_conv,
        out,
        T,
        k,
        x_dt.stride(0),
        x_dt.stride(2),
        x_dt.stride(1),
        halo_dt.stride(0),
        halo_dt.stride(2),
        halo_dt.stride(1),
        out.stride(0),
        out.stride(2),
        out.stride(1),
        BLOCK_T=BLOCK_T,  # type: ignore
        MAX_K=MAX_K,  # type: ignore
        USE_HALO=True,  # type: ignore
    )
    return out


def causal_dwconv_bwd(
    d_conv_dt: torch.Tensor,
    x_dt: torch.Tensor,
    W_conv: torch.Tensor,
    T: int,
    k: int,
    BLOCK_T: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, d, _ = x_dt.shape
    dX_dt = torch.zeros(B, d, T, dtype=torch.float32, device=x_dt.device)  # type: ignore
    dW_conv = torch.zeros_like(W_conv)  # type: ignore
    _causal_dwconv_bwd_kernel[(B, d)](
        d_conv_dt,
        x_dt,
        x_dt,   # dummy HALO_ptr
        W_conv,
        dX_dt,
        dX_dt,  # dummy dHALO_ptr, never written when USE_HALO=False
        dW_conv,
        T,
        k,
        d_conv_dt.stride(0),
        d_conv_dt.stride(2),
        d_conv_dt.stride(1),
        0, 0, 0,  # dummy halo strides
        BLOCK_T=BLOCK_T,  # type: ignore
        MAX_K=MAX_K,  # type: ignore
        USE_HALO=False,  # type: ignore
    )
    return dX_dt, dW_conv


def causal_dwconv_bwd_sp(
    d_conv_dt: torch.Tensor,
    x_dt: torch.Tensor,
    halo_dt: torch.Tensor,
    W_conv: torch.Tensor,
    T: int,
    k: int,
    BLOCK_T: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SP variant: returns (dX_dt, dHalo_dt, dW_conv).

    Args:
        d_conv_dt: Upstream gradient (B, d, T).
        x_dt: Saved forward input (B, d, T).
        halo_dt: Saved forward halo (B, d, k-1).
        W_conv: Depthwise conv weights (d, k).
        T: Local sequence length.
        k: Kernel size.
        BLOCK_T: Triton tile size along T.

    Returns:
        dX_dt (B, d, T), dHalo_dt (B, d, k-1), dW_conv (d, k).
        dHalo_dt is passed back to SPHaloExchange.backward via autograd.
    """
    B, d, _ = x_dt.shape
    dX_dt = torch.zeros(B, d, T, dtype=torch.float32, device=x_dt.device)  # type: ignore
    dHalo_dt = torch.zeros(B, d, k - 1, dtype=torch.float32, device=x_dt.device)  # type: ignore
    dW_conv = torch.zeros_like(W_conv)  # type: ignore
    _causal_dwconv_bwd_kernel[(B, d)](
        d_conv_dt,
        x_dt,
        halo_dt,
        W_conv,
        dX_dt,
        dHalo_dt,
        dW_conv,
        T,
        k,
        d_conv_dt.stride(0),
        d_conv_dt.stride(2),
        d_conv_dt.stride(1),
        halo_dt.stride(0),
        halo_dt.stride(2),
        halo_dt.stride(1),
        BLOCK_T=BLOCK_T,  # type: ignore
        MAX_K=MAX_K,  # type: ignore
        USE_HALO=True,  # type: ignore
    )
    return dX_dt, dHalo_dt, dW_conv
