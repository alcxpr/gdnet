from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd_kernel(
    H_ptr,
    W_norm_ptr,
    H_NORM_ptr,
    RSTD_ptr,
    n_rows,
    d,
    eps,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    # No mask: BLOCK_D == d (power-of-2) always, so all lanes are valid.
    # Mask-free loads let the compiler emit ld.global.v4 instead of predicated scalars.
    cols = tl.arange(0, BLOCK_D)
    base = row * d
    h = tl.load(H_ptr + base + cols)
    h_f = h.to(tl.float32)
    rstd = tl.math.rsqrt(tl.sum(h_f * h_f, axis=0) / d + eps)
    tl.store(RSTD_ptr + row, rstd)
    w_norm = tl.load(W_norm_ptr + cols)
    tl.store(H_NORM_ptr + base + cols, h * rstd * w_norm)


@triton.jit
def _gate_stream_update_fwd_kernel(
    G_PRE_ptr,
    CONV_ptr,
    SIDE_ptr,
    R_ptr,
    FWD_OUT_ptr,
    SIDE_OUT_ptr,
    n_rows,
    d,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    # No mask: BLOCK_D == d (power-of-2) always, so all lanes are valid.
    cols = tl.arange(0, BLOCK_D)
    base = row * d
    g = tl.sigmoid(tl.load(G_PRE_ptr + base + cols).to(tl.float32))
    fwd_t = tl.load(CONV_ptr + base + cols).to(tl.float32)
    side = tl.load(SIDE_ptr + base + cols).to(tl.float32)
    R = tl.load(R_ptr + base + cols).to(tl.float32)
    tl.store(FWD_OUT_ptr + base + cols, fwd_t * g + side * R)
    tl.store(SIDE_OUT_ptr + base + cols, fwd_t * (1.0 - g) + side * (1.0 - R))


@triton.jit
def _gate_bwd_elem_kernel(
    dFWD_ptr,
    dSIDE_ptr,
    G_PRE_ptr,
    CONV_ptr,
    SIDE_ptr,
    R_ptr,
    dG_PRE_ptr,
    dCONV_ptr,
    dSIDE_out_ptr,
    dR_ptr,
    n_rows,
    d,
    BLOCK_D: tl.constexpr,  # tile size, not full d
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    base = row * d

    # NOTE: No mask here intentionally. Adding a mask forces predicated scalar loads
    # (ld.global.b32) and prevents the compiler from emitting v4 vector loads
    # (ld.global.v4.b32). d is always a power-of-2 (asserted in function.py) and
    # BLOCK_D=128 divides every power-of-2 d >= 128, so all tiles are full - no
    # out-of-bounds access is possible. Do not add a mask.
    for tile_start in range(0, d, BLOCK_D):
        cols = tile_start + tl.arange(0, BLOCK_D)

        d_fwd = tl.load(dFWD_ptr + base + cols)
        d_side = tl.load(dSIDE_ptr + base + cols)
        g_pre = tl.load(G_PRE_ptr + base + cols).to(tl.float32)
        conv = tl.load(CONV_ptr + base + cols).to(tl.float32)
        side = tl.load(SIDE_ptr + base + cols).to(tl.float32)
        R = tl.load(R_ptr + base + cols).to(tl.float32)

        g = tl.sigmoid(g_pre)
        diff = d_fwd - d_side

        tl.store(dG_PRE_ptr + base + cols, diff * conv * g * (1.0 - g))
        tl.store(dCONV_ptr + base + cols, d_fwd * g + d_side * (1.0 - g))
        tl.store(dSIDE_out_ptr + base + cols, d_fwd * R + d_side * (1.0 - R))
        tl.store(dR_ptr + base + cols, diff * side)


@triton.jit
def _rmsnorm_bwd_kernel(
    d_h_norm_ptr,
    H_ptr,
    RSTD_ptr,
    W_norm_ptr,
    dH_ptr,
    n_rows,
    d,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    # No mask: BLOCK_D == d (power-of-2) always, so all lanes are valid.
    base = row * d
    cols = tl.arange(0, BLOCK_D)

    d_h_norm = tl.load(d_h_norm_ptr + base + cols)
    h = tl.load(H_ptr + base + cols).to(tl.float32)
    rstd = tl.load(RSTD_ptr + row)
    w_norm = tl.load(W_norm_ptr + cols)

    dot = tl.sum(d_h_norm * w_norm * h, axis=0)
    d_h = d_h_norm * rstd * w_norm - h * (rstd * rstd * rstd / d) * dot
    tl.store(dH_ptr + base + cols, d_h)


def rmsnorm_fwd(
    H: torch.Tensor,
    W_norm: torch.Tensor,
    eps: float,
    BLOCK_D: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_rows, d = H.shape
    H_NORM = torch.empty_like(H)  # type: ignore
    RSTD = torch.empty(n_rows, dtype=torch.float32, device=H.device)  # type: ignore
    _rmsnorm_fwd_kernel[(n_rows,)](
        H,
        W_norm,
        H_NORM,
        RSTD,
        n_rows,
        d,
        eps,
        BLOCK_D=BLOCK_D,  # type: ignore
    )
    return H_NORM, RSTD


def gate_stream_update_fwd(
    g_pre: torch.Tensor,
    conv_flat: torch.Tensor,
    side_flat: torch.Tensor,
    R_flat: torch.Tensor,
    BLOCK_D: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_rows, d = g_pre.shape
    fwd_out = torch.empty(n_rows, d, dtype=g_pre.dtype, device=g_pre.device)  # type: ignore
    side_out = torch.empty(n_rows, d, dtype=g_pre.dtype, device=g_pre.device)  # type: ignore
    _gate_stream_update_fwd_kernel[(n_rows,)](
        g_pre,
        conv_flat,
        side_flat,
        R_flat,
        fwd_out,
        side_out,
        n_rows,
        d,
        BLOCK_D=BLOCK_D,  # type: ignore
    )
    return fwd_out, side_out


# (BLOCK_D, num_warps) pairs. Each keeps sizePerThread = BLOCK_D/(num_warps*32) == 4
# so the compiler always emits ld.global.v4.b32. BLOCK_D must divide d - configs are
# tried largest-first and the first that evenly divides d wins. Tune this table for
# your hardware; do not add configs that violate the sizePerThread==4 invariant.
_BWD_ELEM_CONFIGS: list[tuple[int, int]] = [(512, 4), (256, 2), (128, 1), (64, 1), (32, 1)]


def _bwd_elem_tile(d: int) -> tuple[int, int]:
    for block_d, nw in _BWD_ELEM_CONFIGS:
        if d % block_d == 0:
            return block_d, nw
    return 32, 1


def gate_w2_bwd(
    d_fwd_f: torch.Tensor,
    d_side_f: torch.Tensor,
    g_pre: torch.Tensor,
    conv_flat: torch.Tensor,
    side_flat: torch.Tensor,
    R_flat: torch.Tensor,
    H: torch.Tensor,
    RSTD: torch.Tensor,
    W_norm: torch.Tensor,
    W2: torch.Tensor,
    BLOCK_D: int,
) -> tuple[torch.Tensor, ...]:
    n_rows, d = g_pre.shape
    dtype = g_pre.dtype
    block_d, num_warps = _bwd_elem_tile(d)
    d_g_pre = torch.empty(n_rows, d, dtype=torch.float32, device=g_pre.device)  # type: ignore
    d_conv = torch.empty(n_rows, d, dtype=dtype, device=g_pre.device)  # type: ignore
    d_side = torch.empty(n_rows, d, dtype=dtype, device=g_pre.device)  # type: ignore
    d_R = torch.empty(n_rows, d, dtype=dtype, device=g_pre.device)  # type: ignore
    _gate_bwd_elem_kernel[(n_rows,)](
        d_fwd_f,
        d_side_f,
        g_pre,
        conv_flat,
        side_flat,
        R_flat,
        d_g_pre,
        d_conv,
        d_side,
        d_R,
        n_rows,
        d,
        BLOCK_D=block_d,
        num_warps=num_warps,  # type: ignore
    )
    # Cast d_g_pre once to bf16 for both mm calls; avoids the larger fp32 intermediates
    # that .float() would create on W2/H/W_norm (all already bf16 from the forward).
    d_g_pre_b16 = d_g_pre.to(dtype)
    H_NORM = H * RSTD[:, None].to(dtype) * W_norm
    dW2 = torch.mm(d_g_pre_b16.t(), H_NORM)  # type: ignore
    db2 = d_g_pre.sum(0)
    d_h_norm = torch.mm(d_g_pre_b16, W2)  # type: ignore
    dW_norm = (d_h_norm * H * RSTD[:, None].to(dtype)).sum(0)
    return d_h_norm, d_conv, d_side, d_R, dW2, db2, dW_norm


def rmsnorm_bwd(
    d_h_norm: torch.Tensor,
    H: torch.Tensor,
    RSTD: torch.Tensor,
    W_norm: torch.Tensor,
    BLOCK_D: int,
) -> torch.Tensor:
    n_rows, d = H.shape
    dH = torch.empty_like(H)  # type: ignore
    _rmsnorm_bwd_kernel[(n_rows,)](
        d_h_norm,
        H,
        RSTD,
        W_norm,
        dH,
        n_rows,
        d,
        BLOCK_D=BLOCK_D,  # type: ignore
    )
    return dH
