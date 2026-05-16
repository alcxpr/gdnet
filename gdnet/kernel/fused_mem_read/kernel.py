from __future__ import annotations

import torch
import triton
import triton.language as tl

# Maximum tile size for d_sig and d_c. Programs loop over tiles when dimension
# exceeds this, keeping register footprint bounded regardless of actual size.
_MAX_BLOCK_DSIG = 128
_MAX_BLOCK_DC = 128


@triton.jit
def _fused_mem_read_fwd_kernel(
    Q_ptr,
    GAMMA_ptr,
    E_ptr,
    BTAGS_ptr,
    BVALS_ptr,
    alpha,
    W_ptr,
    OUT_ptr,
    B,
    n_slots,
    d_sig,
    d_c,
    BLOCK_DSIG: tl.constexpr,
    BLOCK_DC: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)
    s_rows = tl.arange(0, BLOCK_S)
    s_mask = s_rows < n_slots

    # NOTE: No d_mask on the d_sig or d_c loops. d_sig and d_c are asserted to be
    # multiples of BLOCK_DSIG/BLOCK_DC in the launcher (both must be powers of 2).
    # Masking forces predicated scalar ld.global.b32; without it the compiler emits
    # vectorized ld.global.v4.b32. Only s_mask is needed since n_slots may not be
    # a power of 2.

    sim_content = tl.zeros([BLOCK_S], dtype=tl.float32)
    sim_pos = tl.zeros([BLOCK_S], dtype=tl.float32)

    for d_start in range(0, d_sig, BLOCK_DSIG):
        d_cols = d_start + tl.arange(0, BLOCK_DSIG)

        q_tile = tl.load(Q_ptr + b * d_sig + d_cols).to(tl.float32)
        gamma_tile = tl.load(GAMMA_ptr + b * d_sig + d_cols).to(tl.float32)
        e_tile = tl.load(
            E_ptr + s_rows[:, None] * d_sig + d_cols[None, :],
            mask=s_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        btags_tile = tl.load(
            BTAGS_ptr + b * n_slots * d_sig + s_rows[:, None] * d_sig + d_cols[None, :],
            mask=s_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        sim_content += tl.sum(btags_tile * q_tile[None, :], axis=1)
        sim_pos += tl.sum(e_tile * gamma_tile[None, :], axis=1)

    sim = tl.where(s_mask, sim_content + alpha * sim_pos, float("-inf"))
    sim_max = tl.max(sim, axis=0)
    exp_sim = tl.where(s_mask, tl.exp(sim - sim_max), 0.0)
    w = exp_sim / tl.sum(exp_sim, axis=0)
    tl.store(W_ptr + b * n_slots + s_rows, w, mask=s_mask)

    for dc_start in range(0, d_c, BLOCK_DC):
        dc_cols = dc_start + tl.arange(0, BLOCK_DC)
        bvals_tile = tl.load(
            BVALS_ptr + b * n_slots * d_c + s_rows[:, None] * d_c + dc_cols[None, :],
            mask=s_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        tl.store(
            OUT_ptr + b * d_c + dc_cols,
            tl.sum(w[:, None] * bvals_tile, axis=0),
        )


@triton.jit
def _fused_mem_read_bwd_kernel(
    DOUT_ptr,
    Q_ptr,
    GAMMA_ptr,
    E_ptr,
    BTAGS_ptr,
    BVALS_ptr,
    W_ptr,
    alpha,
    DQ_ptr,
    DGAMMA_ptr,
    DALPHA_ptr,
    DSIM_ptr,
    DBTAGS_ptr,
    DBVALS_ptr,
    B,
    n_slots,
    d_sig,
    d_c,
    BLOCK_DSIG: tl.constexpr,
    BLOCK_DC: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)
    s_rows = tl.arange(0, BLOCK_S)
    s_mask = s_rows < n_slots

    w = tl.load(W_ptr + b * n_slots + s_rows, mask=s_mask, other=0.0).to(tl.float32)

    # Single d_c pass: accumulate d_w and store d_bvals.
    # d_bvals[s, dc] = w[s] * d_out[dc] -- depends only on w (not d_sim).
    d_w = tl.zeros([BLOCK_S], dtype=tl.float32)
    for dc_start in range(0, d_c, BLOCK_DC):
        dc_cols = dc_start + tl.arange(0, BLOCK_DC)
        d_out_tile = tl.load(DOUT_ptr + b * d_c + dc_cols).to(tl.float32)
        bvals_tile = tl.load(
            BVALS_ptr + b * n_slots * d_c + s_rows[:, None] * d_c + dc_cols[None, :],
            mask=s_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        d_w += tl.sum(bvals_tile * d_out_tile[None, :], axis=1)
        tl.store(
            DBVALS_ptr + b * n_slots * d_c + s_rows[:, None] * d_c + dc_cols[None, :],
            w[:, None] * d_out_tile[None, :],
            mask=s_mask[:, None],
        )

    wdw = tl.sum(w * d_w, axis=0)
    d_sim = tl.where(s_mask, w * (d_w - wdw), 0.0)
    tl.store(DSIM_ptr + b * n_slots + s_rows, d_sim, mask=s_mask)

    # d_sig pass: d_q, d_btags, d_gamma, d_alpha_b.
    d_alpha_b_acc = tl.zeros([BLOCK_S], dtype=tl.float32)
    for d_start in range(0, d_sig, BLOCK_DSIG):
        d_cols = d_start + tl.arange(0, BLOCK_DSIG)

        q_tile = tl.load(Q_ptr + b * d_sig + d_cols).to(tl.float32)
        gamma_tile = tl.load(GAMMA_ptr + b * d_sig + d_cols).to(tl.float32)
        e_tile = tl.load(
            E_ptr + s_rows[:, None] * d_sig + d_cols[None, :],
            mask=s_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        btags_tile = tl.load(
            BTAGS_ptr + b * n_slots * d_sig + s_rows[:, None] * d_sig + d_cols[None, :],
            mask=s_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        tl.store(
            DGAMMA_ptr + b * d_sig + d_cols,
            alpha * tl.sum(d_sim[:, None] * e_tile, axis=0),
        )
        tl.store(
            DBTAGS_ptr
            + b * n_slots * d_sig
            + s_rows[:, None] * d_sig
            + d_cols[None, :],
            d_sim[:, None] * q_tile[None, :],
            mask=s_mask[:, None],
        )
        tl.store(
            DQ_ptr + b * d_sig + d_cols,
            tl.sum(d_sim[:, None] * btags_tile, axis=0),
        )

        d_alpha_b_acc += d_sim * tl.sum(e_tile * gamma_tile[None, :], axis=1)

    tl.store(DALPHA_ptr + b, tl.sum(d_alpha_b_acc))


def _tile_params(n_slots: int, d_sig: int, d_c: int) -> tuple[int, int, int, int]:
    BLOCK_S = triton.next_power_of_2(n_slots)
    # Keep BLOCK_S * BLOCK_DSIG <= 2048 so the 2-D tile (held in registers) stays
    # under ~64 regs/thread at 128 threads/block.  ncu showed that BLOCK_S=32 ×
    # BLOCK_DSIG=128 hits 128 regs/thread, limiting occupancy to 4 blocks/SM (33%).
    # Halving the tile brings it to ~64 regs and doubles occupancy.
    _reg_tile_cap = 2048
    BLOCK_DSIG = min(
        triton.next_power_of_2(d_sig), _MAX_BLOCK_DSIG, max(1, _reg_tile_cap // BLOCK_S)
    )
    BLOCK_DC = min(
        triton.next_power_of_2(d_c), _MAX_BLOCK_DC, max(1, _reg_tile_cap // BLOCK_S)
    )
    total = BLOCK_S * (BLOCK_DSIG + BLOCK_DC)
    num_warps = 4 if total >= 4096 else 2 if total >= 1024 else 1
    return BLOCK_S, BLOCK_DSIG, BLOCK_DC, num_warps  # type: ignore


def fused_mem_read_fwd(
    q: torch.Tensor,
    gamma: torch.Tensor,
    e: torch.Tensor,
    buffer_tags: torch.Tensor,
    buffer_vals: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        q: Content query `(B, d_sig)` float32 contiguous, d_sig must be a power of 2.
        gamma: Position query `(B, d_sig)` float32 contiguous.
        e: Slot embeddings `(n_slots, d_sig)` float32 contiguous.
        buffer_tags: Stored content tags `(B, n_slots, d_sig)` float32 contiguous.
        buffer_vals: Stored compressed values `(B, n_slots, d_c)` float32 contiguous, d_c must be a power of 2.
        alpha: Scalar weight for the position term.

    Returns:
        retrieved_c `(B, d_c)` float32 and w `(B, n_slots)` float32.
    """
    B, d_sig = q.shape
    n_slots = e.shape[0]
    d_c = buffer_vals.shape[2]
    BLOCK_S, BLOCK_DSIG, BLOCK_DC, num_warps = _tile_params(n_slots, d_sig, d_c)
    assert d_sig % BLOCK_DSIG == 0, f"d_sig={d_sig} must be a power of 2"
    assert d_c % BLOCK_DC == 0, f"d_c={d_c} must be a power of 2"
    w = torch.empty(B, n_slots, dtype=torch.float32, device=q.device)  # type: ignore
    retrieved_c = torch.empty(B, d_c, dtype=torch.float32, device=q.device)  # type: ignore
    _fused_mem_read_fwd_kernel[(B,)](
        q,
        gamma,
        e,
        buffer_tags,
        buffer_vals,
        alpha,
        w,
        retrieved_c,
        B,
        n_slots,
        d_sig,
        d_c,
        BLOCK_DSIG=BLOCK_DSIG,
        BLOCK_DC=BLOCK_DC,
        BLOCK_S=BLOCK_S,
        num_warps=num_warps,  # type: ignore
    )
    return retrieved_c, w


def fused_mem_read_bwd(
    d_retrieved_c: torch.Tensor,
    q: torch.Tensor,
    gamma: torch.Tensor,
    e: torch.Tensor,
    buffer_tags: torch.Tensor,
    buffer_vals: torch.Tensor,
    w: torch.Tensor,
    alpha: float,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Args:
        d_retrieved_c: Upstream gradient `(B, d_c)` float32 contiguous.
        q: Content query `(B, d_sig)` float32 contiguous.
        gamma: Position query `(B, d_sig)` float32 contiguous.
        e: Slot embeddings `(n_slots, d_sig)` float32 contiguous.
        buffer_tags: Stored content tags `(B, n_slots, d_sig)` float32 contiguous.
        buffer_vals: Stored compressed values `(B, n_slots, d_c)` float32 contiguous.
        w: Retrieval weights from forward `(B, n_slots)` float32 contiguous.
        alpha: Scalar weight for the position term.

    Returns:
        d_q `(B, d_sig)`, d_gamma `(B, d_sig)`, d_alpha_per_b `(B,)` (sum in Python
        for scalar d_alpha), d_sim `(B, n_slots)` (caller computes
        d_e = alpha * d_sim.T @ gamma), d_buffer_tags `(B, n_slots, d_sig)`,
        d_buffer_vals `(B, n_slots, d_c)`.
    """
    B, d_sig = q.shape
    n_slots = w.shape[1]
    d_c = d_retrieved_c.shape[1]
    BLOCK_S, BLOCK_DSIG, BLOCK_DC, num_warps = _tile_params(n_slots, d_sig, d_c)
    d_q = torch.empty_like(q)  # type: ignore
    d_gamma = torch.empty_like(gamma)  # type: ignore
    d_alpha_per_b = torch.empty(B, dtype=torch.float32, device=q.device)  # type: ignore
    d_sim = torch.empty(B, n_slots, dtype=torch.float32, device=q.device)  # type: ignore
    d_btags = torch.empty_like(buffer_tags)  # type: ignore
    d_bvals = torch.empty_like(buffer_vals)  # type: ignore
    _fused_mem_read_bwd_kernel[(B,)](
        d_retrieved_c,
        q,
        gamma,
        e,
        buffer_tags,
        buffer_vals,
        w,
        alpha,
        d_q,
        d_gamma,
        d_alpha_per_b,
        d_sim,
        d_btags,
        d_bvals,
        B,
        n_slots,
        d_sig,
        d_c,
        BLOCK_DSIG=BLOCK_DSIG,
        BLOCK_DC=BLOCK_DC,
        BLOCK_S=BLOCK_S,
        num_warps=num_warps,  # type: ignore
    )
    return d_q, d_gamma, d_alpha_per_b, d_sim, d_btags, d_bvals
