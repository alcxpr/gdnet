from __future__ import annotations

import torch
import torch.distributed as dist

from ..kernel.gated_causal_depthwise_conv.conv import (
    causal_dwconv_bwd_sp,
    causal_dwconv_fwd_sp,
)


def _fwd_p2p_ops(
    edge: torch.Tensor,
    left_halo: torch.Tensor,
    rank: int,
    world_size: int,
    group: dist.ProcessGroup,
) -> list[dist.P2POp]:
    ops: list[dist.P2POp] = []
    if rank > 0:
        ops.append(dist.P2POp(dist.irecv, left_halo, rank - 1, group=group))
    if rank < world_size - 1:
        ops.append(dist.P2POp(dist.isend, edge, rank + 1, group=group))
    return ops


def _bwd_p2p_ops(
    grad_left_halo: torch.Tensor,
    grad_edge: torch.Tensor,
    rank: int,
    world_size: int,
    group: dist.ProcessGroup,
) -> list[dist.P2POp]:
    ops: list[dist.P2POp] = []
    if rank > 0:
        ops.append(dist.P2POp(dist.isend, grad_left_halo, rank - 1, group=group))
    if rank < world_size - 1:
        ops.append(dist.P2POp(dist.irecv, grad_edge, rank + 1, group=group))
    return ops


def begin_halo_recv(
    x_dt: torch.Tensor,
    k: int,
    sp_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor, list]:
    """Issue async P2P recv for the left halo without blocking.

    Returns (edge, halo_dt, work).
    - edge: the contiguous send buffer; caller must keep it alive until work completes.
    - halo_dt: pre-allocated recv buffer, filled when work completes.
    - work: list of NCCL request handles; caller calls req.wait() on each.
    """
    rank = dist.get_rank(sp_group)
    world_size = dist.get_world_size(sp_group)
    B, d, _ = x_dt.shape
    edge = x_dt[:, :, -(k - 1) :].contiguous()
    halo_dt = torch.zeros(B, d, k - 1, dtype=x_dt.dtype, device=x_dt.device)  # type: ignore
    ops = _fwd_p2p_ops(edge, halo_dt, rank, world_size, sp_group)
    work = dist.batch_isend_irecv(ops) if ops else []
    return edge, halo_dt, work


class FusedHaloConvSP(torch.autograd.Function):
    """Fused SP halo exchange + causal depthwise conv.

    The caller owns the forward recv: call begin_halo_recv, do other compute,
    wait for work, then pass halo_dt here. This lets inter-layer compute overlap
    with the in-flight P2P transfer.

    Backward issues batch_isend_irecv immediately after the Triton bwd kernel --
    both the grad send (to left neighbour) and grad recv (from right neighbour) are
    in flight before waiting. This eliminates the serialization bubble of the old
    two-Function design where the send was delayed until the autograd engine
    dispatched a separate SPHaloExchange.backward call.
    """

    @staticmethod
    def forward(
        ctx,
        x_dt: torch.Tensor,
        halo_dt: torch.Tensor,
        W_conv: torch.Tensor,
        T: int,
        k: int,
        BLOCK_T: int,
        sp_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        rank = dist.get_rank(sp_group)
        world_size = dist.get_world_size(sp_group)
        conv_out = causal_dwconv_fwd_sp(x_dt, halo_dt, W_conv, T, k, BLOCK_T)
        ctx.save_for_backward(x_dt, halo_dt, W_conv)
        ctx.T, ctx.k, ctx.BLOCK_T = T, k, BLOCK_T
        ctx.sp = (rank, world_size, sp_group)
        return conv_out

    @staticmethod
    def backward(ctx, d_conv_dt: torch.Tensor):  # type: ignore
        x_dt, halo_dt, W_conv = ctx.saved_tensors
        rank, world_size, sp_group = ctx.sp

        dX_dt, dHalo_dt, dW_conv = causal_dwconv_bwd_sp(
            d_conv_dt.contiguous(), x_dt, halo_dt, W_conv, ctx.T, ctx.k, ctx.BLOCK_T
        )

        grad_edge = torch.zeros_like(dHalo_dt)  # type: ignore
        ops = _bwd_p2p_ops(dHalo_dt, grad_edge, rank, world_size, sp_group)
        work = dist.batch_isend_irecv(ops) if ops else []
        for req in work:
            req.wait()

        if rank < world_size - 1:
            dX_dt[:, :, -(ctx.k - 1) :] = dX_dt[:, :, -(ctx.k - 1) :] + grad_edge

        return dX_dt, None, dW_conv, None, None, None, None
