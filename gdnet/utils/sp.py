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
    x: torch.Tensor,
    k: int,
    sp_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor, list]:
    """Issue async P2P recv for the left halo without blocking.

    x: (B, T, d) contiguous input; edge slice is (B, k-1, d).

    Returns (edge, halo, work).
    - edge: contiguous send buffer (B, k-1, d); caller must keep alive until work completes.
    - halo: pre-allocated recv buffer (B, k-1, d), filled when work completes.
    - work: list of NCCL request handles; caller calls req.wait() on each.
    """
    rank = dist.get_rank(sp_group)
    world_size = dist.get_world_size(sp_group)
    B, _, d = x.shape
    edge = x[:, -(k - 1) :, :].contiguous()
    halo = torch.zeros(B, k - 1, d, dtype=x.dtype, device=x.device)  # type: ignore
    ops = _fwd_p2p_ops(edge, halo, rank, world_size, sp_group)
    work = dist.batch_isend_irecv(ops) if ops else []
    return edge, halo, work


class FusedHaloConvSP(torch.autograd.Function):
    """Fused SP halo exchange + causal depthwise conv.

    The caller owns the forward recv: call begin_halo_recv, do other compute,
    wait for work, then pass halo here. This lets inter-layer compute overlap
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
        x: torch.Tensor,
        halo: torch.Tensor,
        W_conv: torch.Tensor,
        T: int,
        k: int,
        BLOCK_T: int,
        sp_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        rank = dist.get_rank(sp_group)
        world_size = dist.get_world_size(sp_group)
        conv_out = causal_dwconv_fwd_sp(x, halo, W_conv, T, k, BLOCK_T)
        ctx.save_for_backward(x, halo, W_conv)
        ctx.T, ctx.k, ctx.BLOCK_T = T, k, BLOCK_T
        ctx.sp = (rank, world_size, sp_group)
        return conv_out

    @staticmethod
    def backward(ctx, d_conv: torch.Tensor):  # type: ignore
        x, halo, W_conv = ctx.saved_tensors
        rank, world_size, sp_group = ctx.sp

        dX, dHalo, dW_conv = causal_dwconv_bwd_sp(
            d_conv.contiguous(), x, halo, W_conv, ctx.T, ctx.k, ctx.BLOCK_T
        )

        grad_edge = torch.zeros_like(dHalo)  # type: ignore
        ops = _bwd_p2p_ops(dHalo, grad_edge, rank, world_size, sp_group)
        work = dist.batch_isend_irecv(ops) if ops else []
        for req in work:
            req.wait()

        if rank < world_size - 1:
            dX[:, -(ctx.k - 1) :, :] = dX[:, -(ctx.k - 1) :, :] + grad_edge

        return dX, None, dW_conv, None, None, None, None
