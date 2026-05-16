from __future__ import annotations

import torch
import torch.distributed as dist


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


class SPHaloExchange(torch.autograd.Function):
    """P2P halo exchange for causal depthwise conv under sequence parallelism.

    Takes the already-extracted right edge of x_dt and returns the left halo
    received from the neighbouring rank.

    Forward:
        edge (B, d, k-1) -> left_halo (B, d, k-1).
        Rank 0 receives zeros (no left neighbour).

    Backward:
        grad_left_halo -> grad_edge (same shape).
        Rank N-1 receives zeros (no right neighbour).
        Autograd propagates grad_edge back through the slice to x_dt,
        accumulating into dX_dt[:, :, -(k-1):].

    Args:
        edge: Last (k-1) tokens of x_dt, contiguous, shape (B, d, k-1).
        group: CP process group.

    Returns:
        left_halo: (B, d, k-1) context from the left neighbour rank.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        edge: torch.Tensor,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        ctx.sp = (rank, world_size, group)  # type: ignore

        left_halo = torch.zeros_like(edge)  # type: ignore
        ops = _fwd_p2p_ops(edge, left_halo, rank, world_size, group)
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()

        return left_halo

    @staticmethod
    def backward(  # type: ignore
        ctx: torch.autograd.function.FunctionCtx,
        grad_left_halo: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        rank, world_size, group = ctx.sp  # type: ignore
        grad_left_halo = grad_left_halo.contiguous()
        grad_edge = torch.zeros_like(grad_left_halo)  # type: ignore

        ops = _bwd_p2p_ops(grad_left_halo, grad_edge, rank, world_size, group)
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()

        return grad_edge, None
