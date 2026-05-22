from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mod

from ..kernel.gated_causal_depthwise_conv.conv import (
    causal_dwconv_bwd_sp,
    causal_dwconv_fwd_sp,
)

_SYMM_HANDLES: dict = {}
_COPY_STREAMS: dict[int, torch.cuda.Stream] = {}


def _copy_stream() -> torch.cuda.Stream:
    dev = torch.cuda.current_device()
    if dev not in _COPY_STREAMS:
        _COPY_STREAMS[dev] = torch.cuda.Stream(device=dev, priority=-1)
    return _COPY_STREAMS[dev]


class _SymmHaloHandle:
    def __init__(
        self,
        layer_id: int,
        B: int,
        km1: int,
        d: int,
        dtype: torch.dtype,  # type: ignore
        sp_group: dist.ProcessGroup,
    ) -> None:
        self.rank = dist.get_rank(sp_group)
        self.world_size = dist.get_world_size(sp_group)
        self.shape = (B, km1, d)
        self.dtype = dtype
        flat = B * km1 * d
        device = torch.device(f"cuda:{torch.cuda.current_device()}")  # type: ignore
        fwd_t = symm_mod.empty(flat, dtype=dtype, device=device)
        self.fwd_hdl = symm_mod.rendezvous(fwd_t, sp_group)
        self._fwd = fwd_t
        bwd_t = symm_mod.empty(flat, dtype=dtype, device=device)
        self.bwd_hdl = symm_mod.rendezvous(bwd_t, sp_group)
        self._bwd = bwd_t


def _get_symm_handle(
    layer_id: int,
    B: int,
    km1: int,
    d: int,
    dtype: torch.dtype,  # type: ignore
    sp_group: dist.ProcessGroup,
) -> _SymmHaloHandle:
    key = (layer_id, sp_group, B, km1, d, dtype)
    if key not in _SYMM_HANDLES:
        _SYMM_HANDLES[key] = _SymmHaloHandle(layer_id, B, km1, d, dtype, sp_group)
    return _SYMM_HANDLES[key]


class FusedHaloConvSP(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        W_conv: torch.Tensor,
        T: int,
        k: int,
        BLOCK_T: int,
        sp_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        rank = dist.get_rank(sp_group)
        world_size = dist.get_world_size(sp_group)
        B, _, d = x.shape
        km1 = k - 1

        hdl = _get_symm_handle(W_conv.data_ptr(), B, km1, d, x.dtype, sp_group)

        if rank < world_size - 1:
            edge = x[:, -km1:, :].contiguous()
            cs = _copy_stream()
            edge_ready = torch.cuda.Event()
            edge_ready.record()
            cs.wait_event(edge_ready)
            with torch.cuda.stream(cs):
                peer_fwd = hdl.fwd_hdl.get_buffer(rank + 1, hdl.shape, x.dtype)
                peer_fwd.copy_(edge)
                hdl.fwd_hdl.put_signal(rank + 1)

        if rank > 0:
            hdl.fwd_hdl.wait_signal(rank - 1)
            halo = hdl._fwd.view(B, km1, d).clone()
        else:
            halo = torch.zeros(B, km1, d, dtype=x.dtype, device=x.device)  # type: ignore

        conv_out = causal_dwconv_fwd_sp(x, halo, W_conv, T, k, BLOCK_T)
        ctx.save_for_backward(x, halo, W_conv)
        ctx.T, ctx.k, ctx.BLOCK_T = T, k, BLOCK_T
        ctx.sp = (rank, world_size, sp_group)
        ctx.hdl = hdl
        return conv_out

    @staticmethod
    def backward(ctx, d_conv: torch.Tensor):  # type: ignore
        x, halo, W_conv = ctx.saved_tensors
        rank, world_size, sp_group = ctx.sp
        hdl: _SymmHaloHandle = ctx.hdl
        B, _, d = x.shape
        k = ctx.k

        dX, dHalo, dW_conv = causal_dwconv_bwd_sp(
            d_conv.contiguous(), x, halo, W_conv, ctx.T, k, ctx.BLOCK_T
        )

        if rank > 0:
            cs = _copy_stream()
            kernel_done = torch.cuda.Event()
            kernel_done.record()
            cs.wait_event(kernel_done)
            with torch.cuda.stream(cs):
                peer_bwd = hdl.bwd_hdl.get_buffer(rank - 1, hdl.shape, x.dtype)
                peer_bwd.copy_(dHalo)
                hdl.bwd_hdl.put_signal(rank - 1)

        if rank < world_size - 1:
            hdl.bwd_hdl.wait_signal(rank + 1)
            dX[:, -(k - 1) :, :].add_(hdl._bwd.view(B, k - 1, d))

        return dX, dW_conv, None, None, None, None
