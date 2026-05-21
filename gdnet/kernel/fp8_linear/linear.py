from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from .quantize import FP8_MAX, quantize_fp8


class FP8Linear(nn.Module):
    def __init__(self, linear: nn.Linear, scale_update_freq: int = 16):
        super().__init__()
        self.weight = nn.Parameter(linear.weight.data)
        self.bias = linear.bias
        self.scale_update_freq = scale_update_freq
        dev = linear.weight.device
        self.register_buffer("scale_x", torch.ones(1, device=dev))  # type: ignore
        self.register_buffer("scale_w", torch.ones(1, device=dev))  # type: ignore
        self.register_buffer("_step", torch.zeros(1, dtype=torch.int32, device=dev))  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])

        if self._step % self.scale_update_freq == 0:  # type: ignore
            amax = torch.stack(
                [x_flat.float().abs().max(), self.weight.float().abs().max()]
            )
            if dist.is_initialized():
                dist.all_reduce(amax, op=dist.ReduceOp.MAX)
            self.scale_x.fill_(FP8_MAX / amax[0].clamp(min=1e-12))  # type: ignore
            self.scale_w.fill_(FP8_MAX / amax[1].clamp(min=1e-12))  # type: ignore

        self._step.add_(1)  # type: ignore
        x_fp8, _, _ = quantize_fp8(x_flat, scale=self.scale_x.item(), need_col=False)  # type: ignore
        _, w_fp8_t, _ = quantize_fp8(self.weight, scale=self.scale_w.item(), need_col=True)  # type: ignore
        out = torch._scaled_mm(  # type: ignore
            x_fp8,
            w_fp8_t,
            scale_a=self.scale_x.reciprocal(),  # type: ignore
            scale_b=self.scale_w.reciprocal(),  # type: ignore
            out_dtype=torch.bfloat16,  # type: ignore
        )
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*shape[:-1], self.weight.shape[0])
