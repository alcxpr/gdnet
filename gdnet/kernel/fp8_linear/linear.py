from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from .quantize import FP8_MAX, quantize_fp8


class _Fp8LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        scale_x: torch.Tensor,
        scale_w: torch.Tensor,
    ) -> torch.Tensor:
        x_fp8, x_fp8_col, _ = quantize_fp8(x, scale=scale_x, need_col=True)
        w_fp8, w_fp8_col, _ = quantize_fp8(w, scale=scale_w, need_col=True)
        ctx.save_for_backward(x_fp8_col, w_fp8_col, scale_x, scale_w)
        return torch._scaled_mm(  # type: ignore
            x_fp8,
            w_fp8.T,
            scale_a=scale_x.reciprocal(),
            scale_b=scale_w.reciprocal(),
            out_dtype=torch.bfloat16,  # type: ignore
        )

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore
        x_fp8_col, w_fp8_col, scale_x, scale_w = ctx.saved_tensors

        grad_out = grad_out.contiguous().bfloat16()
        with torch.no_grad():
            amax_go = grad_out.float().abs().amax()
            scale_go = (FP8_MAX / amax_go.clamp(min=1e-12)).unsqueeze(0)

        go_fp8, go_fp8_col, _ = quantize_fp8(grad_out, scale=scale_go, need_col=True)
        dgrad = torch._scaled_mm(  # type: ignore
            w_fp8_col,
            go_fp8.T,
            scale_a=scale_w.reciprocal(),
            scale_b=scale_go.reciprocal(),
            out_dtype=torch.bfloat16,  # type: ignore
        ).T.contiguous()
        wgrad = torch._scaled_mm(  # type: ignore
            go_fp8_col,
            x_fp8_col.T,
            scale_a=scale_go.reciprocal(),
            scale_b=scale_x.reciprocal(),
            out_dtype=torch.bfloat16,  # type: ignore
        )
        return dgrad, wgrad, None, None


class FP8Linear(nn.Module):
    def __init__(self, linear: nn.Linear, scale_update_freq: int = 16):
        super().__init__()
        self.weight = nn.Parameter(linear.weight.data.bfloat16())
        self.register_parameter(
            "bias",
            nn.Parameter(linear.bias.data.bfloat16()) if linear.bias is not None else None,
        )
        self.scale_update_freq = scale_update_freq
        dev = linear.weight.device
        self.register_buffer("scale_x", torch.ones(1, device=dev))  # type: ignore
        self.register_buffer("scale_w", torch.ones(1, device=dev))  # type: ignore
        self._step: int = 0

    @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1]).bfloat16()

        if self._step % self.scale_update_freq == 0:
            with torch.no_grad():
                amax = torch.stack(
                    [x_flat.float().abs().max(), self.weight.float().abs().max()]
                )
                if dist.is_initialized():
                    dist.all_reduce(amax, op=dist.ReduceOp.MAX)
                self.scale_x.fill_(FP8_MAX / amax[0].clamp(min=1e-12))  # type: ignore
                self.scale_w.fill_(FP8_MAX / amax[1].clamp(min=1e-12))  # type: ignore

        self._step += 1

        out = _Fp8LinearFn.apply(
            x_flat,
            self.weight,
            self.scale_x,
            self.scale_w,
        )
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*shape[:-1], self.weight.shape[0])
