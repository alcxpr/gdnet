from __future__ import annotations

import torch
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
        inv_scale_x: torch.Tensor,
        inv_scale_w: torch.Tensor,
        w_fp8_row: torch.Tensor,
        w_fp8_col: torch.Tensor,
        amax_w: torch.Tensor,
        amax_x: torch.Tensor,
    ) -> torch.Tensor:
        x_fp8, x_fp8_col, _ = quantize_fp8(x, scale=scale_x, need_col=True, amax_buf=amax_x)
        quantize_fp8(w, scale=scale_w, need_col=True, out_row=w_fp8_row, out_col=w_fp8_col, amax_buf=amax_w)
        ctx.save_for_backward(x_fp8_col, w_fp8_col, scale_x, scale_w, inv_scale_x, inv_scale_w)
        return torch._scaled_mm(  # type: ignore
            x_fp8,
            w_fp8_row.T,
            scale_a=inv_scale_x,
            scale_b=inv_scale_w,
            out_dtype=torch.bfloat16,  # type: ignore
        )

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore
        x_fp8_col, w_fp8_col, scale_x, scale_w, inv_scale_x, inv_scale_w = ctx.saved_tensors

        grad_out = grad_out.contiguous().bfloat16()
        with torch.no_grad():
            amax_go = grad_out.abs().amax().float()
            scale_go = (FP8_MAX / amax_go.clamp(min=1e-12)).unsqueeze(0)
            inv_scale_go = scale_go.reciprocal()

        go_fp8, go_fp8_col, _ = quantize_fp8(grad_out, scale=scale_go, need_col=True)
        dgrad = torch._scaled_mm(  # type: ignore
            w_fp8_col,
            go_fp8.T.contiguous(),
            scale_a=inv_scale_w,
            scale_b=inv_scale_go,
            out_dtype=torch.bfloat16,  # type: ignore
        ).T.contiguous()
        wgrad = torch._scaled_mm(  # type: ignore
            go_fp8_col,
            x_fp8_col.T,
            scale_a=inv_scale_go,
            scale_b=inv_scale_x,
            out_dtype=torch.bfloat16,  # type: ignore
        )
        return dgrad, wgrad, None, None, None, None, None, None, None, None


class FP8Linear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        out_f, in_f = linear.weight.shape
        self.weight = nn.Parameter(linear.weight.data.bfloat16())
        self.register_parameter(
            "bias",
            nn.Parameter(linear.bias.data.bfloat16()) if linear.bias is not None else None,
        )
        dev = linear.weight.device
        self.register_buffer("scale_x", torch.ones(1, device=dev))  # type: ignore
        self.register_buffer("scale_w", torch.ones(1, device=dev))  # type: ignore
        self.register_buffer("inv_scale_x", torch.ones(1, device=dev))  # type: ignore
        self.register_buffer("inv_scale_w", torch.ones(1, device=dev))  # type: ignore
        self.register_buffer("_w_fp8_row", torch.empty(out_f, in_f, dtype=torch.float8_e4m3fn, device=dev))  # type: ignore
        self.register_buffer("_w_fp8_col", torch.empty(in_f, out_f, dtype=torch.float8_e4m3fn, device=dev))  # type: ignore
        self.register_buffer("_amax_w", torch.zeros(1, dtype=torch.float32, device=dev))  # type: ignore
        self.register_buffer("_amax_x", torch.zeros(1, dtype=torch.float32, device=dev))  # type: ignore

    @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1]).bfloat16()
        out = _Fp8LinearFn.apply(
            x_flat,
            self.weight,
            self.scale_x,
            self.scale_w,
            self.inv_scale_x,
            self.inv_scale_w,
            self._w_fp8_row,
            self._w_fp8_col,
            self._amax_w,
            self._amax_x,
        )
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*shape[:-1], self.weight.shape[0])
