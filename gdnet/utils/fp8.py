from __future__ import annotations

from contextlib import nullcontext
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn as nn

from gdnet.kernel.fp8_linear import FP8Linear
from gdnet.kernel.fp8_linear.quantize import FP8_MAX

Precision = Literal["fp32", "bf16", "fp8"]


def update_fp8_scales(model: nn.Module) -> None:
    """Collect per-layer amax buffers, do one batched allreduce, update all scales.

    Call this from the training loop every scale_update_freq steps before the
    optimizer step. Resets amax buffers after reading so the next window starts
    clean.
    """
    mods = [m for m in model.modules() if isinstance(m, FP8Linear)]
    if not mods:
        return

    amax_parts: list[torch.Tensor] = []
    for m in mods:
        amax_parts.append(m._amax_x)
        amax_parts.append(m._amax_w)
    amaxes = torch.cat(amax_parts).clamp_(min=1e-12)

    if dist.is_initialized():
        dist.all_reduce(amaxes, op=dist.ReduceOp.MAX)

    for i, mod in enumerate(mods):
        ax = amaxes[2 * i : 2 * i + 1]
        aw = amaxes[2 * i + 1 : 2 * i + 2]
        torch.div(FP8_MAX, ax, out=mod.scale_x)
        torch.div(FP8_MAX, aw, out=mod.scale_w)
        torch.reciprocal(mod.scale_x, out=mod.inv_scale_x)
        torch.reciprocal(mod.scale_w, out=mod.inv_scale_w)
        mod._amax_x.zero_()
        mod._amax_w.zero_()


def autocast(precision: Precision = "fp32"):
    if precision == "fp32":
        return nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16)  # type: ignore


def convert_to_fp8(model: nn.Module) -> nn.Module:
    """Convert eligible nn.Linear layers to FP8Linear in-place.

    Scales are not updated here; call update_fp8_scales(model) from the training
    loop at the desired frequency. Amax buffers accumulate across calls between
    updates via tl.atomic_max in the Triton kernel.

    Skips spectral-normalized linears (weight_orig in _parameters) because SN
    replaces weight with a plain tensor computed via hook. Also skips layers
    whose dimensions are not divisible by 16 (torch._scaled_mm requirement).
    CAM layers are skipped because they operate on (B, d_sig) where B is small
    and not guaranteed to be divisible by 16.

    Call once after model construction and before DDP/compile wrapping.
    Requires an fp8-capable GPU (sm_89+, e.g. H100/Ada).
    """

    def _eligible(mod: nn.Module, fqn: str) -> bool:
        if not isinstance(mod, nn.Linear):
            return False
        if "weight_orig" in mod._parameters:
            return False
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
        if fqn.startswith("cam.") or ".cam." in fqn:
            return False
        return True

    for fqn, mod in list(model.named_modules()):
        if not _eligible(mod, fqn):
            continue
        parts = fqn.rsplit(".", 1)
        parent = model.get_submodule(parts[0]) if len(parts) == 2 else model
        attr = parts[-1]
        setattr(parent, attr, FP8Linear(mod))  # type: ignore

    for mod in model.modules():
        if "weight_orig" in mod._parameters:
            mod.bfloat16()

    for fqn, mod in model.named_modules():
        if isinstance(mod, nn.RMSNorm) and mod.weight is not None:
            if not (fqn.startswith("cam.") or ".cam." in fqn):
                mod.weight.data = mod.weight.data.bfloat16()

    cam = getattr(model, "cam", None)
    if cam is not None:
        cam.bfloat16()

    return model
