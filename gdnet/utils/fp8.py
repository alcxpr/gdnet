from __future__ import annotations

from contextlib import nullcontext
from typing import Literal

import torch

Precision = Literal["fp32", "bf16", "fp8"]


def autocast(precision: Precision = "fp32"):
    """Return an autocast context for the requested precision.

    fp8 uses bfloat16 autocast so that SN linears and norms compute in bf16
    rather than fp32. Float8Linear layers handle their own fp8 casting and are
    unaffected by the outer autocast.
    """
    if precision == "fp32":
        return nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16)  # type: ignore


def convert_to_fp8(model: torch.nn.Module) -> torch.nn.Module:
    """Convert eligible nn.Linear layers to torchao Float8Linear in-place.

    Skips spectral-normalized linears (weight_orig in _parameters) because SN
    replaces weight with a plain tensor computed via hook, which Float8Linear
    cannot accept. Also skips layers whose dimensions aren't divisible by 16,
    which is an fp8 alignment requirement. CAM layers are skipped because they
    operate on (B, d_sig) where B is typically 4, breaking the FP8 grad_weight
    computation which requires the batch dim to be divisible by 16.

    Also casts RMSNorm weights to bfloat16 so the fused kernel can dispatch
    when activations are bfloat16 (as they are under the bf16 autocast above).

    Call once after model construction and before DDP/compile wrapping.
    Requires an fp8-capable GPU (sm_89+, e.g. H100/Ada).
    """
    from torchao.float8 import convert_to_float8_training  # type: ignore

    def _filter(mod: torch.nn.Module, fqn: str) -> bool:
        if "weight_orig" in mod._parameters:
            return False
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
        # CAM layers operate on (B, d_sig) where B is small and not
        # guaranteed to be divisible by 16, which breaks FP8 grad_weight.
        if fqn.startswith("cam.") or ".cam." in fqn:
            return False
        return True

    convert_to_float8_training(model, module_filter_fn=_filter)

    # Cast RMSNorm weights to bfloat16 so the fused implementation fires
    # when activations are bfloat16 under autocast.
    for mod in model.modules():
        if isinstance(mod, torch.nn.RMSNorm) and mod.weight is not None:
            mod.weight.data = mod.weight.data.bfloat16()

    return model
