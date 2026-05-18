from __future__ import annotations

from contextlib import nullcontext
from typing import Literal

import torch

Precision = Literal["fp32", "bf16", "fp8"]


def autocast(precision: Precision = "fp32"):
    """Return an autocast context for the requested precision.

    fp8 is handled at the model level via convert_to_fp8(); no context is
    needed here and nullcontext is returned for that case.
    """
    if precision == "fp32":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)  # type: ignore
    return nullcontext()


def convert_to_fp8(model: torch.nn.Module) -> torch.nn.Module:
    """Convert eligible nn.Linear layers to torchao Float8Linear in-place.

    Skips spectral-normalized linears (weight_orig in _parameters) because SN
    replaces weight with a plain tensor computed via hook, which Float8Linear
    cannot accept. Also skips layers whose dimensions aren't divisible by 16,
    which is an fp8 alignment requirement.

    Call once after model construction and before DDP/compile wrapping.
    Requires an fp8-capable GPU (sm_89+, e.g. H100/Ada).
    """
    from torchao.float8 import convert_to_float8_training  # type: ignore

    def _filter(mod: torch.nn.Module, fqn: str) -> bool:
        if "weight_orig" in mod._parameters:
            return False
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
        return True

    convert_to_float8_training(model, module_filter_fn=_filter)
    return model
