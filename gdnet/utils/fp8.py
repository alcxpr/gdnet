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
    """Convert all nn.Linear layers in model to torchao Float8Linear in-place.

    Call once after model construction and before DDP/compile wrapping.
    Requires an fp8-capable GPU (sm_89+, e.g. H100/Ada).
    """
    from torchao.float8 import convert_to_float8_training  # type: ignore

    convert_to_float8_training(model)
    return model
