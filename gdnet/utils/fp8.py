from __future__ import annotations

from contextlib import nullcontext
from typing import Literal

import torch

Precision = Literal["fp32", "bf16", "fp8"]


def autocast(precision: Precision = "fp32", amax_history_len: int = 16):
    """Return an autocast context for the requested precision.

    fp8 requires TransformerEngine and uses HYBRID format (E4M3 forward,
    E5M2 backward) with delayed scaling. te.Linear module replacement is
    intentionally avoided as it breaks FSDP compute/communication overlap.

    Args:
        precision: "fp32" disables autocast, "bf16" uses torch.autocast,
            "fp8" uses TransformerEngine fp8_autocast.
        amax_history_len: Rolling window length for amax tracking (fp8 only).
    """
    if precision == "fp32":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)  # type: ignore
    from transformer_engine.common.recipe import DelayedScaling, Format  # type: ignore
    from transformer_engine.pytorch import fp8_autocast  # type: ignore

    recipe = DelayedScaling(
        fp8_format=Format.HYBRID,  # type: ignore
        amax_history_len=amax_history_len,  # type: ignore
        amax_compute_algo="max",  # type: ignore
    )
    return fp8_autocast(enabled=True, fp8_recipe=recipe)
