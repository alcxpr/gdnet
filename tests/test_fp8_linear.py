from __future__ import annotations

import pytest
import torch

from gdnet.kernel.fp8_linear.linear import FP8Linear
from gdnet.utils.fp8 import update_fp8_scales

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
    reason="FP8 linear requires SM90+",
)

D = 2048


def _make_fp8linear() -> FP8Linear:
    linear = torch.nn.Linear(D, D).cuda().bfloat16()
    fp8 = FP8Linear(linear).cuda()
    update_fp8_scales(fp8)
    return fp8


@pytest.mark.parametrize(
    "B,T",
    [
        (4, 512),
        (16, 512),
        (64, 512),
        (16, 1024),
        (32, 1024),
        (64, 1024),
        (16, 4096),
        (64, 4096),
    ],
)
def test_backward_large_batch(B, T):
    fp8 = _make_fp8linear()
    x = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)  # type: ignore
    out = fp8(x)
    out.sum().backward()
    torch.cuda.synchronize()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_backward_grad_correct():
    torch.manual_seed(0)
    fp8 = _make_fp8linear()
    x = torch.randn(8, 64, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)  # type: ignore
    out = fp8(x)
    out.sum().backward()
    assert x.grad is not None
    assert not x.grad.isnan().any()
    assert not x.grad.isinf().any()
