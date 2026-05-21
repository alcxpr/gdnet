from __future__ import annotations

import pytest
import torch

from gdnet.kernel.fp8_linear import quantize_fp8
from gdnet.kernel.fp8_linear.gemm import fp8_gemm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
    reason="FP8 GEMM requires SM90+",
)

FP8_MAX = 448.0


def _make_fp8(shape: tuple[int, ...], seed: int = 0) -> tuple[torch.Tensor, float]:
    torch.manual_seed(seed)
    x = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")  # type: ignore
    scale = FP8_MAX / x.float().abs().max().item()
    x_fp8, _, _ = quantize_fp8(x, scale=scale)
    inv_scale = 1.0 / scale
    return x_fp8.contiguous(), inv_scale


def _make_fp8_b(K: int, N: int, seed: int = 0) -> tuple[torch.Tensor, float]:
    x_fp8, inv_scale = _make_fp8((N, K), seed=seed)
    return x_fp8.T.contiguous(), inv_scale


def _reference(
    a_fp8: torch.Tensor,
    b_fp8: torch.Tensor,
    inv_scale_a: float,
    inv_scale_b: float,
) -> torch.Tensor:
    return ((a_fp8.float() @ b_fp8.float()) * inv_scale_a * inv_scale_b).to(
        torch.bfloat16  # type: ignore
    )


@pytest.mark.parametrize(
    "M,N,K",
    [
        (64, 64, 64),
        (128, 128, 128),
        (128, 256, 128),
        (256, 128, 256),
        (512, 512, 256),
    ],
)
def test_shapes(M, N, K):
    a_fp8, inv_a = _make_fp8((M, K), seed=0)
    b_fp8, inv_b = _make_fp8_b(K, N, seed=1)
    out = fp8_gemm(a_fp8, b_fp8, inv_a, inv_b)
    assert out.shape == (M, N)
    assert out.dtype == torch.bfloat16  # type: ignore


@pytest.mark.parametrize("M,N,K", [(128, 128, 128), (256, 128, 256), (128, 256, 128)])
def test_correctness(M, N, K):
    a_fp8, inv_a = _make_fp8((M, K), seed=2)
    b_fp8, inv_b = _make_fp8_b(K, N, seed=3)
    out = fp8_gemm(a_fp8, b_fp8, inv_a, inv_b)
    ref = _reference(a_fp8, b_fp8, inv_a, inv_b)
    torch.testing.assert_close(out, ref, atol=1.0, rtol=0.05)


def test_scale_applied():
    M, N, K = 128, 128, 128
    a_fp8, inv_a = _make_fp8((M, K), seed=20)
    b_fp8, inv_b = _make_fp8_b(K, N, seed=21)
    out_scaled = fp8_gemm(a_fp8, b_fp8, inv_a, inv_b)
    out_unit = fp8_gemm(a_fp8, b_fp8, 1.0, 1.0)
    expected = out_unit.float() * (inv_a * inv_b)
    torch.testing.assert_close(out_scaled.float(), expected, atol=1e-3, rtol=0.01)


def test_kernel_cache():
    M, N, K = 128, 128, 128
    from gdnet.kernel.fp8_linear.gemm import _kernel_cache

    a_fp8, inv_a = _make_fp8((M, K), seed=30)
    b_fp8, inv_b = _make_fp8_b(K, N, seed=31)
    fp8_gemm(a_fp8, b_fp8, inv_a, inv_b)
    size_after_first = len(_kernel_cache)
    fp8_gemm(a_fp8, b_fp8, inv_a, inv_b)
    assert len(_kernel_cache) == size_after_first


@pytest.mark.parametrize("M,N,K", [(128, 128, 128), (256, 256, 256)])
def test_forward_wgrad_interface(M, N, K):
    x_fp8, inv_x = _make_fp8((M, K), seed=40)
    x_col_fp8, _ = _make_fp8((K, M), seed=40)
    w_col_fp8, inv_w = _make_fp8_b(K, N, seed=41)

    fwd = fp8_gemm(x_fp8, w_col_fp8, inv_x, inv_w)
    assert fwd.shape == (M, N)

    wgrad = fp8_gemm(w_col_fp8.T.contiguous(), x_col_fp8, inv_w, inv_x)
    assert wgrad.shape == (N, M)
