from __future__ import annotations

import pytest
import torch

from gdnet.kernel.fp8_linear import quantize_fp8

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
    reason="FP8 compute requires SM90+",
)

FP8_MAX = 448.0


def _reference(x: torch.Tensor, scale: float):
    x_f32 = x.float()
    x_fp8 = x_f32.mul(scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)  # type: ignore
    return x_fp8


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)


@pytest.mark.parametrize(
    "M,K",
    [
        (64, 64),
        (128, 256),
        (512, 1024),
        (100, 200),
        (65, 65),
        (1, 64),
        (64, 1),
    ],
)
def test_shapes(M, K):
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")  # type: ignore
    x_row, x_col, amax = quantize_fp8(x, scale=1.0)
    assert x_row.shape == (M, K)
    assert x_row.dtype == torch.float8_e4m3fn  # type: ignore
    assert x_col.shape == (K, M)
    assert x_col.dtype == torch.float8_e4m3fn  # type: ignore
    assert amax.shape == ()
    assert amax.dtype == torch.float32  # type: ignore


@pytest.mark.parametrize("M,K", [(128, 256), (100, 200), (65, 65)])
def test_rowwise_matches_reference(M, K):
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")  # type: ignore
    scale = FP8_MAX / x.float().abs().max().item()

    x_row, _, _ = quantize_fp8(x, scale=scale)
    ref = _reference(x, scale)

    assert torch.equal(x_row.float(), ref.float()), (  # type: ignore
        f"max diff: {(x_row.float() - ref.float()).abs().max()}"
    )


@pytest.mark.parametrize("M,K", [(128, 256), (100, 200), (65, 65)])
def test_colwise_is_transpose_of_rowwise(M, K):
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")  # type: ignore
    scale = FP8_MAX / x.float().abs().max().item()

    x_row, x_col, _ = quantize_fp8(x, scale=scale)

    assert torch.equal(x_col.float(), x_row.float().T.contiguous())  # type: ignore


@pytest.mark.parametrize("M,K", [(128, 256), (512, 512)])
def test_amax_correct(M, K):
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")  # type: ignore
    _, _, amax = quantize_fp8(x, scale=1.0)
    expected = x.float().abs().max()
    torch.testing.assert_close(amax, expected, atol=1e-3, rtol=0.0)


def test_scale_applied():
    M, K = 64, 64
    x = torch.ones(M, K, dtype=torch.bfloat16, device="cuda")  # type: ignore
    scale = 2.0
    x_row, _, _ = quantize_fp8(x, scale=scale)
    assert torch.all(x_row.float() == 2.0)  # type: ignore


def test_clamp_applied():
    M, K = 64, 64
    x = torch.full((M, K), 1000.0, dtype=torch.bfloat16, device="cuda")  # type: ignore
    x_row, _, _ = quantize_fp8(x, scale=1.0)
    assert torch.all(x_row.float() == FP8_MAX)  # type: ignore


def test_3d_input_shape():
    B, T, d = 2, 128, 256
    x = torch.randn(B, T, d, dtype=torch.bfloat16, device="cuda")  # type: ignore
    scale = 1.0
    x_row, x_col, amax = quantize_fp8(x, scale=scale)
    assert x_row.shape == (B, T, d)
    assert x_col.shape == (d, B * T)
    assert amax.shape == ()


def test_non_contiguous_input():
    x = torch.randn(256, 128, dtype=torch.bfloat16, device="cuda").T  # type: ignore
    assert not x.is_contiguous()
    M, K = x.shape
    x_row, x_col, _ = quantize_fp8(x, scale=1.0)
    assert x_row.shape == (M, K)
    assert x_col.shape == (K, M)
