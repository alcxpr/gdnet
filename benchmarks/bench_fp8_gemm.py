from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import triton

from gdnet.kernel.fp8_linear import fp8_gemm, quantize_fp8

FP8_MAX = 448.0

CONFIGS_SYNTHETIC = [
    (4096, 4096, 4096),
    (4096, 16384, 4096),
    (16384, 4096, 4096),
    (8192, 8192, 8192),
    (4096, 4096, 8192),
    (8192, 32768, 8192),
]

# d=2048, token_budget=32768 (B*T is always 32768)
# fwd W1/W2: (32768, 2048, 2048); fwd W1 gated: (32768, 2048, 4096)
# wgrad W1/W2: (2048, 2048, 32768); wgrad W1 gated: (2048, 4096, 32768)
# dgrad W1/W2: (32768, 2048, 2048); dgrad W1 gated: (32768, 4096, 2048)
CONFIGS_TRAINING = [
    (32768, 2048, 2048),
    (32768, 2048, 4096),
    (32768, 4096, 2048),
    (2048, 2048, 32768),
    (2048, 4096, 32768),
]

CONFIGS = CONFIGS_TRAINING

H200_FP8_PEAK_TFLOPS = 3958.0


def _make_fp8_pair(M: int, K: int, N: int, seed: int = 0):
    torch.manual_seed(seed)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")  # type: ignore
    b = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")  # type: ignore
    scale_a = FP8_MAX / a.float().abs().max().item()
    scale_b = FP8_MAX / b.float().abs().max().item()
    a_fp8, _, _ = quantize_fp8(a, scale=scale_a)
    b_fp8, _, _ = quantize_fp8(b, scale=scale_b)
    inv_a = 1.0 / scale_a
    inv_b = 1.0 / scale_b
    return a_fp8.contiguous(), b_fp8.contiguous(), inv_a, inv_b


def _flops(M: int, N: int, K: int) -> float:
    return 2.0 * M * N * K


def bench_fp8_gemm():
    print(f"{'M':>6} {'N':>6} {'K':>6}  {'ms':>8}  {'TFLOP/s':>10}  {'util%':>7}")
    print("-" * 55)
    for M, N, K in CONFIGS:
        a_fp8, b_fp8, inv_a, inv_b = _make_fp8_pair(M, K, N)
        fp8_gemm(a_fp8, b_fp8, inv_a, inv_b)

        ms = triton.testing.do_bench(
            lambda: fp8_gemm(a_fp8, b_fp8, inv_a, inv_b),
            warmup=25,
            rep=100,
        )
        tflops = _flops(M, N, K) / ms * 1e-9  # type: ignore
        util = tflops / H200_FP8_PEAK_TFLOPS * 100.0
        print(f"{M:>6} {N:>6} {K:>6}  {ms:>8.3f}  {tflops:>10.1f}  {util:>7.1f}")


def bench_baseline():
    print(
        f"\n{'M':>6} {'N':>6} {'K':>6}  {'ms':>8}  {'TFLOP/s':>10}  (F.linear bf16 baseline)"
    )
    print("-" * 58)
    import torch.nn.functional as F

    for M, N, K in CONFIGS:
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")  # type: ignore
        w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")  # type: ignore
        F.linear(x, w)

        ms = triton.testing.do_bench(
            lambda: F.linear(x, w),
            warmup=25,
            rep=100,
        )
        tflops = _flops(M, N, K) / ms * 1e-9  # type: ignore
        print(f"{M:>6} {N:>6} {K:>6}  {ms:>8.3f}  {tflops:>10.1f}")


if __name__ == "__main__":
    if torch.cuda.get_device_capability() < (9, 0):
        print("FP8 GEMM requires SM90+, skipping")
        sys.exit(0)

    bench_fp8_gemm()
    bench_baseline()
