"""Training throughput: tokens/sec, MFU, and step time across batch/sequence configs.

MFU (Model FLOP Utilization) is estimated as:
    flops_per_step / (step_time_s * peak_flops)

where flops_per_step = (1 fwd + 2 bwd) * 2*N_params*B*T
                     + N_WRITE * 2*N_params*B*T  (cam buffer writes, fwd only)

Peak FLOPS used: H100 non-sparse tensor core throughput.

Usage:
    uv run python scripts/throughput.py
    uv run python scripts/throughput.py --precision bf16
    uv run python scripts/throughput.py --precision bf16 --compile
    uv run python scripts/throughput.py --cam both   # run cam=on and cam=off, compare
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch._functorch.config

torch.set_float32_matmul_precision("high")

from gdnet.layer import freeze_sn_iteration
from gdnet.loss import projected_step
from gdnet.model import GDNet
from gdnet.utils.fp8 import Precision

VOCAB_SIZE = 1024
N_WRITE = 4

# Effective peak TFLOPS for MFU denominator.
# Triton kernels (gated conv, fused_mem_read) always run fp32 via custom_fwd cast,
# so the dominant compute path is fp32 regardless of autocast precision.
# bf16 tensor cores only benefit nn.Linear layers, which are a smaller fraction.
# fp8 entry reflects a hypothetical future CUTLASS/scaled_mm path.
H100_PEAK_TFLOPS: dict[str, float] = {
    "fp32": 67.0,
    "bf16": 67.0,
    "fp8": 1979.0,
}

CONFIGS = [
    # (B, T)
    (1, 128),
    (4, 128),
    (8, 128),
    (16, 128),
    (4, 256),
    (4, 512),
    (8, 512),
    (16, 512),
    (32, 512),
    (64, 512),
    (32, 1024),
]


def make_model(T: int) -> GDNet:
    return GDNet(
        vocab_size=VOCAB_SIZE,
        d_embed=128,
        d=1024,
        n_layers=8,
        n_cycles=2,
        chunk_size=T,
    ).cuda()


def non_embedding_params(model: GDNet) -> int:
    embed_params = sum(p.numel() for p in model.embed.parameters())
    return sum(p.numel() for p in model.parameters()) - embed_params


def estimate_flops(model: GDNet, B: int, T: int, use_cam: bool) -> float:
    N = non_embedding_params(model)
    tokens = B * T
    # Layers are weight-shared across cycles, so each param is used n_cycles times per fwd.
    # 1 fwd + 2 bwd ≈ 5x fwd; fwd ≈ 2*N*n_cycles*tokens
    step_flops = 5 * 2 * N * model.n_cycles * tokens  # type: ignore
    cam_flops = (
        N_WRITE * 2 * N * model.n_cycles * tokens
        if use_cam and model.cam_enabled
        else 0
    )  # type: ignore
    return step_flops + cam_flops


def run_config(
    B: int,
    T: int,
    precision: Precision,
    use_cam: bool,
    compile_model: bool,
    n_warmup: int = 5,
    n_steps: int = 30,
) -> tuple[float, float, float]:
    model = make_model(T)
    if compile_model:
        torch._functorch.config.donated_buffer = (
            False  # incompatible with retain_graph=True
        )
        model = torch.compile(model)  # type: ignore

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  # type: ignore
    params = list(model.parameters())  # type: ignore

    flops = estimate_flops(model, B, T, use_cam)  # type: ignore
    peak = H100_PEAK_TFLOPS[precision] * 1e12

    tokens = torch.randint(0, VOCAB_SIZE, (B, T), device="cuda")  # type: ignore
    targets = torch.randint(0, VOCAB_SIZE, (B, T), device="cuda")  # type: ignore
    write_chunks = (
        torch.randint(0, VOCAB_SIZE, (B, N_WRITE, T), device="cuda")  # type: ignore
        if use_cam and model.cam_enabled  # type: ignore
        else None
    )

    for i in range(n_warmup):
        ctx = freeze_sn_iteration(model) if i % 50 != 0 else nullcontext()  # type: ignore
        with ctx:
            projected_step(
                model,  # type: ignore
                params,
                optimizer,
                tokens,
                targets,
                precision=precision,
                write_chunks=write_chunks,
            )
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for i in range(n_steps):
        ctx = freeze_sn_iteration(model) if (i + n_warmup) % 50 != 0 else nullcontext()  # type: ignore
        with ctx:
            projected_step(
                model,  # type: ignore
                params,
                optimizer,
                tokens,
                targets,
                precision=precision,
                write_chunks=write_chunks,
            )
    torch.cuda.synchronize()

    ms_per_step = (time.perf_counter() - t0) / n_steps * 1000
    tokens_per_sec = B * T / (ms_per_step / 1000)
    mfu = flops / (ms_per_step / 1000 * peak) * 100

    del model
    torch.cuda.empty_cache()
    return ms_per_step, tokens_per_sec, mfu


def print_table(
    results: list[tuple[int, int, float, float, float]],
    cam_label: str,
) -> None:
    print(f"\ncam={cam_label}")
    print(f"{'B':>4}  {'T':>4}  {'ms/step':>10}  {'tok/s':>12}  {'MFU':>7}")
    print("-" * 48)
    for B, T, ms, tps, mfu in results:
        print(f"{B:>4}  {T:>4}  {ms:>10.2f}  {tps:>12,.0f}  {mfu:>6.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp8"], default="fp32")
    parser.add_argument("--cam", choices=["on", "off", "both"], default="on")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    precision: Precision = args.precision  # type: ignore

    if precision == "fp8":
        try:
            import transformer_engine  # type: ignore  # noqa: F401
        except ImportError:
            print("TransformerEngine not found - cannot run fp8")
            return

    compile_flag = args.compile
    cam_modes = [True, False] if args.cam == "both" else [args.cam == "on"]

    print(f"precision={precision}  compile={compile_flag}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    for use_cam in cam_modes:
        results = []
        for B, T in CONFIGS:
            try:
                ms, tps, mfu = run_config(B, T, precision, use_cam, compile_flag)
                results.append((B, T, ms, tps, mfu))
            except torch.cuda.OutOfMemoryError:
                results.append((B, T, float("nan"), float("nan"), float("nan")))
                torch.cuda.empty_cache()
                print(f"  B={B} T={T}: OOM")
        print_table(results, "on" if use_cam else "off")


if __name__ == "__main__":
    main()
