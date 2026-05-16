"""Profile projected_step under fp32 / bf16 / fp8 precision.

Runs a full model training step (fwd + two bwd via retain_graph) and reports
wall-clock time per step. fp8 requires TransformerEngine on an H100/A100.

Usage:
    uv run python scripts/fp8_profile.py
    uv run python scripts/fp8_profile.py --torch-profile   # adds torch.profiler table
    uv run python scripts/fp8_profile.py --precision fp8   # fp8 only
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.profiler

from gdnet.layer import freeze_sn_iteration
from gdnet.loss import projected_step
from gdnet.model import GDNet
from gdnet.utils.fp8 import Precision

B, T = 4, 128
N_WRITE = 4
VOCAB_SIZE = 1024


def make_model() -> GDNet:
    return GDNet(
        vocab_size=VOCAB_SIZE,
        d_embed=64,
        d=256,
        n_layers=4,
        n_cycles=2,
        chunk_size=T,
    ).cuda()


def make_batch():
    tokens = torch.randint(0, VOCAB_SIZE, (B, T), device="cuda")  # type: ignore
    targets = torch.randint(0, VOCAB_SIZE, (B, T), device="cuda")  # type: ignore
    write_chunks = torch.randint(0, VOCAB_SIZE, (B, N_WRITE, T), device="cuda")  # type: ignore
    return tokens, targets, write_chunks


def step(
    model,
    optimizer,
    params,
    tokens,
    targets,
    write_chunks,
    precision: Precision,
    i: int,
):
    ctx = freeze_sn_iteration(model) if i % 50 != 0 else nullcontext()
    with ctx:
        projected_step(
            model,
            params,
            optimizer,
            tokens,
            targets,
            precision=precision,
            write_chunks=write_chunks if model.cam_enabled else None,
        )


def bench(
    model: GDNet,
    precision: Precision,
    n_warmup: int = 5,
    n_steps: int = 20,
) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    params = list(model.parameters())
    tokens, targets, write_chunks = make_batch()

    for i in range(n_warmup):
        step(model, optimizer, params, tokens, targets, write_chunks, precision, i)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for i in range(n_steps):
        step(
            model,
            optimizer,
            params,
            tokens,
            targets,
            write_chunks,
            precision,
            i + n_warmup,
        )
    torch.cuda.synchronize()

    return (time.perf_counter() - t0) / n_steps * 1000


def run_torch_profile(model: GDNet, precision: Precision, n_steps: int = 3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    params = list(model.parameters())
    tokens, targets, write_chunks = make_batch()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for i in range(n_steps):
            step(model, optimizer, params, tokens, targets, write_chunks, precision, i)
    return prof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--precision",
        choices=["fp32", "bf16", "fp8", "all"],
        default="all",
    )
    parser.add_argument("--torch-profile", action="store_true")
    args = parser.parse_args()

    precisions: list[Precision] = (
        ["fp32", "bf16", "fp8"] if args.precision == "all" else [args.precision]  # type: ignore
    )

    if "fp8" in precisions:
        try:
            import transformer_engine  # type: ignore
        except ImportError:
            print("TransformerEngine not found - skipping fp8")
            precisions = [p for p in precisions if p != "fp8"]

    print(f"B={B} T={T} n_write={N_WRITE} d=256 n_layers=4 n_cycles=2")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    for precision in precisions:
        model = make_model()
        ms = bench(model, precision)
        print(f"{precision:<6}  {ms:.2f} ms/step")

        if args.torch_profile:
            print(f"\n--- torch.profiler ({precision}) ---")
            model2 = make_model()
            prof = run_torch_profile(model2, precision)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
            del model2

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
