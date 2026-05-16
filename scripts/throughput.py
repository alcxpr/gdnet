"""Training throughput: tokens/sec and step time across batch/sequence configs.

Runs projected_step (single fwd + two bwd via retain_graph) and reports
throughput for each (B, T) config at a given precision.

Usage:
    uv run python scripts/throughput.py
    uv run python scripts/throughput.py --precision bf16
    uv run python scripts/throughput.py --precision fp8
    uv run python scripts/throughput.py --no-cam   # skip write_chunks
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from gdnet.layer import freeze_sn_iteration
from gdnet.loss import projected_step
from gdnet.model import GDNet
from gdnet.utils.fp8 import Precision

VOCAB_SIZE = 1024
N_WRITE = 4

CONFIGS = [
    # (B, T)
    (1,  128),
    (4,  128),
    (8,  128),
    (16, 128),
    (4,  256),
    (4,  512),
    (8,  512),
]


def make_model(T: int) -> GDNet:
    return GDNet(
        vocab_size=VOCAB_SIZE,
        d_embed=64,
        d=256,
        n_layers=4,
        n_cycles=2,
        chunk_size=T,
    ).cuda()


def run_config(
    B: int,
    T: int,
    precision: Precision,
    use_cam: bool,
    n_warmup: int = 5,
    n_steps: int = 30,
) -> tuple[float, float]:
    model = make_model(T)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    params = list(model.parameters())

    tokens = torch.randint(0, VOCAB_SIZE, (B, T), device="cuda")
    targets = torch.randint(0, VOCAB_SIZE, (B, T), device="cuda")
    write_chunks = (
        torch.randint(0, VOCAB_SIZE, (B, N_WRITE, T), device="cuda")
        if use_cam and model.cam_enabled
        else None
    )

    for i in range(n_warmup):
        ctx = freeze_sn_iteration(model) if i % 50 != 0 else nullcontext()
        with ctx:
            projected_step(model, params, optimizer, tokens, targets,
                           precision=precision, write_chunks=write_chunks)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for i in range(n_steps):
        ctx = freeze_sn_iteration(model) if (i + n_warmup) % 50 != 0 else nullcontext()
        with ctx:
            projected_step(model, params, optimizer, tokens, targets,
                           precision=precision, write_chunks=write_chunks)
    torch.cuda.synchronize()

    ms_per_step = (time.perf_counter() - t0) / n_steps * 1000
    tokens_per_sec = B * T / (ms_per_step / 1000)

    del model
    torch.cuda.empty_cache()
    return ms_per_step, tokens_per_sec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp8"], default="fp32")
    parser.add_argument("--no-cam", action="store_true")
    args = parser.parse_args()

    precision: Precision = args.precision  # type: ignore
    use_cam = not args.no_cam

    if precision == "fp8":
        try:
            import transformer_engine  # type: ignore  # noqa: F401
        except ImportError:
            print("TransformerEngine not found — cannot run fp8")
            return

    print(f"precision={precision}  cam={'on' if use_cam else 'off'}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()
    print(f"{'B':>4}  {'T':>4}  {'ms/step':>10}  {'tok/s':>12}")
    print("-" * 38)

    for B, T in CONFIGS:
        try:
            ms, tps = run_config(B, T, precision, use_cam)
            print(f"{B:>4}  {T:>4}  {ms:>10.2f}  {tps:>12,.0f}")
        except torch.cuda.OutOfMemoryError:
            print(f"{B:>4}  {T:>4}  {'OOM':>10}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
