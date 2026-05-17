"""Training throughput: tok/s and step time across batch/sequence configs.

Single-GPU:
    uv run python scripts/throughput.py
    uv run python scripts/throughput.py --precision bf16 --compile

Multi-GPU (sequence parallelism, T is split across ranks):
    torchrun --nproc_per_node=2 scripts/throughput.py
    torchrun --nproc_per_node=4 scripts/throughput.py --precision bf16
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch._functorch.config
import torch.distributed as dist

torch.set_float32_matmul_precision("high")

from gdnet.layer import freeze_sn_iteration
from gdnet.loss import projected_step
from gdnet.model import GDNet
from gdnet.utils.fp8 import Precision

VOCAB_SIZE = 100_000
N_WRITE = 4

# B*T
CONFIGS = [
    (4, 512),
    (8, 512),
    (4, 1024),
    (8, 1024),
    (16, 1024),
    (4, 2048),
    (8, 2048),
    (16, 2048),
    (4, 4096),
    (8, 4096),
    (16, 4096),
    (4, 8192),
    (8, 8192),
]


def setup_dist() -> tuple[int, int, dist.ProcessGroup | None]:
    """Init process group if running under torchrun. Returns (rank, world_size, sp_group)."""
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, None
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    sp_group = dist.group.WORLD
    return dist.get_rank(), dist.get_world_size(), sp_group


def make_model(T_local: int) -> GDNet:
    return GDNet(
        vocab_size=VOCAB_SIZE,
        d_embed=512,
        d=1024,
        n_layers=8,
        n_cycles=2,
        chunk_size=T_local,
    ).cuda()


def param_counts(model: GDNet) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    embed = sum(p.numel() for p in model.embed.parameters())
    return total, total - embed


def run_config(
    B: int,
    T: int,
    precision: Precision,
    use_cam: bool,
    compile_model: bool,
    sp_group: dist.ProcessGroup | None,
    world_size: int,
    n_warmup: int = 5,
    n_steps: int = 30,
) -> tuple[float, float, int, int, float]:
    T_local = T // world_size
    model = make_model(T_local)
    total_params, non_embed_params = param_counts(model)

    if compile_model:
        torch._functorch.config.donated_buffer = False
        model = torch.compile(model)  # type: ignore

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  # type: ignore
    params = list(model.parameters())  # type: ignore

    device = next(iter(model.parameters())).device  # type: ignore
    tokens = torch.randint(0, VOCAB_SIZE, (B, T_local), device=device)  # type: ignore
    targets = torch.randint(0, VOCAB_SIZE, (B, T_local), device=device)  # type: ignore
    write_chunks = (
        torch.randint(0, VOCAB_SIZE, (B, N_WRITE, T_local), device=device)  # type: ignore
        if use_cam and model.cam_enabled  # type: ignore
        else None
    )

    torch.cuda.reset_peak_memory_stats(device)

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
                sp_group=sp_group,
            )
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(device)
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
                sp_group=sp_group,
            )
    torch.cuda.synchronize()

    ms_per_step = (time.perf_counter() - t0) / n_steps * 1000
    tokens_per_sec = B * T / (ms_per_step / 1000)  # full sequence across all ranks
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1024**3

    del model
    torch.cuda.empty_cache()
    return ms_per_step, tokens_per_sec, total_params, non_embed_params, peak_mem_gb


def print_table(
    results: list[tuple[int, int, float, float, int, int, float]],
    cam_label: str,
    world_size: int,
) -> None:
    print(f"\ncam={cam_label}  gpus={world_size}")
    print(
        f"{'B':>4}  {'T':>4}  {'ms/step':>10}  {'tok/s':>12}  {'total params':>14}  {'non-embed':>12}  {'mem/gpu':>9}"
    )
    print("-" * 82)
    for B, T, ms, tps, total, non_embed, mem_gb in results:
        if ms != ms:
            print(f"{B:>4}  {T:>4}  {'OOM':>10}")
        else:
            print(
                f"{B:>4}  {T:>4}  {ms:>10.2f}  {tps:>12,.0f}"
                f"  {total / 1e6:>11.2f}M  {non_embed / 1e6:>9.2f}M"
                f"  {mem_gb:>7.2f}G"
            )


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
            print("TransformerEngine not found — cannot run fp8")
            return

    rank, world_size, sp_group = setup_dist()
    compile_flag = args.compile
    cam_modes = [True, False] if args.cam == "both" else [args.cam == "on"]

    if rank == 0:
        print(f"precision={precision}  compile={compile_flag}  gpus={world_size}")
        print(f"Device: {torch.cuda.get_device_name(0)}")

    for use_cam in cam_modes:
        results = []
        for B, T in CONFIGS:
            if T % world_size != 0:
                if rank == 0:
                    print(
                        f"  skip B={B} T={T}: not divisible by world_size={world_size}"
                    )
                continue
            try:
                ms, tps, total, non_embed, mem_gb = run_config(
                    B, T, precision, use_cam, compile_flag, sp_group, world_size
                )
                results.append((B, T, ms, tps, total, non_embed, mem_gb))
            except torch.cuda.OutOfMemoryError:
                results.append((B, T, float("nan"), float("nan"), 0, 0, 0.0))
                torch.cuda.empty_cache()
        if rank == 0:
            print_table(results, "on" if use_cam else "off", world_size)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
