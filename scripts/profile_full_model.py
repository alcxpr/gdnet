"""Full model profiling: torch.profiler + cProfile + bytecode analysis.

Single GPU:
    uv run python scripts/profile_full_model.py --precision fp8

Multi-GPU (SP):
    torchrun --nproc_per_node=2 scripts/profile_full_model.py --precision fp8

Outputs:
    profiles/trace.json      Chrome trace (open in chrome://tracing or Perfetto)
    profiles/memory.json     Memory timeline
    profiles/cprofile.txt    Python-level hotspots
    profiles/bytecode.txt    Disassembly of key forward methods
"""

from __future__ import annotations

import argparse
import cProfile
import dis
import io
import pstats
import sys
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch._functorch.config
import torch.distributed as dist
import torch.profiler
from torch.nn.parallel import DistributedDataParallel as DDP

torch.set_float32_matmul_precision("high")

from gdnet.layer import freeze_sn_iteration
from gdnet.loss import projected_step
from gdnet.model import GDNet
from gdnet.utils.distributed import (
    destroy,
    get_sp_group,
    get_world_size,
    init_distributed,
    is_main_process,
)
from gdnet.utils.fp8 import Precision, convert_to_fp8

VOCAB_SIZE = 100_277
B = 4
T = 8192
N_WRITE = 4
OUT_DIR = Path(__file__).parent.parent / "profiles"


def make_model(T_local: int) -> GDNet:
    return GDNet(
        vocab_size=VOCAB_SIZE,
        d_embed=512,
        d=2048,
        n_layers=8,
        n_cycles=2,
        kernel_size=9,
        chunk_size=T_local,
    ).cuda()


def run(
    precision: Precision,
    compile_model: bool,
    sp_group: dist.ProcessGroup | None,
    world_size: int,
    local_rank: int,
    n_warmup: int = 5,
    n_profile_steps: int = 5,
) -> None:
    main_proc = is_main_process()
    T_local = T // world_size
    device = torch.device(f"cuda:{local_rank}")

    model = make_model(T_local)
    if precision == "fp8":
        convert_to_fp8(model)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], static_graph=True)  # type: ignore
    if compile_model:
        torch._functorch.config.donated_buffer = False
        model = torch.compile(model)  # type: ignore

    base_model: GDNet = getattr(model, "module", model)  # type: ignore

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  # type: ignore
    params = list(model.parameters())  # type: ignore

    tokens = torch.randint(0, VOCAB_SIZE, (B, T_local), device=device)
    targets = torch.randint(0, VOCAB_SIZE, (B, T_local), device=device)
    write_chunks = (
        torch.randint(0, VOCAB_SIZE, (B, N_WRITE, T_local), device=device)
        if base_model.cam_enabled
        else None
    )

    def step(i: int) -> None:
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

    for i in range(n_warmup):
        step(i)
    torch.cuda.synchronize(device)

    if main_proc:
        OUT_DIR.mkdir(exist_ok=True)

    # --- torch.profiler ---
    if main_proc:
        print("\n[1/3] torch.profiler ...", flush=True)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for i in range(n_profile_steps):
            step(n_warmup + i)
            prof.step()

    if main_proc:
        prof.export_chrome_trace(str(OUT_DIR / "trace.json"))
        try:
            prof.export_memory_timeline(
                str(OUT_DIR / "memory.json"), device=f"cuda:{local_rank}"
            )
        except Exception:
            pass
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
        print(f"\nChrome trace -> {OUT_DIR / 'trace.json'}")

    # --- cProfile ---
    if main_proc:
        print("\n[2/3] cProfile ...", flush=True)

    pr = cProfile.Profile()
    pr.enable()
    for i in range(n_profile_steps):
        step(n_warmup + n_profile_steps + i)
    torch.cuda.synchronize(device)
    pr.disable()

    if main_proc:
        buf = io.StringIO()
        ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
        ps.print_stats(40)
        txt = buf.getvalue()
        print(txt)
        (OUT_DIR / "cprofile.txt").write_text(txt)
        print(f"cProfile -> {OUT_DIR / 'cprofile.txt'}")

    # --- Bytecode ---
    if main_proc:
        print("\n[3/3] Bytecode (key forward methods) ...", flush=True)
        buf = io.StringIO()
        targets_map: list[tuple[str, object]] = [
            ("GDNet.forward", base_model.forward),
            ("GDNet.one_cycle", base_model.one_cycle),
        ]
        if base_model.layers:
            targets_map.append(("GDLayer.forward", base_model.layers[0].forward))
        for name, fn in targets_map:
            buf.write(f"\n{'=' * 60}\n{name}\n{'=' * 60}\n")
            dis.dis(fn, file=buf)  # type: ignore
        txt = buf.getvalue()
        (OUT_DIR / "bytecode.txt").write_text(txt)
        print(f"Bytecode -> {OUT_DIR / 'bytecode.txt'}")
        suspicious = [
            line
            for line in txt.splitlines()
            if any(
                kw in line for kw in ("contiguous", "clone", "copy_", "float(", "to(")
            )
        ]
        if suspicious:
            print("\nSuspicious intermediates found in bytecode:")
            for line in suspicious[:20]:
                print(" ", line.strip())
        else:
            print("No obvious intermediates in bytecode.")

    if main_proc:
        peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
        reserved_gb = torch.cuda.max_memory_reserved(device) / 1024**3
        print(f"\nPeak allocated: {peak_gb:.2f} GB  reserved: {reserved_gb:.2f} GB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp8"], default="bf16")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    init_distributed()
    world_size = get_world_size()
    local_rank = int(__import__("os").environ.get("LOCAL_RANK", 0))
    sp_group = get_sp_group()

    try:
        run(
            precision=args.precision,
            compile_model=args.compile,
            sp_group=sp_group,
            world_size=world_size,
            local_rank=local_rank,
        )
    finally:
        destroy()


if __name__ == "__main__":
    main()
