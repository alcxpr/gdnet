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
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pynvml
import torch
import torch._functorch.config
import torch.distributed as dist
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text
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
N_WRITE = 4

CONFIGS = [
    (4, 512),
    (8, 512),
    (16, 512),
    (4, 1024),
    (8, 1024),
    (16, 1024),
    (4, 2048),
    (8, 2048),
    (16, 2048),
    (4, 4096),
    (8, 4096),
    (16, 4096),
    (32, 4096),
    (64, 4096),
    (4, 8192),
    (8, 8192),
    (12, 8192),
    (16, 8192),
    (24, 8192),
    (32, 8192),
    (4, 16384),
    (8, 16384),
    (12, 16384),
    (16, 16384),
]


def gpu_table(handles: list) -> Table:
    t = Table(title="GPU", expand=True)
    t.add_column("GPU", no_wrap=True)
    t.add_column("Util", justify="right")
    t.add_column("VRAM Used", justify="right")
    t.add_column("VRAM Total", justify="right")
    t.add_column("VRAM %", justify="right")

    total_util, total_used, total_vram = 0.0, 0.0, 0.0
    for i, handle in enumerate(handles):
        name = pynvml.nvmlDeviceGetName(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_gb = mem.used / 1024**3  # type: ignore
        tot_gb = mem.total / 1024**3  # type: ignore
        pct = used_gb / tot_gb * 100
        uc = "green" if util.gpu < 50 else "yellow" if util.gpu < 85 else "red"  # type: ignore
        mc = "green" if pct < 50 else "yellow" if pct < 85 else "red"
        t.add_row(
            f"[{i}] {name}",
            f"[{uc}]{util.gpu}%[/]",
            f"[{mc}]{used_gb:.2f} GB[/]",
            f"{tot_gb:.2f} GB",
            f"[{mc}]{pct:.1f}%[/]",
        )
        total_util += util.gpu  # type: ignore
        total_used += used_gb
        total_vram += tot_gb

    n = len(handles)
    avg_pct = total_used / total_vram * 100
    au = "green" if total_util / n < 50 else "yellow" if total_util / n < 85 else "red"
    am = "green" if avg_pct < 50 else "yellow" if avg_pct < 85 else "red"
    t.add_section()
    t.add_row(
        f"[bold]avg ({n})[/bold]",
        f"[{au}]{total_util / n:.1f}%[/]",
        f"[{am}]{total_used / n:.2f} GB[/]",
        f"{total_vram / n:.2f} GB",
        f"[{am}]{avg_pct:.1f}%[/]",
    )
    return t


def results_table(rows: list, cam_label: str, world_size: int, status: str) -> Table:
    t = Table(title=f"cam={cam_label}  gpus={world_size}  {status}", expand=True)
    t.add_column("B", justify="right")
    t.add_column("T", justify="right")
    t.add_column("ms/step", justify="right")
    t.add_column("tok/s", justify="right")
    t.add_column("params", justify="right")
    t.add_column("mem/gpu", justify="right")
    for B, T, ms, tps, total, _, mem_gb in rows:
        if ms != ms:
            t.add_row(str(B), str(T), "[red]OOM[/]", "", "", "")
        else:
            t.add_row(
                str(B),
                str(T),
                f"{ms:.1f}",
                f"{tps:,.0f}",
                f"{total / 1e6:.1f}M",
                f"{mem_gb:.2f}G",
            )
    return t


class Monitor:
    def __init__(self, handles: list, interval: float = 0.5):
        self._handles = handles
        self._interval = interval
        self._stop = threading.Event()
        self._rows: list = []
        self._status = Text("")
        self._cam_label = ""
        self._world_size = 1
        self._live: Live | None = None

    def start(self, live: Live) -> None:
        self._live = live
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def set_context(self, cam_label: str, world_size: int) -> None:
        self._cam_label = cam_label
        self._world_size = world_size
        self._rows = []

    def set_status(self, status: str) -> None:
        self._status = Text(status)

    def push_row(self, row: tuple) -> None:
        self._rows.append(row)

    def _render(self) -> Columns:
        return Columns(
            [
                results_table(
                    self._rows, self._cam_label, self._world_size, str(self._status)
                ),
                gpu_table(self._handles),
            ]
        )

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._live is not None:
                self._live.update(self._render())
            self._stop.wait(self._interval)


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


def param_counts(model: GDNet) -> tuple[int, int]:
    total = sum(p.numel() for _, p in model.named_parameters(remove_duplicate=False))
    embed = model.embed.weight.numel()
    return total, total - embed


def run_config(
    B: int,
    T: int,
    precision: Precision,
    use_cam: bool,
    compile_model: bool,
    sp_group: dist.ProcessGroup | None,
    world_size: int,
    local_rank: int,
    n_warmup: int = 5,
    n_steps: int = 30,
) -> tuple[float, float, int, int, float]:
    T_local = T // world_size
    model = make_model(T_local)
    total_params, non_embed_params = param_counts(model)

    if precision == "bf16":
        model.to(torch.bfloat16)  # type: ignore
    if precision == "fp8":
        model.embed.weight.data = model.embed.weight.data.bfloat16()  # type: ignore
        convert_to_fp8(model)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], static_graph=True)  # type: ignore

    if compile_model:
        torch._functorch.config.donated_buffer = False
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        model = torch.compile(model)  # type: ignore

    base_model: GDNet = getattr(model, "module", model)  # type: ignore

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  # type: ignore
    params = list(model.parameters())  # type: ignore

    device = torch.device(f"cuda:{local_rank}")  # type: ignore
    tokens = torch.randint(0, VOCAB_SIZE, (B, T_local), device=device)  # type: ignore
    targets = torch.randint(0, VOCAB_SIZE, (B, T_local), device=device)  # type: ignore
    write_chunks = (
        torch.randint(0, VOCAB_SIZE, (B, N_WRITE, T_local), device=device)  # type: ignore
        if use_cam and base_model.cam_enabled
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
    torch.cuda.synchronize(device)

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
    torch.cuda.synchronize(device)

    ms_per_step = (time.perf_counter() - t0) / n_steps * 1000
    tokens_per_sec = B * T / (ms_per_step / 1000)
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1024**3

    del model
    torch.cuda.empty_cache()
    return ms_per_step, tokens_per_sec, total_params, non_embed_params, peak_mem_gb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp8"], default="fp32")
    parser.add_argument("--cam", choices=["on", "off", "both"], default="on")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    precision: Precision = args.precision  # type: ignore

    if precision == "fp8":
        try:
            import torchao.float8  # type: ignore  # noqa: F401
        except ImportError:
            print("torchao not found — cannot run fp8")
            return

    init_distributed()
    world_size = get_world_size()
    local_rank = int(__import__("os").environ.get("LOCAL_RANK", 0))
    sp_group = get_sp_group()
    compile_flag = args.compile
    cam_modes = [True, False] if args.cam == "both" else [args.cam == "on"]

    main_proc = is_main_process()

    if main_proc:
        pynvml.nvmlInit()
        n_gpus = pynvml.nvmlDeviceGetCount()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n_gpus)]
        console = Console()
        console.print(
            f"precision={precision}  compile={compile_flag}  gpus={world_size}"
        )
        console.print(f"Device: {torch.cuda.get_device_name(0)}")
        monitor = Monitor(handles)

    try:
        for use_cam in cam_modes:
            cam_label = "on" if use_cam else "off"

            if main_proc:
                monitor.set_context(cam_label, world_size)  # type: ignore

            with (
                Live(console=console, refresh_per_second=4)  # type: ignore
                if main_proc
                else nullcontext()
            ) as live:  # type: ignore
                if main_proc:
                    monitor.start(live)  # type: ignore

                for B, T in CONFIGS:
                    if T % world_size != 0:
                        if main_proc:
                            monitor.set_status(  # type: ignore
                                f"skip B={B} T={T}: not divisible by {world_size}"
                            )
                        continue

                    if main_proc:
                        monitor.set_status(f"running B={B} T={T} (warmup)...")  # type: ignore

                    try:
                        ms, tps, total, non_embed, mem_gb = run_config(
                            B,
                            T,
                            precision,
                            use_cam,
                            compile_flag,
                            sp_group,
                            world_size,
                            local_rank,
                        )
                        if main_proc:
                            monitor.set_status(f"done B={B} T={T}")  # type: ignore
                            monitor.push_row((B, T, ms, tps, total, non_embed, mem_gb))  # type: ignore
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if main_proc:
                            monitor.set_status(f"OOM at B={B} T={T}")  # type: ignore
                            monitor.push_row(  # type: ignore
                                (B, T, float("nan"), float("nan"), 0, 0, 0.0)
                            )

                if main_proc:
                    monitor.stop()  # type: ignore
    finally:
        if main_proc:
            pynvml.nvmlShutdown()
        destroy()


if __name__ == "__main__":
    main()
