"""GDNet language model pre-training.

Single-GPU:
    uv run python scripts/training.py

Multi-GPU (DDP + sequence parallelism, T split across ranks):
    torchrun --nproc_per_node=4 scripts/training.py --config configs/training.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import sys
import threading
import time
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import pynvml
import torch
import torch._functorch.config
import yaml
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.table import Table
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from gdnet.data import FineWebEduDataset, NemotronDataset
from gdnet.layer import freeze_sn_iteration
from gdnet.loss import projected_step
from gdnet.model import GDNet
from gdnet.utils.distributed import (
    destroy,
    get_rank,
    get_sp_group,
    get_world_size,
    init_distributed,
    is_main_process,
)
from gdnet.utils.fp8 import Precision, convert_to_fp8

torch.set_float32_matmul_precision("high")


@dataclasses.dataclass
class Config:
    # model
    vocab_size: int = 100_277
    d_embed: int = 512
    d: int = 1024
    n_layers: int = 8
    n_cycles: int = 2
    kernel_size: int = 7
    n_slots: int = 32
    # data
    tokenizer: str = "cl100k_base"
    token_budget: int = 32_768
    seed: int = 42
    # optimizer
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    beta_proj: float = 0.1
    # WSD schedule
    warmup_steps: int = 2_000
    decay_start: int = 80_000
    # curriculum
    phases: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    cam_start: int = 10_000
    trans_start: int = 20_000
    n_write: int = 4
    # precision / compile
    precision: str = "bf16"
    compile: bool = False
    # infra
    ckpt_dir: str = "checkpoints"
    ckpt_every: int = 1_000
    resume: str | None = None
    log_every: int = 10

    @property
    def total_steps(self) -> int:
        return sum(p["steps"] for p in self.phases)

    @classmethod
    def from_yaml(cls, path: str) -> Config:
        with open(path) as f:
            d: dict[str, Any] = yaml.safe_load(f) or {}
        fields = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - fields
        if unknown:
            raise ValueError(f"unknown config keys: {unknown}")
        return cls(**{k: v for k, v in d.items() if v is not None})


def _gpu_table(handles: list) -> Table:
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
            f"[{uc}]{util.gpu}%[/]",  # type: ignore
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


class TrainingMonitor:
    _MAX_ROWS = 15

    def __init__(self, handles: list, total_steps: int) -> None:
        self._handles = handles
        self._total_steps = total_steps
        self._step = 0
        self._phase = ""
        self._rows: list[tuple] = []
        self._stop = threading.Event()
        self._live: Live | None = None

    def start(self, live: Live) -> None:
        self._live = live
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def set_phase(self, label: str) -> None:
        self._phase = label
        self._rows.clear()

    def tick(self, step: int) -> None:
        self._step = step

    def log(self, step: int, loss: float, lr: float, tps: float, ms: float, tok_per_step: int) -> None:
        self._step = step
        self._rows.append((step, loss, lr, tps, ms, tok_per_step))
        if len(self._rows) > self._MAX_ROWS:
            self._rows.pop(0)

    def _metrics_table(self) -> Table:
        pct = self._step / max(self._total_steps, 1) * 100
        t = Table(
            title=f"{self._phase}  step={self._step}/{self._total_steps}  ({pct:.1f}%)",
            expand=True,
        )
        t.add_column("step", justify="right")
        t.add_column("loss", justify="right")
        t.add_column("lr", justify="right")
        t.add_column("tok/s", justify="right")
        t.add_column("tok/step", justify="right")
        t.add_column("ms/step", justify="right")
        for step, loss, lr, tps, ms, tpst in self._rows:
            t.add_row(str(step), f"{loss:.4f}", f"{lr:.2e}", f"{tps:,.0f}", f"{tpst:,}", f"{ms:.1f}")
        return t

    def _render(self) -> Columns:
        return Columns([self._metrics_table(), _gpu_table(self._handles)])

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._live is not None:
                self._live.update(self._render())
            self._stop.wait(0.5)


def make_dataset(
    phase: dict[str, Any], cfg: Config
) -> torch.utils.data.IterableDataset:
    source = phase.get("dataset", "fineweb-edu")
    seq_len = phase["seq_len"]

    if source == "fineweb-edu":
        return FineWebEduDataset(
            seq_len=seq_len,
            min_int_score=phase["min_int_score"],
            max_int_score=phase["max_int_score"],
            encoding=cfg.tokenizer,
            subset=phase.get("subset", "sample-10BT"),
            seed=cfg.seed,
        )

    if source == "nemotron":
        return NemotronDataset(
            seq_len=seq_len,
            encoding=cfg.tokenizer,
            subset=phase.get("subset", "everything"),
            min_chars=phase.get("min_chars", 0),
            seed=cfg.seed,
        )

    raise ValueError(f"unknown dataset source '{source}'")


def wsd_lr(step: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return step / max(cfg.warmup_steps, 1)
    if step < cfg.decay_start:
        return 1.0
    t = (step - cfg.decay_start) / max(cfg.total_steps - cfg.decay_start, 1)
    return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * t)
    )


def save_ckpt(
    path: Path, step: int, model: torch.nn.Module, optimizer, scheduler
) -> None:
    base = getattr(model, "module", model)
    torch.save(
        {
            "step": step,
            "model": base.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        path,
    )


def load_ckpt(path: Path, model: torch.nn.Module, optimizer, scheduler) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    base = getattr(model, "module", model)
    base.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt["step"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training.yaml")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if not cfg.phases:
        raise ValueError("config must define at least one phase")

    precision: Precision = cfg.precision  # type: ignore

    init_distributed()
    rank = get_rank()
    world_size = get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    sp_group = get_sp_group()
    main_proc = is_main_process()
    device = torch.device(f"cuda:{local_rank}")  # type: ignore

    max_seq_len = max(p["seq_len"] for p in cfg.phases)
    if max_seq_len % world_size != 0:
        raise ValueError(
            f"max seq_len {max_seq_len} must be divisible by world_size {world_size}"
        )

    model = GDNet(
        vocab_size=cfg.vocab_size,
        d_embed=cfg.d_embed,
        d=cfg.d,
        n_layers=cfg.n_layers,
        n_cycles=cfg.n_cycles,
        chunk_size=max_seq_len // world_size,
        kernel_size=cfg.kernel_size,
        n_slots=cfg.n_slots,
    ).cuda()

    if precision in ("bf16", "fp8"):
        for mod in model.modules():
            if isinstance(mod, torch.nn.RMSNorm):
                mod.weight.data = mod.weight.data.to(torch.bfloat16)

    if precision == "fp8":
        convert_to_fp8(model)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], static_graph=True)  # type: ignore

    if cfg.compile:
        torch._functorch.config.donated_buffer = False
        model = torch.compile(model, dynamic=True)  # type: ignore

    base_model: GDNet = getattr(model, "module", model)  # type: ignore

    optimizer = torch.optim.AdamW(
        model.parameters(),  # type: ignore
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )
    params = list(model.parameters())  # type: ignore

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: wsd_lr(s, cfg),
    )

    step = 0
    if cfg.resume:
        step = load_ckpt(Path(cfg.resume), model, optimizer, scheduler)  # type: ignore

    ckpt_dir = Path(cfg.ckpt_dir)

    if step >= cfg.cam_start:
        base_model.cam_enabled = True
    if step >= cfg.trans_start:
        base_model.trans_enabled = True

    if main_proc:
        pynvml.nvmlInit()
        handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i)
            for i in range(pynvml.nvmlDeviceGetCount())
        ]
        console = Console()
        n_params = sum(p.numel() for p in base_model.parameters())
        console.print(
            f"params={n_params / 1e6:.1f}M  precision={precision}  world={world_size}  total_steps={cfg.total_steps}"
        )
        if cfg.resume:
            console.print(f"resumed from {cfg.resume} at step {step}")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        monitor = TrainingMonitor(handles, cfg.total_steps)

    try:
        with (
            Live(console=console, refresh_per_second=4) if main_proc else nullcontext()  # type: ignore
        ) as live:
            if main_proc:
                monitor.start(live)  # type: ignore

            phase_start = 0
            for phase in cfg.phases:
                phase_end = phase_start + phase["steps"]

                if step >= phase_end:
                    phase_start = phase_end
                    continue

                source = phase.get("dataset", "fineweb-edu")
                T = phase["seq_len"]
                if T % world_size != 0:
                    raise ValueError(
                        f"phase seq_len {T} must be divisible by world_size {world_size}"
                    )
                T_local = T // world_size
                B = max(1, cfg.token_budget // T)

                if main_proc:
                    monitor.set_phase(  # type: ignore
                        f"{source}  seq_len={T}  B={B}  steps {phase_start}-{phase_end}"
                    )

                ds = make_dataset(phase, cfg)
                loader = DataLoader(ds, batch_size=B, num_workers=0, pin_memory=True)
                write_buf: deque[torch.Tensor] = deque(maxlen=cfg.n_write)
                t0 = time.perf_counter()

                for batch in loader:
                    if step >= phase_end:
                        break

                    if step == cfg.cam_start and not base_model.cam_enabled:
                        base_model.cam_enabled = True
                    if step == cfg.trans_start and not base_model.trans_enabled:
                        base_model.trans_enabled = True

                    seq = batch.to(device, non_blocking=True)
                    tokens = seq[:, :-1][:, rank * T_local : (rank + 1) * T_local]
                    targets = seq[:, 1:][:, rank * T_local : (rank + 1) * T_local]

                    write_chunks = None
                    if base_model.cam_enabled and len(write_buf) == cfg.n_write:
                        write_chunks = torch.stack(list(write_buf), dim=1)
                    write_buf.append(tokens.clone())

                    ctx = (
                        freeze_sn_iteration(model) if step % 50 != 0 else nullcontext()  # type: ignore
                    )  # type: ignore
                    with ctx:
                        loss = projected_step(
                            model,  # type: ignore
                            params,
                            optimizer,
                            tokens,
                            targets,
                            beta=cfg.beta_proj,
                            precision=precision,
                            write_chunks=write_chunks,
                            sp_group=sp_group,
                            grad_clip=cfg.grad_clip,
                        )

                    scheduler.step()
                    step += 1

                    if main_proc:
                        monitor.tick(step)  # type: ignore

                    if main_proc and step % cfg.log_every == 0:
                        dt = (time.perf_counter() - t0) / cfg.log_every
                        monitor.log(  # type: ignore
                            step,
                            loss,
                            scheduler.get_last_lr()[0],
                            B * T / dt,
                            dt * 1000,
                            B * T,
                        )
                        t0 = time.perf_counter()

                    if main_proc and step % cfg.ckpt_every == 0:
                        path = ckpt_dir / f"step_{step:07d}.pt"
                        save_ckpt(path, step, model, optimizer, scheduler)  # type: ignore
                        console.log(f"checkpoint: {path}")  # type: ignore

                phase_start = phase_end

            if main_proc:
                monitor.stop()  # type: ignore

    finally:
        if main_proc:
            path = ckpt_dir / f"step_{step:07d}_final.pt"
            save_ckpt(path, step, model, optimizer, scheduler)  # type: ignore
            console.log(f"final checkpoint: {path}")  # type: ignore
            pynvml.nvmlShutdown()
        destroy()


if __name__ == "__main__":
    main()
