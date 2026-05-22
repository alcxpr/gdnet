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
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import pynvml
import torch
import torch._functorch.config
import watchfiles
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from gdnet.data import FineWebEduDataset, NemotronDataset, PackedTokenDataset
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
from gdnet.utils.fp8 import Precision, convert_to_fp8, update_fp8_scales

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
    accum_steps: int = 1
    # WSD schedule
    warmup_steps: int = 2_000
    decay_start: int = 80_000
    phases: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    n_write: int = 4
    # logging
    project: str = ""
    run: str = ""
    # precision / compile
    precision: str = "bf16"
    compile: bool = False
    # infra
    ckpt_dir: str = "checkpoints"
    ckpt_every: int = 1_000
    resume: str | None = None
    log_every: int = 10
    num_data_workers: int = 4

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


@dataclasses.dataclass
class State:
    step: int = 0
    total_steps: int = 0
    loss: float = float("nan")
    lr: float = 0.0
    tok_per_sec: float = 0.0
    ms_per_step: float = 0.0
    phase: str = ""


class Shell:
    def __init__(
        self,
        state: State,
        optimizer: torch.optim.Optimizer,
        handles: list,
        stop_event: threading.Event,
        save_event: threading.Event,
    ) -> None:
        self._state = state
        self._optimizer = optimizer
        self._handles = handles
        self._stop = stop_event
        self._save = save_event

    def start(self) -> None:
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def _write(self, s: str) -> None:
        sys.stdout.write(s)
        sys.stdout.flush()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                sys.stdout.write(">> ")
                sys.stdout.flush()
                line = sys.stdin.readline()
                if not line:
                    break
                self._handle(line.strip())
            except (EOFError, OSError):
                break

    def _handle(self, line: str) -> None:
        parts = line.split()
        if not parts:
            return
        cmd, *args = parts
        s = self._state

        if cmd == "status":
            pct = s.step / max(s.total_steps, 1) * 100
            self._write(
                f"phase      : {s.phase}\n"
                f"step       : {s.step}/{s.total_steps}  ({pct:.1f}%)\n"
                f"loss       : {s.loss:.4f}\n"
                f"lr         : {s.lr:.2e}\n"
                f"throughput : {s.tok_per_sec:,.0f} tok/s\n"
                f"ms/step    : {s.ms_per_step:.1f}\n"
            )

        elif cmd == "eta":
            remaining = s.total_steps - s.step
            if s.ms_per_step > 0 and remaining > 0:
                secs = int(remaining * s.ms_per_step / 1000)
                h, r = divmod(secs, 3600)
                m, sc = divmod(r, 60)
                self._write(f"eta: {h}h {m}m {sc}s  ({remaining} steps)\n")
            else:
                self._write("eta: unknown\n")

        elif cmd == "gpu":
            for i, h in enumerate(self._handles):
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                used = mem.used / 1024**3
                total = mem.total / 1024**3
                self._write(
                    f"gpu[{i}]  util={util.gpu}%  mem={used:.1f}/{total:.1f} GB  ({used/total*100:.1f}%)\n"
                )

        elif cmd == "save":
            self._save.set()
            self._write("checkpoint queued\n")

        elif cmd == "stop":
            self._stop.set()
            self._write("stopping after current step...\n")

        elif cmd == "lr":
            if not args:
                self._write(f"lr = {self._optimizer.param_groups[0]['lr']:.2e}\n")
            else:
                try:
                    val = float(args[0])
                    for pg in self._optimizer.param_groups:
                        pg["lr"] = val
                    self._write(f"lr -> {val:.2e}\n")
                except ValueError:
                    self._write(f"bad value: {args[0]}\n")

        elif cmd == "help":
            self._write("status  eta  gpu  save  stop  lr [val]  help\n")

        else:
            self._write(f"unknown: {cmd}\n")


class CudaPrefetcher:
    """Overlaps H2D transfer of the next batch with the current step's GPU compute.

    Wraps any DataLoader. Uses a dedicated CUDA stream so the transfer runs
    concurrently with the default stream's forward/backward. The main thread
    calls wait_stream before using the batch, which costs nothing when the
    transfer is already done.
    """

    def __init__(self, loader, device: torch.device) -> None:  # type: ignore
        self._loader = loader
        self._device = device
        self._stream = torch.cuda.Stream(device)
        self._next: torch.Tensor | None = None
        self._iter = iter(loader)
        self._preload()

    def _preload(self) -> None:
        try:
            batch = next(self._iter)
        except StopIteration:
            self._next = None
            return
        with torch.cuda.stream(self._stream):
            self._next = batch.to(self._device, non_blocking=True)

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        torch.cuda.current_stream(self._device).wait_stream(self._stream)
        batch = self._next
        if batch is None:
            raise StopIteration
        self._preload()
        return batch


def _wait_for_file(path: str, rank: int) -> None:
    """Block until path exists and is non-empty, using inotify on its parent directory."""
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        return
    if rank == 0:
        print(f"[data] waiting for {path} ...", flush=True)
    for _ in watchfiles.watch(
        str(p.parent), yield_on_timeout=True, rust_timeout=60_000
    ):
        if p.exists() and p.stat().st_size > 0:
            break


def make_dataset(
    phase: dict[str, Any], cfg: Config
) -> torch.utils.data.IterableDataset:
    source = phase.get("dataset", "fineweb-edu")
    seq_len = phase["seq_len"]

    path = phase.get("path")
    if path and Path(path).exists():
        return PackedTokenDataset(paths=path, seq_len=seq_len)

    if source == "packed":
        raise FileNotFoundError(f"packed dataset not found: {path}")

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

    if precision == "bf16":
        model.to(torch.bfloat16)  # type: ignore

    if precision == "fp8":
        model.embed.weight.data = model.embed.weight.data.bfloat16()  # type: ignore
        convert_to_fp8(model)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], static_graph=True, bucket_cap_mb=200)  # type: ignore

    if cfg.compile:
        torch._functorch.config.donated_buffer = False
        torch._dynamo.config.optimize_ddp = False
        model = torch.compile(model, dynamic=True)  # type: ignore

    base_model: GDNet = getattr(model, "module", model)  # type: ignore

    import bitsandbytes as bnb

    optimizer = bnb.optim.AdamW8bit(
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

    if main_proc:
        pynvml.nvmlInit()
        handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i)
            for i in range(pynvml.nvmlDeviceGetCount())
        ]
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        state = State(total_steps=cfg.total_steps, step=step)
        stop_event = threading.Event()
        save_event = threading.Event()
        shell = Shell(state, optimizer, handles, stop_event, save_event)
        shell.start()

        if cfg.project:
            import wandb

            wandb.init(
                project=cfg.project,
                name=cfg.run or None,
                config=dataclasses.asdict(cfg),
                resume="allow",
            )

    try:
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
                state.phase = f"{source}  seq_len={T}  B={B}  steps {phase_start}-{phase_end}"  # type: ignore

            if phase.get("path"):
                _wait_for_file(phase["path"], rank)
            ds = make_dataset(phase, cfg)
            loader = DataLoader(
                ds,
                batch_size=B,
                num_workers=cfg.num_data_workers,
                pin_memory=True,
                prefetch_factor=2 if cfg.num_data_workers > 0 else None,
                persistent_workers=cfg.num_data_workers > 0,
            )
            prefetcher = CudaPrefetcher(loader, device)
            write_buf: deque[torch.Tensor] = deque(maxlen=cfg.n_write)
            t0 = time.perf_counter()

            for seq in prefetcher:
                if step >= phase_end:
                    break
                if main_proc and stop_event.is_set():  # type: ignore
                    break

                tokens = seq[:, :-1][:, rank * T_local : (rank + 1) * T_local]
                targets = seq[:, 1:][:, rank * T_local : (rank + 1) * T_local]

                write_chunks = None
                if base_model.cam_enabled and len(write_buf) == cfg.n_write:
                    write_chunks = torch.stack(list(write_buf), dim=1)
                write_buf.append(tokens.clone())

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
                    accum_steps=cfg.accum_steps,
                )

                scheduler.step()
                step += 1
                if precision == "fp8" and step % 16 == 0:
                    update_fp8_scales(base_model)

                if main_proc:
                    state.step = step  # type: ignore

                if main_proc and step % cfg.log_every == 0:
                    dt = (time.perf_counter() - t0) / cfg.log_every
                    lr_now = scheduler.get_last_lr()[0]  # type: ignore
                    tps = B * T / dt
                    state.loss = loss  # type: ignore
                    state.lr = lr_now  # type: ignore
                    state.tok_per_sec = tps  # type: ignore
                    state.ms_per_step = dt * 1000  # type: ignore
                    if cfg.project:
                        wandb.log(  # type: ignore
                            {
                                "loss": loss,
                                "lr": lr_now,
                                "tok_per_sec": tps,
                                "ms_per_step": dt * 1000,
                            },
                            step=step,
                        )
                    t0 = time.perf_counter()

                if main_proc and step % cfg.ckpt_every == 0:
                    path = ckpt_dir / f"step_{step:07d}.pt"
                    save_ckpt(path, step, model, optimizer, scheduler)  # type: ignore

                if main_proc and save_event.is_set():  # type: ignore
                    save_event.clear()  # type: ignore
                    path = ckpt_dir / f"step_{step:07d}_manual.pt"
                    save_ckpt(path, step, model, optimizer, scheduler)  # type: ignore

            if main_proc and stop_event.is_set():  # type: ignore
                break
            phase_start = phase_end

    finally:
        if main_proc:
            path = ckpt_dir / f"step_{step:07d}_final.pt"
            save_ckpt(path, step, model, optimizer, scheduler)  # type: ignore
            stop_event.set()  # type: ignore
            if cfg.project:
                wandb.finish()  # type: ignore
            pynvml.nvmlShutdown()
        destroy()


if __name__ == "__main__":
    main()
