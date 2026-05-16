from __future__ import annotations

import functools
import logging
import os
from datetime import timedelta
from typing import Any, Callable, Optional, Type

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.multiprocessing.spawn import spawn

log = logging.getLogger(__name__)

_FSDP_MIXED_PRECISION = MixedPrecision(
    param_dtype=torch.bfloat16,  # type: ignore
    reduce_dtype=torch.float32,  # type: ignore
    buffer_dtype=torch.bfloat16,  # type: ignore
)


def init_distributed(backend: str = "nccl", timeout_minutes: int = 30) -> None:
    """Initialize the default process group.

    Safe to call even if already initialized -- subsequent calls are no-ops.
    Sets the CUDA device to the local rank automatically.

    Args:
        backend: Distributed backend. "nccl" for GPU, "gloo" for CPU.
        timeout_minutes: Timeout for collective ops.

    Raises:
        RuntimeError: If CUDA is not available and backend is "nccl".
    """
    if dist.is_available() and dist.is_initialized():
        return

    if not dist.is_available():
        log.warning("torch.distributed not available, running single-process.")
        return

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size == 1:
        return

    if backend == "nccl" and not torch.cuda.is_available():
        raise RuntimeError("NCCL backend requires CUDA. Use backend='gloo' for CPU.")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_minutes))

    log.info(
        "dist initialized: rank=%d / world_size=%d, local_rank=%d, backend=%s",
        rank,
        world_size,
        local_rank,
        backend,
    )


def _detect_layer_cls(model: nn.Module) -> Optional[Type[nn.Module]]:
    core = getattr(model, "module", model)
    layers = getattr(core, "layers", None)
    if layers is not None and len(layers) > 0:
        return type(layers[0])
    log.warning(
        "Could not detect layer class, falling back to size_based_auto_wrap_policy."
    )
    return None


def wrap_fsdp(
    model: nn.Module,
    sharding_strategy: ShardingStrategy = ShardingStrategy.SHARD_GRAD_OP,
    offload: bool = False,
    min_num_params: int = 1_000_000,
) -> FSDP:
    """Wrap model in FSDP with sensible defaults for single-node multi-GPU training.

    Args:
        model: The model to wrap.
        sharding_strategy: FULL_SHARD by default. Use SHARD_GRAD_OP for faster
            forward passes at the cost of higher memory.
        offload: Enable CPU offload for parameters and gradients.
        min_num_params: Fallback size threshold when layer class detection fails.

    Returns:
        FSDP-wrapped model on the current CUDA device.

    Raises:
        RuntimeError: If called before init_distributed.
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "wrap_fsdp called before init_process_group. "
            "Call gdnet.utils.distributed.init_distributed() first."
        )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    layer_cls = _detect_layer_cls(model)
    if layer_cls is not None:
        auto_wrap = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={layer_cls},
        )
        log.info("FSDP auto-wrap: transformer_auto_wrap_policy(%s)", layer_cls.__name__)
    else:
        auto_wrap = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params,
        )
        log.info(
            "FSDP auto-wrap: size_based_auto_wrap_policy(min_params=%d)", min_num_params
        )

    wrapped = FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        sharding_strategy=sharding_strategy,
        mixed_precision=_FSDP_MIXED_PRECISION,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        cpu_offload=CPUOffload(offload_params=True) if offload else None,
        device_id=device,
        # Required for spectral norm: FSDP normally replaces params with flat
        # shards, breaking the weight_orig/weight_u/weight_v layout SN hooks expect.
        use_orig_params=True,
        sync_module_states=True,
    )

    log.info(
        "FSDP wrap complete. strategy=%s, offload=%s, device=%s",
        sharding_strategy.name,
        offload,
        device,
    )
    return wrapped


def get_rank() -> int:
    """Return global rank, or 0 if not distributed."""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    """Return world size, or 1 if not distributed."""
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_process() -> bool:
    """True for rank 0 only. Use to gate logging and checkpointing."""
    return get_rank() == 0


def barrier() -> None:
    """Block until all ranks reach this point. No-op if not distributed."""
    if dist.is_initialized():
        dist.barrier()


def destroy() -> None:
    """Tear down the process group. Call at end of training."""
    if dist.is_initialized():
        dist.destroy_process_group()
        log.info("dist process group destroyed.")


def _worker(
    rank: int,
    world_size: int,
    fn: Callable,
    args: tuple,
    kwargs: dict,
    backend: str,
    timeout_minutes: int,
    master_addr: str,
    master_port: int,
) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    init_distributed(backend=backend, timeout_minutes=timeout_minutes)
    try:
        fn(*args, **kwargs)
    finally:
        destroy()


def launch(
    fn: Callable,
    *args: Any,
    devices: int = torch.cuda.device_count(),
    backend: str = "nccl",
    timeout_minutes: int = 30,
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
    **kwargs: Any,
) -> None:
    """Spawn one process per GPU and run fn on each.

    fn is called identically on every rank; use is_main_process() to gate
    rank-0-only work. Handles process group init and teardown internally.

    Args:
        fn: Training function to run on each rank. Must be importable at
            module level so mp.spawn can pickle it.
        *args: Positional arguments forwarded to fn.
        devices: Number of GPUs. Defaults to all visible GPUs.
        backend: Distributed backend. "nccl" for GPU, "gloo" for CPU.
        timeout_minutes: Collective op timeout.
        master_addr: Rendezvous address. "127.0.0.1" for single-node.
        master_port: Rendezvous port.
        **kwargs: Keyword arguments forwarded to fn.
    """
    if devices == 1 or not torch.cuda.is_available():
        fn(*args, **kwargs)
        return

    log.info("launch: spawning %d workers on %s:%d", devices, master_addr, master_port)
    spawn(
        _worker,
        args=(
            devices,
            fn,
            args,
            kwargs,
            backend,
            timeout_minutes,
            master_addr,
            master_port,
        ),
        nprocs=devices,
        join=True,
    )
