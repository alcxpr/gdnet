"""Minimal two-rank SymmetricMemory smoke test.

torchrun --nproc_per_node=2 scripts/test_symm_mem.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mod

from gdnet.utils.distributed import destroy, init_distributed


def main() -> None:
    init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    print(f"[rank {rank}] dist ok, device={device}", flush=True)

    B, km1, d = 4, 8, 2048
    flat = B * km1 * d
    dtype = torch.bfloat16
    sp_group = dist.group.WORLD

    print(f"[rank {rank}] allocating symm fwd buffer flat={flat}", flush=True)
    fwd_t = symm_mod.empty(flat, dtype=dtype, device=device)
    print(f"[rank {rank}] fwd alloc ok, calling rendezvous", flush=True)
    fwd_hdl = symm_mod.rendezvous(fwd_t, sp_group)
    print(f"[rank {rank}] rendezvous ok", flush=True)

    dist.barrier()

    if rank == 0:
        edge = torch.ones(B, km1, d, dtype=dtype, device=device)
        peer_buf = fwd_hdl.get_buffer(1, (B, km1, d), dtype)
        print(f"[rank {rank}] get_buffer ok, copying edge to rank 1", flush=True)
        peer_buf.copy_(edge)
        print(f"[rank {rank}] copy ok, put_signal to rank 1", flush=True)
        fwd_hdl.put_signal(1)
        print(f"[rank {rank}] put_signal ok", flush=True)
    else:
        print(f"[rank {rank}] waiting for signal from rank 0", flush=True)
        fwd_hdl.wait_signal(0)
        halo = fwd_t.view(B, km1, d).clone()
        print(f"[rank {rank}] wait_signal ok, halo sum={halo.sum().item():.1f}", flush=True)
        expected = float(B * km1 * d)
        assert abs(halo.sum().item() - expected) < 1.0, f"data mismatch: {halo.sum().item()} vs {expected}"
        print(f"[rank {rank}] data check PASSED", flush=True)

    dist.barrier()
    print(f"[rank {rank}] all done", flush=True)
    destroy()


if __name__ == "__main__":
    main()
