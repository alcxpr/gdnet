"""Profile the fused_mem_read kernel in isolation (fwd + bwd).

Benchmarks wall-clock time and torch.profiler CUDA breakdown for the
fused Triton kernel vs the equivalent einsum reference.

Usage (wall-clock only):
    python scripts/profile_fused_mem_read.py

Usage (ncu -- targeted, fast):
    sudo ncu --set default -k _fused_mem_read_fwd_kernel \
        python scripts/profile_fused_mem_read.py --ncu

    sudo ncu --set default -k _fused_mem_read_bwd_kernel \
        python scripts/profile_fused_mem_read.py --ncu

    # Roofline metrics only (much faster than --set full):
    sudo ncu \
        --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_bytes_lookup_hit.sum,dram__bytes.sum \
        -k _fused_mem_read_fwd_kernel \
        python scripts/profile_fused_mem_read.py --ncu
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import torch.profiler

from gdnet.kernel.fused_mem_read import fused_mem_read

DEVICE = torch.device("cuda")  # type: ignore

B = 128
N_SLOTS = 8
D_SIG = 8
D_C = 8

B_LARGE = 512
N_SLOTS_LARGE = 32
D_SIG_LARGE = 64
D_C_LARGE = 64

B_XL = 512
N_SLOTS_XL = 32
D_SIG_XL = 512
D_C_XL = 512


def make_inputs(b: int, n_slots: int, d_sig: int, d_c: int, dtype=torch.float32):  # type: ignore
    q = torch.randn(b, d_sig, device=DEVICE, dtype=dtype, requires_grad=True)
    gamma = torch.sigmoid(  # type: ignore
        torch.randn(b, d_sig, device=DEVICE, dtype=dtype)
    ).requires_grad_(True)
    e = torch.randn(n_slots, d_sig, device=DEVICE, dtype=dtype, requires_grad=True)
    btags = torch.randn(
        b, n_slots, d_sig, device=DEVICE, dtype=dtype, requires_grad=True
    )
    bvals = torch.randn(b, n_slots, d_c, device=DEVICE, dtype=dtype, requires_grad=True)
    alpha = torch.tensor([1.0], device=DEVICE, dtype=dtype, requires_grad=True)  # type: ignore
    return q, gamma, e, btags, bvals, alpha


def ref_fwd(q, gamma, e, btags, bvals, alpha):
    sim_content = torch.einsum("bd,bsd->bs", q, btags)  # type: ignore
    sim_pos = torch.einsum("bd,sd->bs", gamma, e)  # type: ignore
    w = F.softmax(sim_content + alpha * sim_pos, dim=-1)
    retrieved_c = torch.einsum("bs,bsd->bd", w, bvals)  # type: ignore
    return retrieved_c, w


def step_fused(inputs):
    q, gamma, e, btags, bvals, alpha = inputs
    out, w = fused_mem_read(q, gamma, e, btags, bvals, alpha)
    out.sum().backward()


def step_ref(inputs):
    q, gamma, e, btags, bvals, alpha = inputs
    out, w = ref_fwd(q, gamma, e, btags, bvals, alpha)
    out.sum().backward()


def bench(fn, inputs, n_warmup: int = 10, n_steps: int = 50) -> float:
    for _ in range(n_warmup):
        fn(inputs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        fn(inputs)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_steps * 1000


def run_profile(fn, inputs, n_steps: int = 5):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(n_steps):
            fn(inputs)
    return prof


def main():
    ncu_mode = "--ncu" in sys.argv

    if ncu_mode:
        cfg = (B_XL, N_SLOTS_XL, D_SIG_XL, D_C_XL) if "--ncu-xl" in sys.argv else (B, N_SLOTS, D_SIG, D_C)
        inputs = make_inputs(*cfg)
        for _ in range(3):
            step_fused(inputs)
        torch.cuda.synchronize()
        return

    configs = [
        ("small  B=128 S=8  dsig=8  dc=8", B, N_SLOTS, D_SIG, D_C),
        ("large  B=512 S=32 dsig=64  dc=64", B_LARGE, N_SLOTS_LARGE, D_SIG_LARGE, D_C_LARGE),
        ("xl     B=512 S=32 dsig=512 dc=512", B_XL, N_SLOTS_XL, D_SIG_XL, D_C_XL),
    ]

    for label, b, n_slots, d_sig, d_c in configs:
        print(f"\n--- {label} ---")
        inputs_fused = make_inputs(b, n_slots, d_sig, d_c)
        inputs_ref = make_inputs(b, n_slots, d_sig, d_c)
        ms_fused = bench(step_fused, inputs_fused)
        ms_ref = bench(step_ref, inputs_ref)
        print(f"  fused:  {ms_fused:.3f} ms/iter")
        print(f"  ref:    {ms_ref:.3f} ms/iter")
        print(f"  speedup: {ms_ref / ms_fused:.2f}x")

    print("\n--- torch.profiler (small config, fused) ---")
    inputs_fused = make_inputs(B, N_SLOTS, D_SIG, D_C)
    prof = run_profile(step_fused, inputs_fused)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    print("\n--- torch.profiler (small config, ref) ---")
    inputs_ref = make_inputs(B, N_SLOTS, D_SIG, D_C)
    prof = run_profile(step_ref, inputs_ref)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


if __name__ == "__main__":
    main()
