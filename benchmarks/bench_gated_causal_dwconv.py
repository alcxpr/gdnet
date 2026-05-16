import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyperf
import torch
import torch.nn.functional as F
import triton

from gdnet.kernel.gated_causal_depthwise_conv import gated_causal_depthwise_conv
from gdnet.kernel.gated_causal_depthwise_conv.conv import (
    causal_dwconv_bwd,
    causal_dwconv_bwd_sp,
    causal_dwconv_fwd,
    causal_dwconv_fwd_sp,
)
from gdnet.kernel.gated_causal_depthwise_conv.gate_norm import (
    gate_stream_update_fwd,
    gate_w2_bwd,
    rmsnorm_bwd,
    rmsnorm_fwd,
)

CONFIGS = [
    (2, 512, 512, 7),
    (4, 512, 512, 7),
    (8, 512, 512, 7),
    (4, 256, 1024, 7),
    (4, 512, 1024, 7),
    (8, 512, 1024, 7),
]


def _make(B, T, d, k):
    return (
        torch.randn(B, T, d, dtype=torch.bfloat16, device="cuda"),  # type: ignore  # x
        torch.randn(B, T, d, dtype=torch.bfloat16, device="cuda"),  # type: ignore  # side
        torch.sigmoid(torch.randn(B, T, d, device="cuda")).bfloat16(),  # type: ignore  # R
        torch.randn(d, k, device="cuda"),  # W_conv
        torch.randn(d, d, device="cuda"),  # W1
        torch.zeros(d, device="cuda"),  # type: ignore                                # b1
        torch.ones(d, device="cuda"),  # type: ignore                                 # W_norm
        torch.randn(d, d, device="cuda"),  # W2
        torch.full((d,), 2.0, device="cuda"),  # type: ignore                         # b2
    )


def _time(loops, fn):
    for _ in range(10):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(loops):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000


def _bwd_fn(x, side, R, W_conv, W1, b1, W_norm, W2, b2):
    x_ = x.detach().requires_grad_(True)
    side_ = side.detach().requires_grad_(True)
    R_ = R.detach().requires_grad_(True)
    W_conv_ = W_conv.detach().requires_grad_(True)
    W1_ = W1.detach().requires_grad_(True)
    b1_ = b1.detach().requires_grad_(True)
    W_norm_ = W_norm.detach().requires_grad_(True)
    W2_ = W2.detach().requires_grad_(True)
    b2_ = b2.detach().requires_grad_(True)
    fo, so = gated_causal_depthwise_conv(
        x_, side_, R_, W_conv_, W1_, b1_, W_norm_, W2_, b2_
    )
    (fo.sum() + so.sum()).backward()


def _register(runner, B, T, d, k):
    tag = f"B{B}_T{T}_d{d}_k{k}"
    x, side, R, W_conv, W1, b1, W_norm, W2, b2 = _make(B, T, d, k)  # type: ignore
    n_rows = B * T
    BLOCK_T = min(triton.next_power_of_2(T), 64)
    BLOCK_D = d

    x_dt = x.float().permute(0, 2, 1).contiguous()
    halo_dt = torch.randn(B, d, k - 1, device="cuda")
    conv_out_dt = causal_dwconv_fwd(x_dt, W_conv, T, k, BLOCK_T)  # type: ignore
    conv_flat = conv_out_dt.permute(0, 2, 1).contiguous().view(n_rows, d)
    side_flat = side.contiguous().view(n_rows, d)
    R_flat = R.contiguous().view(n_rows, d)
    H = F.silu(F.linear(conv_flat, W1, b1))
    H_NORM, RSTD = rmsnorm_fwd(H, W_norm, 1e-6, BLOCK_D)
    g_pre = F.linear(H_NORM, W2, b2)
    d_fwd_f = torch.randn(n_rows, d, device="cuda")
    d_side_f = torch.randn(n_rows, d, device="cuda")
    d_h_norm = torch.randn(n_rows, d, device="cuda")
    d_conv_dt = torch.randn(B, d, T, device="cuda")

    runner.bench_time_func(
        f"fwd_causal_dwconv_{tag}",
        lambda loops: _time(
            loops,
            lambda: causal_dwconv_fwd(x_dt, W_conv, T, k, BLOCK_T),  # type: ignore
        ),
    )
    runner.bench_time_func(
        f"fwd_causal_dwconv_sp_{tag}",
        lambda loops: _time(
            loops,
            lambda: causal_dwconv_fwd_sp(x_dt, halo_dt, W_conv, T, k, BLOCK_T),  # type: ignore
        ),
    )
    runner.bench_time_func(
        f"fwd_gate_update_{tag}",
        lambda loops: _time(
            loops,
            lambda: gate_stream_update_fwd(
                g_pre, conv_flat, side_flat, R_flat, BLOCK_D
            ),
        ),
    )
    runner.bench_time_func(
        f"fwd_e2e_{tag}",
        lambda loops: _time(
            loops,
            lambda: gated_causal_depthwise_conv(
                x, side, R, W_conv, W1, b1, W_norm, W2, b2
            ),
        ),
    )
    runner.bench_time_func(
        f"bwd_e2e_{tag}",
        lambda loops: _time(
            loops, lambda: _bwd_fn(x, side, R, W_conv, W1, b1, W_norm, W2, b2)
        ),
    )
    runner.bench_time_func(
        f"bwd_causal_dwconv_{tag}",
        lambda loops: _time(
            loops,
            lambda: causal_dwconv_bwd(d_conv_dt, x_dt, W_conv, T, k, BLOCK_T),  # type: ignore
        ),
    )
    runner.bench_time_func(
        f"bwd_causal_dwconv_sp_{tag}",
        lambda loops: _time(
            loops,
            lambda: causal_dwconv_bwd_sp(
                d_conv_dt,
                x_dt,
                halo_dt,
                W_conv,
                T,
                k,
                BLOCK_T,  # type: ignore
            ),
        ),
    )
    runner.bench_time_func(
        f"bwd_gate_w2_{tag}",
        lambda loops: _time(
            loops,
            lambda: gate_w2_bwd(
                d_fwd_f,
                d_side_f,
                g_pre,
                conv_flat,
                side_flat,
                R_flat,
                H,
                RSTD,
                W_norm,
                W2,
                BLOCK_D,
            ),
        ),
    )
    runner.bench_time_func(
        f"bwd_rmsnorm_{tag}",
        lambda loops: _time(
            loops, lambda: rmsnorm_bwd(d_h_norm, H, RSTD, W_norm, BLOCK_D)
        ),
    )


runner = pyperf.Runner()
for B, T, d, k in CONFIGS:
    _register(runner, B, T, d, k)
