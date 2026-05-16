import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import triton

from gdnet.kernel.gated_causal_depthwise_conv.conv import causal_dwconv_fwd
from gdnet.kernel.gated_causal_depthwise_conv.gate_norm import gate_w2_bwd, rmsnorm_fwd

B, T, d, k = 4, 512, 512, 7
n_rows = B * T
BLOCK_T = min(triton.next_power_of_2(T), 64)
BLOCK_D = d

torch.manual_seed(0)
x = torch.randn(B, T, d, dtype=torch.bfloat16, device="cuda")  # type: ignore
side = torch.randn(B, T, d, dtype=torch.bfloat16, device="cuda")
R = torch.sigmoid(torch.randn(B, T, d, device="cuda")).bfloat16()  # type: ignore
W_conv = torch.randn(d, k, device="cuda")
W1 = torch.randn(d, d, device="cuda")
b1 = torch.zeros(d, device="cuda")  # type: ignore
W_norm = torch.ones(d, device="cuda")  # type: ignore
W2 = torch.randn(d, d, device="cuda")

x_dt = x.float().permute(0, 2, 1).contiguous()
conv_out_dt = causal_dwconv_fwd(x_dt, W_conv, T, k, BLOCK_T)  # type: ignore
conv_flat = conv_out_dt.permute(0, 2, 1).contiguous().view(n_rows, d)
side_flat = side.contiguous().view(n_rows, d)
R_flat = R.contiguous().view(n_rows, d)
H = F.silu(F.linear(conv_flat, W1, b1))
H_NORM, RSTD = rmsnorm_fwd(H, W_norm, 1e-6, BLOCK_D)
g_pre = F.linear(H_NORM, W2, torch.full((d,), 2.0, device="cuda"))  # type: ignore
d_fwd_f = torch.randn(n_rows, d, device="cuda")
d_side_f = torch.randn(n_rows, d, device="cuda")

for _ in range(10):
    gate_w2_bwd(
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
    )
torch.cuda.synchronize()

for _ in range(3):
    gate_w2_bwd(
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
    )
torch.cuda.synchronize()
