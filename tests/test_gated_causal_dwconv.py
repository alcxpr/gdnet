import pytest
import torch
import torch.nn.functional as F
import triton

from gdnet.kernel.gated_causal_depthwise_conv import gated_causal_depthwise_conv
from gdnet.kernel.gated_causal_depthwise_conv.conv import causal_dwconv_fwd
from gdnet.kernel.gated_causal_depthwise_conv.gate_norm import (
    gate_stream_update_fwd,
    rmsnorm_fwd,
)

DEVICE = "cuda"


def _reference_fwd(x, fwd, side, R, W_conv, W1, b1, W_norm, W2, b2, eps=1e-6):
    B, T, d = x.shape
    k = W_conv.shape[1]
    x_t = x.float().transpose(1, 2)
    conv_out = F.conv1d(
        F.pad(x_t, (k - 1, 0)), W_conv.unsqueeze(1), groups=d
    ).transpose(1, 2)
    n_rows = B * T
    h = F.silu(F.linear(conv_out.reshape(n_rows, d), W1, b1))
    h_norm = h / h.pow(2).mean(-1, keepdim=True).add(eps).sqrt() * W_norm
    g = F.sigmoid(F.linear(h_norm, W2, b2)).view(B, T, d)
    s, R_f = side.float(), R.float()
    return (conv_out * g + s * R_f).bfloat16(), (
        conv_out * (1 - g) + s * (1 - R_f)
    ).bfloat16()


def _make(B=2, T=16, d=128, k=4, seed=0):
    torch.manual_seed(seed)
    return (
        torch.randn(B, T, d, dtype=torch.bfloat16, device=DEVICE),  # type: ignore
        torch.randn(B, T, d, dtype=torch.bfloat16, device=DEVICE),  # type: ignore
        torch.randn(B, T, d, dtype=torch.bfloat16, device=DEVICE),  # type: ignore
        F.sigmoid(torch.randn(B, T, d, device=DEVICE)).bfloat16(),
        torch.randn(d, k, device=DEVICE),
        torch.randn(d, d, device=DEVICE) * 0.1,
        torch.zeros(d, device=DEVICE),  # type: ignore
        torch.ones(d, device=DEVICE),  # type: ignore
        torch.randn(d, d, device=DEVICE) * 0.1,
        torch.zeros(d, device=DEVICE),  # type: ignore
    )


@pytest.mark.parametrize("B,T,d,k", [(2, 16, 128, 4), (2, 32, 256, 7)])
def test_causal_dwconv_fwd(B, T, d, k):
    x, _, _, _, W_conv, *_ = _make(B, T, d, k)
    x_dt = x.float().permute(0, 2, 1).contiguous()
    BLOCK_T = min(triton.next_power_of_2(T), 64)

    tri = causal_dwconv_fwd(x_dt, W_conv, T, k, BLOCK_T)  # type: ignore
    ref = F.conv1d(
        F.pad(x.float().transpose(1, 2), (k - 1, 0)), W_conv.unsqueeze(1), groups=d
    )

    assert torch.allclose(tri, ref, atol=1e-4), (  # type: ignore
        f"max diff {(tri - ref).abs().max():.2e}"
    )


@pytest.mark.parametrize("n_rows,d", [(64, 128), (512, 512)])
def test_rmsnorm_fwd(n_rows, d):
    H = torch.randn(n_rows, d, device=DEVICE)
    W_norm = torch.ones(d, device=DEVICE)  # type: ignore
    H_NORM, RSTD = rmsnorm_fwd(H, W_norm, 1e-6, BLOCK_D=d)
    rms = H.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
    assert torch.allclose(H_NORM, H / rms * W_norm, atol=1e-5)  # type: ignore
    assert torch.allclose(RSTD, (1.0 / rms).squeeze(-1), atol=1e-5)  # type: ignore


@pytest.mark.parametrize("n_rows,d", [(64, 128), (512, 512)])
def test_gate_stream_update_fwd(n_rows, d):
    g_pre = torch.randn(n_rows, d, device=DEVICE)
    conv_f = torch.randn(n_rows, d, device=DEVICE)
    side = torch.randn(n_rows, d, dtype=torch.bfloat16, device=DEVICE)  # type: ignore
    R = F.sigmoid(torch.randn(n_rows, d, device=DEVICE)).bfloat16()

    fwd_out, side_out = gate_stream_update_fwd(g_pre, conv_f, side, R, BLOCK_D=d)
    g = F.sigmoid(g_pre)
    s, Rf = side.float(), R.float()

    assert torch.allclose(  # type: ignore
        fwd_out.float(), (conv_f * g + s * Rf).bfloat16().float(), atol=1e-2
    )
    assert torch.allclose(  # type: ignore
        side_out.float(),
        (conv_f * (1 - g) + s * (1 - Rf)).bfloat16().float(),
        atol=1e-2,
    )


@pytest.mark.parametrize("B,T,d,k", [(2, 16, 128, 4), (4, 32, 256, 7)])
def test_fwd(B, T, d, k):
    args = _make(B, T, d, k)
    ref_fo, ref_so = _reference_fwd(*args)
    tri_fo, tri_so = gated_causal_depthwise_conv(*args)
    assert torch.allclose(ref_fo.float(), tri_fo.float(), rtol=1e-2, atol=1e-2)  # type: ignore
    assert torch.allclose(ref_so.float(), tri_so.float(), rtol=1e-2, atol=1e-2)  # type: ignore


@pytest.mark.parametrize("B,T,d,k", [(2, 16, 128, 4), (4, 32, 256, 7)])
def test_bwd(B, T, d, k):
    args = _make(B, T, d, k)
    x, fwd, side, R, W_conv, W1, b1, W_norm, W2, b2 = args

    def _grads(fn):
        params = [
            t.detach().float().requires_grad_(True)
            for t in (x, side, R, W_conv, W1, b1, W_norm, W2, b2)
        ]
        o1, o2 = fn(
            params[0].bfloat16(),
            fwd,
            params[1].bfloat16(),
            params[2].bfloat16(),
            *params[3:],
        )
        (o1.float().sum() + o2.float().sum()).backward()
        return [p.grad for p in params]

    ref_grads = _grads(_reference_fwd)
    tri_grads = _grads(gated_causal_depthwise_conv)

    names = ["x", "side", "R", "W_conv", "W1", "b1", "W_norm", "W2", "b2"]
    for name, rg, tg in zip(names, ref_grads, tri_grads):
        assert torch.allclose(rg.float(), tg.float(), atol=1e-2), (  # type: ignore
            f"d{name} max diff {(rg - tg).abs().max():.2e}"  # type: ignore
        )
