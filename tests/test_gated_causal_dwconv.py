import pytest
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
    rmsnorm_fwd,
)

DEVICE = "cuda"


def _reference_fwd(x, side, R, W_conv, W1, b1, W_norm, W2, b2, eps=1e-6):
    B, T, d = x.shape
    k = W_conv.shape[1]
    dtype = x.dtype
    # conv: fp32 accumulation, output in input dtype (matches Triton kernel)
    conv_out = F.conv1d(
        F.pad(x.float().transpose(1, 2), (k - 1, 0)), W_conv.unsqueeze(1), groups=d
    ).transpose(1, 2).to(dtype)
    n_rows = B * T
    conv_flat = conv_out.reshape(n_rows, d)
    # linear in input dtype (weights cast to match, same as gated_output)
    h = F.silu(F.linear(conv_flat, W1.to(dtype), b1.to(dtype)))
    # rmsnorm: fp32 variance, output in dtype (matches _rmsnorm_fwd_kernel)
    h_f = h.float()
    rstd = (h_f.pow(2).mean(-1, keepdim=True) + eps).rsqrt()
    h_norm = (h_f * rstd * W_norm.float()).to(dtype)
    # linear in dtype, sigmoid in fp32 (matches kernel)
    g = torch.sigmoid(F.linear(h_norm, W2.to(dtype), b2.to(dtype)).float()).view(B, T, d)
    s, R_f = side.float(), R.float()
    conv_out_f = conv_out.float()
    return (conv_out_f * g + s * R_f).to(dtype), (conv_out_f * (1 - g) + s * (1 - R_f)).to(dtype)


def _make(B=2, T=16, d=128, k=4, seed=0):
    torch.manual_seed(seed)
    return (
        torch.randn(B, T, d, dtype=torch.bfloat16, device=DEVICE),  # type: ignore  # x
        torch.randn(B, T, d, dtype=torch.bfloat16, device=DEVICE),  # type: ignore  # side
        F.sigmoid(torch.randn(B, T, d, device=DEVICE)).bfloat16(),               # R
        torch.randn(d, k, device=DEVICE),                                          # W_conv
        torch.randn(d, d, device=DEVICE) * 0.1,                                   # W1
        torch.zeros(d, device=DEVICE),  # type: ignore                            # b1
        torch.ones(d, device=DEVICE),  # type: ignore                             # W_norm
        torch.randn(d, d, device=DEVICE) * 0.1,                                   # W2
        torch.zeros(d, device=DEVICE),  # type: ignore                            # b2
    )


@pytest.mark.parametrize("B,T,d,k", [(2, 16, 128, 4), (2, 32, 256, 7)])
def test_causal_dwconv_fwd(B, T, d, k):
    x, _, _, W_conv, *_ = _make(B, T, d, k)
    BLOCK_T = min(triton.next_power_of_2(T), 64)

    tri = causal_dwconv_fwd(x.float(), W_conv, T, k, BLOCK_T)  # type: ignore
    ref = F.conv1d(
        F.pad(x.float().transpose(1, 2), (k - 1, 0)), W_conv.unsqueeze(1), groups=d
    ).transpose(1, 2)

    assert torch.allclose(tri, ref, atol=1e-4), (  # type: ignore
        f"max diff {(tri - ref).abs().max():.2e}"
    )


@pytest.mark.parametrize("B,T,d,k", [(2, 16, 128, 4), (2, 32, 256, 7)])
def test_causal_dwconv_fwd_sp(B, T, d, k):
    # Reference: causal conv over [halo | x], output the last T positions.
    # causal_dwconv_fwd_sp should match this without materialising the cat.
    x, _, _, W_conv, *_ = _make(B, T, d, k)
    x_f = x.float()
    halo = torch.randn(B, k - 1, d, device=DEVICE)
    BLOCK_T = min(triton.next_power_of_2(T), 64)

    tri = causal_dwconv_fwd_sp(x_f, halo, W_conv, T, k, BLOCK_T)  # type: ignore

    # F.conv1d on [halo | x] in (B, d, T) layout with no extra padding gives T outputs.
    padded = torch.cat([halo.transpose(1, 2), x_f.transpose(1, 2)], dim=2)  # (B, d, T+k-1)
    ref = F.conv1d(padded, W_conv.unsqueeze(1), groups=d).transpose(1, 2)  # (B, T, d)

    assert torch.allclose(tri, ref, atol=1e-4), (  # type: ignore
        f"max diff {(tri - ref).abs().max():.2e}"
    )


@pytest.mark.parametrize("B,T,d,k", [(2, 16, 128, 4), (2, 32, 256, 7)])
def test_causal_dwconv_bwd(B, T, d, k):
    x, _, _, W_conv, *_ = _make(B, T, d, k)
    x_f = x.float()
    BLOCK_T = min(triton.next_power_of_2(T), 64)

    x_ref = x_f.detach().requires_grad_(True)
    out_ref = F.conv1d(
        F.pad(x_ref.transpose(1, 2), (k - 1, 0)), W_conv.unsqueeze(1), groups=d
    ).transpose(1, 2)
    d_conv = torch.ones_like(out_ref)  # type: ignore
    out_ref.backward(d_conv)

    dX_tri, dW_tri = causal_dwconv_bwd(d_conv.contiguous(), x_f, W_conv, T, k, BLOCK_T)  # type: ignore

    assert torch.allclose(dX_tri, x_ref.grad, atol=1e-4), (  # type: ignore
        f"dX max diff {(dX_tri - x_ref.grad).abs().max():.2e}"  # type: ignore
    )


@pytest.mark.parametrize("B,T,d,k", [(2, 16, 128, 4), (2, 32, 256, 7)])
def test_causal_dwconv_bwd_sp(B, T, d, k):
    # Gradient correctness: bwd_sp(dX, x, halo) must match slicing the gradient
    # of the padded causal conv w.r.t. x and halo separately.
    x, _, _, W_conv, *_ = _make(B, T, d, k)
    x_f = x.float()
    halo = torch.randn(B, k - 1, d, device=DEVICE)
    BLOCK_T = min(triton.next_power_of_2(T), 64)

    # Reference gradients via autograd on the padded conv in (B, d, T) layout.
    x_ref = x_f.detach().requires_grad_(True)
    halo_ref = halo.detach().requires_grad_(True)
    padded = torch.cat([halo_ref.transpose(1, 2), x_ref.transpose(1, 2)], dim=2)
    out_ref = F.conv1d(padded, W_conv.unsqueeze(1), groups=d).transpose(1, 2)
    d_conv = torch.ones_like(out_ref)  # type: ignore
    out_ref.backward(d_conv)

    # Triton SP backward.
    dX_tri, dHalo_tri, dW_tri = causal_dwconv_bwd_sp(
        d_conv.contiguous(),
        x_f,
        halo,
        W_conv,
        T,
        k,
        BLOCK_T,  # type: ignore
    )

    assert torch.allclose(dX_tri, x_ref.grad, atol=1e-4), (  # type: ignore
        f"dX max diff {(dX_tri - x_ref.grad).abs().max():.2e}"  # type: ignore
    )
    assert torch.allclose(dHalo_tri, halo_ref.grad, atol=1e-4), (  # type: ignore
        f"dHalo max diff {(dHalo_tri - halo_ref.grad).abs().max():.2e}"  # type: ignore
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

    assert fwd_out.dtype == g_pre.dtype
    assert side_out.dtype == g_pre.dtype
    assert torch.allclose(  # type: ignore
        fwd_out.float(), conv_f * g + s * Rf, atol=1e-5
    )
    assert torch.allclose(  # type: ignore
        side_out.float(),
        conv_f * (1 - g) + s * (1 - Rf),
        atol=1e-5,
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
    x, side, R, W_conv, W1, b1, W_norm, W2, b2 = args

    def _grads(fn):
        params = [
            t.detach().float().requires_grad_(True)
            for t in (x, side, R, W_conv, W1, b1, W_norm, W2, b2)
        ]
        o1, o2 = fn(
            params[0].bfloat16(),
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
