import pytest
import torch

from gdnet.layer import GDLayer

DEVICE = "cuda"
DTYPE = torch.bfloat16


def _make_layer(d: int, k: int) -> GDLayer:
    return GDLayer(d=d, size=k).to(DEVICE).to(DTYPE)


def _make_inputs(B: int, T: int, d: int, seed: int = 0):
    torch.manual_seed(seed)
    fwd = torch.randn(B, T, d, device=DEVICE, dtype=DTYPE)
    side = torch.randn(B, T, d, device=DEVICE, dtype=DTYPE)
    return fwd, side


@pytest.mark.parametrize("B,T,d,k", [(2, 16, 128, 4), (2, 32, 256, 7)])
def test_fwd_step_shape_dtype(B, T, d, k):
    layer = _make_layer(d, k)
    fwd, side = _make_inputs(B, T, d)
    fo, so = layer.fwd_step(fwd, side)
    assert fo.shape == (B, T, d)
    assert so.shape == (B, T, d)
    assert fo.dtype == DTYPE
    assert so.dtype == DTYPE


@pytest.mark.parametrize("B,T,d,k", [(2, 16, 128, 4), (2, 32, 256, 7)])
def test_bwd_step_shape_dtype(B, T, d, k):
    layer = _make_layer(d, k)
    fwd, side = _make_inputs(B, T, d)
    fo, so = layer.bwd_step(fwd, side)
    assert fo.shape == (B, T, d)
    assert so.shape == (B, T, d)
    assert fo.dtype == DTYPE
    assert so.dtype == DTYPE


@pytest.mark.parametrize("B,T,d,k", [(2, 16, 128, 4)])
def test_return_gate_shape_dtype(B, T, d, k):
    layer = _make_layer(d, k)
    fwd, side = _make_inputs(B, T, d)
    fo, so, gate = layer.fwd_step(fwd, side, return_gate=True)
    assert fo.shape == (B, T, d)
    assert so.shape == (B, T, d)
    assert gate.shape == (B, T, d)
    assert fo.dtype == DTYPE
    assert so.dtype == DTYPE


@pytest.mark.parametrize("step_name,active_prefixes", [
    ("fwd_step", ("conv_fwd", "gf_", "rf_")),
    ("bwd_step", ("conv_bwd", "gb_", "rb_")),
])
def test_gradients_flow(step_name, active_prefixes):
    B, T, d, k = 2, 16, 128, 4
    layer = _make_layer(d, k)
    fwd, side = _make_inputs(B, T, d)
    fwd = fwd.requires_grad_(True)
    side = side.requires_grad_(True)

    fo, so = getattr(layer, step_name)(fwd, side)
    (fo.float().sum() + so.float().sum()).backward()

    for name, param in layer.named_parameters():
        if not any(name.startswith(p) for p in active_prefixes):
            continue
        assert param.grad is not None, f"{name} has no gradient"

    assert fwd.grad is not None and fwd.grad.abs().sum() > 0
    assert side.grad is not None and side.grad.abs().sum() > 0


def _ref_step(layer: GDLayer, fwd: torch.Tensor, side: torch.Tensor, prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, d = fwd.shape
    k = getattr(layer, f"conv_{prefix}").size
    W_conv = getattr(layer, f"conv_{prefix}").conv.weight.squeeze(1)
    dtype = fwd.dtype
    conv_out = torch.nn.functional.conv1d(
        torch.nn.functional.pad(fwd.float().transpose(1, 2), (k - 1, 0)),
        W_conv.unsqueeze(1).float(),
        groups=d,
    ).transpose(1, 2).to(dtype)
    conv_flat = conv_out.reshape(B * T, d)

    rp = "rf" if prefix == "fwd" else "rb"
    R = torch.sigmoid(
        getattr(layer, f"{rp}_norm")(
            getattr(layer, f"{rp}_W2")(
                torch.nn.functional.silu(
                    getattr(layer, f"{rp}_W1")(torch.cat([side, conv_out], dim=-1))
                )
            )
        )
    )

    gp = "gf" if prefix == "fwd" else "gb"
    W1 = getattr(layer, f"{gp}_W1").weight
    b1 = getattr(layer, f"{gp}_W1").bias
    W_norm = getattr(layer, f"{gp}_norm").weight
    W2 = getattr(layer, f"{gp}_W2").weight
    b2 = getattr(layer, f"{gp}_W2").bias

    H = torch.nn.functional.silu(torch.nn.functional.linear(conv_flat, W1, b1))
    h_f = H.float()
    rstd = (h_f.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt()
    h_norm = (h_f * rstd * W_norm.float()).to(dtype)
    g = torch.sigmoid(torch.nn.functional.linear(h_norm, W2, b2).float())
    R_flat = R.reshape(B * T, d).float()
    conv_f = conv_flat.float()
    side_f = side.reshape(B * T, d).float()
    fo = (conv_f * g + side_f * R_flat).to(dtype).view(B, T, d)
    so = (conv_f * (1 - g) + side_f * (1 - R_flat)).to(dtype).view(B, T, d)
    return fo, so


@pytest.mark.parametrize("step_name,conv_prefix", [
    ("fwd_step", "fwd"),
    ("bwd_step", "bwd"),
])
def test_matches_reference(step_name, conv_prefix):
    B, T, d, k = 2, 16, 128, 4
    layer = _make_layer(d, k)
    fwd, side = _make_inputs(B, T, d)
    fo_tri, so_tri = getattr(layer, step_name)(fwd, side)
    fo_ref, so_ref = _ref_step(layer, fwd, side, conv_prefix)
    assert torch.allclose(fo_tri.float(), fo_ref.float(), atol=1e-2), \
        f"fwd_out max diff {(fo_tri.float() - fo_ref.float()).abs().max():.4f}"
    assert torch.allclose(so_tri.float(), so_ref.float(), atol=1e-2), \
        f"side_out max diff {(so_tri.float() - so_ref.float()).abs().max():.4f}"


@pytest.mark.parametrize("B,T,d,k", [(2, 16, 128, 4)])
def test_return_gate_outputs_match(B, T, d, k):
    layer = _make_layer(d, k)
    fwd, side = _make_inputs(B, T, d)
    fo1, so1 = layer.fwd_step(fwd, side, return_gate=False)
    fo2, so2, _ = layer.fwd_step(fwd, side, return_gate=True)
    assert torch.equal(fo1, fo2)
    assert torch.equal(so1, so2)


@pytest.mark.parametrize("B,T,d,k", [(2, 16, 128, 4)])
def test_fwd_bwd_step_differ(B, T, d, k):
    layer = _make_layer(d, k)
    fwd, side = _make_inputs(B, T, d)
    fo_f, so_f = layer.fwd_step(fwd, side)
    fo_b, so_b = layer.bwd_step(fwd, side)
    assert not torch.equal(fo_f, fo_b)
