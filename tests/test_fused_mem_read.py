import pytest
import torch
import torch.nn.functional as F

from gdnet.kernel.fused_mem_read import fused_mem_read

DEVICE = "cuda"


def _ref_fwd(q, gamma, e, btags, bvals, alpha):
    sim_content = torch.einsum("bd,bsd->bs", q, btags)  # type: ignore
    sim_pos = torch.einsum("bd,sd->bs", gamma, e)  # type: ignore
    w = F.softmax(sim_content + alpha * sim_pos, dim=-1)
    retrieved_c = torch.einsum("bs,bsd->bd", w, bvals)  # type: ignore
    return retrieved_c, w


def _make(B, n_slots, d_sig, d_c, seed=0):
    torch.manual_seed(seed)
    kwargs = dict(device=DEVICE, dtype=torch.float32)  # type: ignore
    q = torch.randn(B, d_sig, **kwargs, requires_grad=True)  # type: ignore
    gamma = torch.sigmoid(torch.randn(B, d_sig, **kwargs)).requires_grad_(True)  # type: ignore
    e = torch.randn(n_slots, d_sig, **kwargs, requires_grad=True)  # type: ignore
    btags = torch.randn(B, n_slots, d_sig, **kwargs, requires_grad=True)  # type: ignore
    bvals = torch.randn(B, n_slots, d_c, **kwargs, requires_grad=True)  # type: ignore
    alpha = torch.tensor([1.0], **kwargs, requires_grad=True)  # type: ignore
    return q, gamma, e, btags, bvals, alpha


def _clone_inputs(inputs):
    return tuple(t.detach().clone().requires_grad_(t.requires_grad) for t in inputs)


@pytest.mark.parametrize(
    "B,n_slots,d_sig,d_c",
    [
        (128, 8, 8, 8),
        (64, 16, 16, 32),
        (32, 32, 32, 64),
    ],
)
def test_fwd(B, n_slots, d_sig, d_c):
    inputs = _make(B, n_slots, d_sig, d_c)
    out_tri, w_tri = fused_mem_read(*inputs)
    out_ref, w_ref = _ref_fwd(*_clone_inputs(inputs))

    assert torch.allclose(out_tri.float(), out_ref.float(), atol=1e-4), (  # type: ignore
        f"retrieved_c max diff {(out_tri.float() - out_ref.float()).abs().max():.2e}"
    )
    assert torch.allclose(w_tri.float(), w_ref.float(), atol=1e-5), (  # type: ignore
        f"w max diff {(w_tri.float() - w_ref.float()).abs().max():.2e}"
    )


@pytest.mark.parametrize(
    "B,n_slots,d_sig,d_c",
    [
        (128, 8, 8, 8),
        (64, 16, 16, 32),
        (32, 32, 32, 64),
    ],
)
def test_bwd(B, n_slots, d_sig, d_c):
    inputs_tri = _make(B, n_slots, d_sig, d_c)
    inputs_ref = _clone_inputs(inputs_tri)

    out_tri, _ = fused_mem_read(*inputs_tri)
    out_tri.sum().backward()

    out_ref, _ = _ref_fwd(*inputs_ref)
    out_ref.sum().backward()

    names = ["q", "gamma", "e", "btags", "bvals", "alpha"]
    # d_alpha sums over B*n_slots; float32 accumulation sets the tolerance floor
    atol_alpha = max(1e-4, 1e-7 * B * n_slots)
    atols = {name: 1e-4 for name in names}
    atols["alpha"] = atol_alpha

    for name, t_tri, t_ref in zip(names, inputs_tri, inputs_ref):
        assert torch.allclose(  # type: ignore
            t_tri.grad.float(),  # type: ignore
            t_ref.grad.float(),  # type: ignore
            atol=atols[name],  # type: ignore
        ), f"d_{name} max diff {(t_tri.grad - t_ref.grad).abs().max():.2e}"  # type: ignore
