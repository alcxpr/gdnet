from __future__ import annotations

import torch

from .kernel import fused_mem_read_bwd, fused_mem_read_fwd


class FusedMemReadFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        gamma: torch.Tensor,
        e: torch.Tensor,
        buffer_tags: torch.Tensor,
        buffer_vals: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = [t.contiguous() for t in (q, gamma, e, buffer_tags, buffer_vals)]
        alpha_c = alpha.contiguous()
        retrieved_c, w = fused_mem_read_fwd(*tensors, alpha_c)
        ctx.save_for_backward(*tensors, w, alpha_c)
        ctx.dtype = q.dtype
        return retrieved_c, w

    @staticmethod
    def backward(  # type: ignore
        ctx,
        d_retrieved_c: torch.Tensor,
        d_w: torch.Tensor,
    ) -> tuple:
        q_f, gamma_f, e_f, btags_f, bvals_f, w, alpha = ctx.saved_tensors

        d_q, d_gamma, d_alpha_per_b, d_sim, d_btags, d_bvals = fused_mem_read_bwd(
            d_retrieved_c.contiguous(),
            q_f,
            gamma_f,
            e_f,
            btags_f,
            bvals_f,
            w,
            alpha,
        )

        d_e = alpha.to(d_sim.dtype) * (d_sim.t() @ gamma_f.to(d_sim.dtype))
        d_alpha = d_alpha_per_b.sum().reshape_as(alpha)

        dtype = ctx.dtype
        return (
            d_q,
            d_gamma,
            d_e.to(dtype),
            d_btags,
            d_bvals,
            d_alpha,
        )


@torch.compiler.disable
def fused_mem_read(
    q: torch.Tensor,
    gamma: torch.Tensor,
    e: torch.Tensor,
    buffer_tags: torch.Tensor,
    buffer_vals: torch.Tensor,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused sim + softmax + retrieval aggregation for the CAM read operation.

    Args:
        q: Content query `(B, d_sig)`.
        gamma: Position query `(B, d_sig)`, sigmoid(W_pos(q)).
        e: Learned slot embeddings `(n_slots, d_sig)`.
        buffer_tags: Stored content tags `(B, n_slots, d_sig)`.
        buffer_vals: Stored compressed values `(B, n_slots, d_c)`.
        alpha: Learned scalar `()` used as position-term weight.

    Returns:
        retrieved_c `(B, d_c)` and w `(B, n_slots)` retrieval weights, no grad on w.
    """
    return FusedMemReadFunction.apply(q, gamma, e, buffer_tags, buffer_vals, alpha)
