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
        alpha_f = float(alpha.item())

        q_c = q.contiguous()
        gamma_c = gamma.contiguous()
        e_c = e.contiguous()
        btags_c = buffer_tags.contiguous()
        bvals_c = buffer_vals.contiguous()

        retrieved_c, w = fused_mem_read_fwd(
            q_c, gamma_c, e_c, btags_c, bvals_c, alpha_f
        )

        ctx.save_for_backward(q_c, gamma_c, e_c, btags_c, bvals_c, w, alpha)
        ctx.dtype = q.dtype
        return retrieved_c, w

    @staticmethod
    def backward(  # type: ignore
        ctx,
        d_retrieved_c: torch.Tensor,
        d_w: torch.Tensor,
    ) -> tuple:
        q_f, gamma_f, e_f, btags_f, bvals_f, w, alpha = ctx.saved_tensors
        alpha_f = float(alpha.item())

        d_q, d_gamma, d_alpha_per_b, d_sim, d_btags, d_bvals = fused_mem_read_bwd(
            d_retrieved_c.contiguous(),
            q_f,
            gamma_f,
            e_f,
            btags_f,
            bvals_f,
            w,
            alpha_f,
        )

        # d_e needs a reduction over B; one matmul, no atomics
        d_e = alpha_f * (d_sim.t() @ gamma_f.float())
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
        alpha: Learned log-scale scalar `()`, exp(rho).

    Returns:
        retrieved_c `(B, d_c)` and w `(B, n_slots)` retrieval weights, no grad on w.
    """
    return FusedMemReadFunction.apply(q, gamma, e, buffer_tags, buffer_vals, alpha)
