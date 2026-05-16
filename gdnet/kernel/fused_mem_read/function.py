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
        dtype = q.dtype
        alpha_f = float(alpha.item())

        q_f = q.float().contiguous()
        gamma_f = gamma.float().contiguous()
        e_f = e.float().contiguous()
        btags_f = buffer_tags.float().contiguous()
        bvals_f = buffer_vals.float().contiguous()

        retrieved_c, w = fused_mem_read_fwd(
            q_f, gamma_f, e_f, btags_f, bvals_f, alpha_f
        )

        ctx.save_for_backward(q_f, gamma_f, e_f, btags_f, bvals_f, w, alpha)
        ctx.dtype = dtype
        return retrieved_c.to(dtype), w

    @staticmethod
    def backward(  # type: ignore
        ctx,
        d_retrieved_c: torch.Tensor,
        d_w: torch.Tensor,
    ) -> tuple:
        q_f, gamma_f, e_f, btags_f, bvals_f, w, alpha = ctx.saved_tensors
        alpha_f = float(alpha.item())

        d_q, d_gamma, d_alpha_per_b, d_sim, d_btags, d_bvals = fused_mem_read_bwd(
            d_retrieved_c.float().contiguous(),
            q_f,
            gamma_f,
            e_f,
            btags_f,
            bvals_f,
            w,
            alpha_f,
        )

        # d_e needs a reduction over B; one matmul, no atomics
        d_e = alpha_f * (d_sim.t() @ gamma_f)
        d_alpha = d_alpha_per_b.sum().reshape_as(alpha)

        dtype = ctx.dtype
        return (
            d_q.to(dtype),
            d_gamma.to(dtype),
            d_e.to(dtype),
            d_btags.to(dtype),
            d_bvals.to(dtype),
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
        q: (B, d_sig) content query (W_tag output on current chunk)
        gamma: (B, d_sig) position query (sigmoid(W_pos @ q))
        e: (n_slots, d_sig) learned slot embeddings (W_slot output)
        buffer_tags: (B, n_slots, d_sig) stored content tags
        buffer_vals: (B, n_slots, d_c) stored compressed values
        alpha: () learned log-scale scalar (exp(rho))

    Returns:
        retrieved_c: (B, d_c)
        w: (B, n_slots) retrieval weights (for logging; no grad flows through)
    """
    return FusedMemReadFunction.apply(q, gamma, e, buffer_tags, buffer_vals, alpha)
