import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernel.fused_mem_read import fused_mem_read


class Memory(nn.Module):
    """Content-Addressed Memory with input-conditioned position gating.

    Retrieval combines content similarity and a learned positional prior:

        q           = W_tag(fwd[:, -1, :])
        gamma       = sigmoid(W_pos(q))           - input-conditioned position gate
        e           = W_slot(slot_ids)            - learned slot embeddings
        sim_content = einsum("bd,bsd->bs", q, buffer_tags)
        sim_pos     = einsum("bd,sd->bs", gamma, e)
        w           = softmax(sim_content + exp(rho) * sim_pos)
        retrieved_c = einsum("bs,bsd->bd", w, buffer_vals)

    W_pos is zero-initialized so gamma starts at 0.5, without any initial recency bias.
    W_slot uses geometric decay init (decay=0.85) for a soft recency prior that
    alpha = exp(rho) can scale up or down.

    Args:
        d: Hidden dimension of the main network.
        d_c: Compressed value dimension.
        d_sig: Tag/query dimension.
        n_slots: Number of buffer slots.
        chunk_size: Sequence chunk size used during training.
    """

    def __init__(
        self,
        d: int,
        d_c: int,
        d_sig: int,
        n_slots: int,
        chunk_size: int,
    ) -> None:
        super().__init__()
        self.d = d
        self.d_c = d_c
        self.d_sig = d_sig
        self.n_slots = n_slots
        self.chunk_size = chunk_size

        self.W_tag = nn.Linear(d, d_sig, bias=False)
        self.W_pos = nn.Linear(d_sig, d_sig, bias=False)
        nn.init.zeros_(self.W_pos.weight)
        self.W_slot = nn.Embedding(n_slots, d_sig)
        decay = 0.85
        with torch.no_grad():
            for s in range(n_slots):
                self.W_slot.weight[s] *= decay**s

        self.rho = nn.Parameter(torch.zeros(1))  # type: ignore[reportPrivateImportUsage]

        self.W_c = nn.Linear(d, d_c, bias=False)
        self.W_d = nn.Linear(d_c, d, bias=False)
        self.W_up = nn.Linear(d_c, d, bias=False)
        self.W_r1 = nn.utils.spectral_norm(nn.Linear(d * 3, d))
        self.W_r2 = nn.Linear(d, d)
        self.r_norm = nn.RMSNorm(d)
        nn.init.normal_(self.W_r2.weight, std=0.01)
        nn.init.constant_(self.W_r2.bias, -2.0)

    def write(
        self,
        tag_input: torch.Tensor,
        val_input: torch.Tensor,
        buffer_tags: torch.Tensor,
        buffer_vals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write a chunk summary to the buffer.

        Rolls the buffer forward and writes the new tag and value to slot 0.
        Called at chunk boundaries.

        Args:
            tag_input: Last forward token of the chunk `(B, d)`.
            val_input: Mean of the side stream over the chunk `(B, d)`.
            buffer_tags: Current tag buffer `(B, n_slots, d_sig)`.
            buffer_vals: Current value buffer `(B, n_slots, d_c)`.

        Returns:
            Updated `(buffer_tags, buffer_vals)`.
        """
        tag = self.W_tag(tag_input)
        val = self.W_c(val_input)
        buffer_tags = torch.roll(buffer_tags, 1, dims=1)  # type: ignore[reportPrivateImportUsage]
        buffer_vals = torch.roll(buffer_vals, 1, dims=1)  # type: ignore[reportPrivateImportUsage]
        buffer_tags[:, 0, :] = tag
        buffer_vals[:, 0, :] = val.detach()
        return buffer_tags, buffer_vals

    def read(
        self,
        fwd: torch.Tensor,
        side: torch.Tensor,
        buffer_tags: torch.Tensor,
        buffer_vals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Query the buffer and blend retrieved content into the forward stream.

        Args:
            fwd: Forward stream `(B, T, d)`.
            side: Side stream `(B, T, d)`.
            buffer_tags: Tag buffer `(B, n_slots, d_sig)`.
            buffer_vals: Value buffer `(B, n_slots, d_c)`.

        Returns:
            Updated forward stream `(B, T, d)` and retrieval weights `(B, n_slots)`, detached.
        """
        q = self.W_tag(fwd[:, -1, :])
        gamma = F.sigmoid(self.W_pos(q))
        slot_ids = torch.arange(self.n_slots, device=fwd.device)  # type: ignore[reportPrivateImportUsage]
        e = self.W_slot(slot_ids)
        retrieved_c, w = fused_mem_read(q, gamma, e, buffer_tags, buffer_vals, self.rho)
        retrieved = self.W_up(retrieved_c)
        retrieved_e = retrieved.unsqueeze(1).expand_as(fwd)
        R = F.sigmoid(
            self.r_norm(
                self.W_r2(
                    F.silu(self.W_r1(torch.cat([side, fwd, retrieved_e], dim=-1)))  # type: ignore[reportPrivateImportUsage]
                )
            )
        )
        return fwd * (1 - R) + retrieved_e * R, w.detach()

    def recon_loss(self, val_input: torch.Tensor) -> torch.Tensor:
        """Reconstruction loss encouraging value compression to retain information.

        Args:
            val_input: Side stream mean `(B, d)`.

        Returns:
            Scalar MSE loss.
        """
        return F.mse_loss(self.W_d(self.W_c(val_input)), val_input)
