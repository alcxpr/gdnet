from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentCAM(nn.Module):
    """Baseline: content-only softmax retrieval."""

    def __init__(self, d: int, d_sig: int, d_c: int, n_slots: int) -> None:
        super().__init__()
        self.n_slots = n_slots
        self.d_sig = d_sig
        self.d_c = d_c
        self.W_tag = nn.Linear(d, d_sig, bias=False)
        self.W_c = nn.Linear(d, d_c, bias=False)
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
        tag = self.W_tag(tag_input)
        val = self.W_c(val_input)
        buffer_tags = torch.roll(buffer_tags, 1, dims=1)  # type: ignore
        buffer_vals = torch.roll(buffer_vals, 1, dims=1)  # type: ignore
        buffer_tags[:, 0, :] = tag
        buffer_vals[:, 0, :] = val.detach()
        return buffer_tags, buffer_vals

    def read(
        self,
        fwd: torch.Tensor,
        side: torch.Tensor,
        buffer_tags: torch.Tensor,
        buffer_vals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.W_tag(fwd[:, -1, :])
        sim = torch.einsum("bd,bsd->bs", q, buffer_tags)  # type: ignore
        w = torch.softmax(sim, dim=-1)  # type: ignore
        retrieved_c = torch.einsum("bs,bsd->bd", w, buffer_vals)  # type: ignore
        retrieved = self.W_up(retrieved_c)
        retrieved_e = retrieved.unsqueeze(1).expand_as(fwd)
        R = F.sigmoid(
            self.r_norm(
                self.W_r2(
                    F.silu(self.W_r1(torch.cat([side, fwd, retrieved_e], dim=-1)))  # type: ignore
                )
            )
        )
        return fwd * (1 - R) + retrieved_e * R, R, w


class PosGateCAM(nn.Module):
    """Proposed: content + input-conditioned position gate with learned slot embeddings.

    sim_s = q^T (t_s + alpha * e_s o gamma)  where gamma = sigmoid(W_pos @ q)
    """

    def __init__(self, d: int, d_sig: int, d_c: int, n_slots: int) -> None:
        super().__init__()
        self.n_slots = n_slots
        self.d_sig = d_sig
        self.d_c = d_c
        self.W_tag = nn.Linear(d, d_sig, bias=False)
        self.W_slot = nn.Embedding(n_slots, d_sig)
        self.W_pos = nn.Linear(d_sig, d_sig, bias=False)
        self.rho = nn.Parameter(torch.zeros(1))  # type: ignore
        self.W_c = nn.Linear(d, d_c, bias=False)
        self.W_up = nn.Linear(d_c, d, bias=False)
        self.W_r1 = nn.utils.spectral_norm(nn.Linear(d * 3, d))
        self.W_r2 = nn.Linear(d, d)
        self.r_norm = nn.RMSNorm(d)
        nn.init.normal_(self.W_r2.weight, std=0.01)
        nn.init.constant_(self.W_r2.bias, -2.0)
        decay = 0.85
        with torch.no_grad():
            for s in range(n_slots):
                self.W_slot.weight[s] *= decay**s

    def write(
        self,
        tag_input: torch.Tensor,
        val_input: torch.Tensor,
        buffer_tags: torch.Tensor,
        buffer_vals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tag = self.W_tag(tag_input)
        val = self.W_c(val_input)
        buffer_tags = torch.roll(buffer_tags, 1, dims=1)  # type: ignore
        buffer_vals = torch.roll(buffer_vals, 1, dims=1)  # type: ignore
        buffer_tags[:, 0, :] = tag
        buffer_vals[:, 0, :] = val.detach()
        return buffer_tags, buffer_vals

    def read(
        self,
        fwd: torch.Tensor,
        side: torch.Tensor,
        buffer_tags: torch.Tensor,
        buffer_vals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.W_tag(fwd[:, -1, :])
        gamma = F.sigmoid(self.W_pos(q))  # type: ignore
        alpha = torch.exp(self.rho)  # type: ignore
        slot_ids = torch.arange(self.n_slots, device=fwd.device)  # type: ignore
        e = self.W_slot(slot_ids)  # (n_slots, d_sig)
        biased = buffer_tags + alpha * e.unsqueeze(0) * gamma.unsqueeze(1)
        sim = torch.einsum("bd,bsd->bs", q, biased)  # type: ignore
        w = F.softmax(sim, dim=-1)
        retrieved_c = torch.einsum("bs,bsd->bd", w, buffer_vals)  # type: ignore
        retrieved = self.W_up(retrieved_c)
        retrieved_e = retrieved.unsqueeze(1).expand_as(fwd)
        R = F.sigmoid(
            self.r_norm(
                self.W_r2(
                    F.silu(self.W_r1(torch.cat([side, fwd, retrieved_e], dim=-1)))  # type: ignore
                )
            )
        )
        return fwd * (1 - R) + retrieved_e * R, R, w
