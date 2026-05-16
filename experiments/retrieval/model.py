from __future__ import annotations

import torch
import torch.nn as nn
from cam import ContentCAM, PosGateCAM

from gdnet.layer import GDLayer


class ChunkEncoder(nn.Module):
    def __init__(self, d: int, n_layers: int, kernel_size: int, n_cycles: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([GDLayer(d, kernel_size) for _ in range(n_layers)])
        self.n_cycles = n_cycles

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        side: list[torch.Tensor] = [torch.zeros_like(x) for _ in self.layers]  # type: ignore
        for _ in range(self.n_cycles):
            for i, layer in enumerate(self.layers):
                x, side[i] = layer.fwd_step(x, side[i])  # type: ignore
            for i, layer in reversed(list(enumerate(self.layers))):
                x, side[i] = layer.bwd_step(x, side[i])  # type: ignore
        return x, side


class RetrievalModel(nn.Module):
    """Chunk-sequential model with pluggable CAM.

    Args:
        vocab_size: Token vocabulary size.
        n_values: Number of value classes (prediction head output size).
        d: Hidden dimension.
        d_sig: CAM tag dimension.
        d_c: CAM value compression dimension.
        n_slots: Number of CAM buffer slots.
        n_layers: GDLayer stack depth for the chunk encoder.
        kernel_size: Depthwise conv kernel size.
        n_cycles: Forward+backward cycles per chunk.
        use_pos_gate: If True, use PosGateCAM; otherwise ContentCAM.
    """

    def __init__(
        self,
        vocab_size: int,
        n_values: int,
        d: int,
        d_sig: int,
        d_c: int,
        n_slots: int,
        n_layers: int,
        kernel_size: int,
        n_cycles: int,
        use_pos_gate: bool,
    ) -> None:
        super().__init__()
        self.n_slots = n_slots
        self.d_sig = d_sig
        self.d_c = d_c
        self.embed = nn.Embedding(vocab_size, d)
        self.encoder = ChunkEncoder(d, n_layers, kernel_size, n_cycles)
        self.cam: ContentCAM | PosGateCAM = (
            PosGateCAM(d, d_sig, d_c, n_slots)
            if use_pos_gate
            else ContentCAM(d, d_sig, d_c, n_slots)
        )
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, n_values)

    def forward(
        self, chunks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            chunks: (B, n_chunks, T) - last chunk is the query.

        Returns:
            logits: (B, n_values)
            r_scalar: (B,) - mean R at the last token of the query chunk
            w: (B, n_slots) - retrieval weights
        """
        B, n_chunks, T = chunks.shape
        device, dtype = chunks.device, self.embed.weight.dtype

        buffer_tags = torch.zeros(  # type: ignore
            B, self.n_slots, self.d_sig, device=device, dtype=dtype
        )
        buffer_vals = torch.zeros(B, self.n_slots, self.d_c, device=device, dtype=dtype)  # type: ignore

        for i in range(n_chunks - 1):
            x = self.embed(chunks[:, i, :])
            fwd, side = self.encoder(x)
            buffer_tags, buffer_vals = self.cam.write(
                fwd[:, -1, :], side[0].mean(dim=1), buffer_tags, buffer_vals
            )

        x = self.embed(chunks[:, -1, :])
        fwd, side = self.encoder(x)
        fwd_new, R, w = self.cam.read(fwd, side[0], buffer_tags, buffer_vals)

        logits = self.head(self.norm(fwd_new[:, -1, :]))
        r_scalar = R[:, -1, :].mean(-1)
        return logits, r_scalar, w
