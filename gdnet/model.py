from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .layer import GDLayer
from .memory import Memory as CAM
from .operators import TransitionOperators


class GDNet(nn.Module):
    """Gated Dissipative Network for sequence modeling.

    A language model built on stacked GDLayers with optional content-addressed
    memory and transition operators. The embedding and output head share weights.
    CAM and transition operators are disabled at init and enabled during training
    via `cam_enabled` and `trans_enabled` flags.

    Args:
        vocab_size: Vocabulary size.
        d_embed: Token embedding dimension.
        d: Hidden dimension.
        n_layers: Number of GDLayers.
        n_cycles: Number of forward+backward cycles per forward pass.
        chunk_size: Sequence chunk size used during training and generation.
        d_c: CAM value compression dimension. Defaults to `d // 4`.
        d_sig: CAM tag signature dimension. Defaults to `d // 8`.
        n_slots: Number of CAM buffer slots.
        kernel_size: Kernel size for the causal depthwise convolution carrier.
        n_ops: Number of transition operators.
    """

    def __init__(
        self,
        vocab_size: int,
        d_embed: int,
        d: int,
        n_layers: int,
        n_cycles: int = 3,
        chunk_size: int = 512,
        d_c: int | None = None,
        d_sig: int | None = None,
        n_slots: int = 32,
        kernel_size: int = 7,
        n_ops: int = 8,
    ) -> None:
        super().__init__()
        d_c = d_c or d // 4
        d_sig = d_sig or d // 8

        self.embed = nn.Embedding(vocab_size, d_embed)
        self.proj_in = nn.Linear(d_embed, d, bias=False)
        self.layers = nn.ModuleList([GDLayer(d, kernel_size) for _ in range(n_layers)])
        self.proj_out = nn.Linear(d, d_embed, bias=False)
        self.head = nn.Linear(d_embed, vocab_size, bias=False)
        self.head.weight = self.embed.weight

        self.cam = CAM(d, d_c, d_sig, n_slots, chunk_size)
        self.trans_ops = TransitionOperators(d, n_ops)
        self.norm_out = nn.RMSNorm(d)

        self.d = d
        self.d_embed = d_embed
        self.n_layers = n_layers
        self.n_cycles = n_cycles
        self.chunk_size = chunk_size
        self.cam_enabled = False
        self.trans_enabled = False

    def one_cycle(
        self,
        fwd: torch.Tensor,
        side: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run one forward+backward pass through all layers.

        Args:
            fwd: Forward stream `(B, T, d)`.
            side: Per-layer side streams, each `(B, T, d)`.

        Returns:
            Updated `(fwd, side)`.
        """
        for i, layer in enumerate(self.layers):
            fwd, side[i] = layer.fwd_step(fwd, side[i])  # type: ignore
        for i, layer in reversed(list(enumerate(self.layers))):
            fwd, side[i] = layer.bwd_step(fwd, side[i])  # type: ignore
        return fwd, side

    def forward(
        self,
        tokens: torch.Tensor,
        buffer_tags: torch.Tensor | None = None,
        buffer_vals: torch.Tensor | None = None,
        return_gates: bool = False,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor],
        torch.Tensor | None,
    ]:
        """Forward pass.

        Args:
            tokens: Input token ids `(B, T)`.
            buffer_tags: CAM tag buffer `(B, n_slots, d_sig)`. Initialized to zeros
                if not provided.
            buffer_vals: CAM value buffer `(B, n_slots, d_c)`. Initialized to zeros
                if not provided.
            return_gates: If `True`, collect forward gate values for loss computation.
                Gate tensors retain gradients when collected this way.

        Returns:
            - logits `(B, T, vocab_size)`
            - side streams, list of `(B, T, d)` per layer
            - buffer_tags `(B, n_slots, d_sig)`
            - buffer_vals `(B, n_slots, d_c)`
            - gate_vals: list of gate tensors, empty if `return_gates=False`
            - cam_weights: ReLA retrieval weights `(B, n_slots)` or `None`
        """
        B, T = tokens.shape
        fwd = self.proj_in(self.embed(tokens))
        dtype, device = fwd.dtype, fwd.device
        side: list[torch.Tensor] = [
            torch.zeros(B, T, self.d, dtype=dtype, device=device)  # type: ignore
            for _ in self.layers
        ]

        if buffer_tags is None:
            buffer_tags = torch.zeros(  # type: ignore
                B, self.cam.n_slots, self.cam.d_sig, dtype=dtype, device=device
            )
        if buffer_vals is None:
            buffer_vals = torch.zeros(  # type: ignore
                B, self.cam.n_slots, self.cam.d_c, dtype=dtype, device=device
            )

        gate_vals: list[torch.Tensor] = []
        for _ in range(self.n_cycles):
            if return_gates:
                for i, layer in enumerate(self.layers):
                    fwd, side[i], g = layer.fwd_step(fwd, side[i], return_gate=True)  # type: ignore
                    gate_vals.append(g)
                for i, layer in reversed(list(enumerate(self.layers))):
                    fwd, side[i] = layer.bwd_step(fwd, side[i])  # type: ignore
            else:
                fwd, side = self.one_cycle(fwd, side)

        cam_weights: torch.Tensor | None = None
        if self.cam_enabled:
            fwd, cam_weights = self.cam.read(fwd, side[0], buffer_tags, buffer_vals)

        logits = self.head(self.proj_out(self.norm_out(fwd)))
        return logits, side, buffer_tags, buffer_vals, gate_vals, cam_weights

    def write_cam(
        self,
        side: list[torch.Tensor],
        buffer_tags: torch.Tensor,
        buffer_vals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write the current side stream summary to the CAM buffer.

        No-op if `cam_enabled` is `False`.

        Args:
            side: Per-layer side streams.
            buffer_tags: Current tag buffer `(B, n_slots, d_sig)`.
            buffer_vals: Current value buffer `(B, n_slots, d_c)`.

        Returns:
            Updated `(buffer_tags, buffer_vals)`.
        """
        if self.cam_enabled:
            buffer_tags, buffer_vals = self.cam.write(
                side[0].mean(dim=1), buffer_tags, buffer_vals
            )
        return buffer_tags, buffer_vals

    @torch.no_grad()
    def collect_side_samples(
        self,
        loader: DataLoader,
        n_batches: int = 50,
    ) -> torch.Tensor:
        """Collect side stream means for CAM PCA initialization.

        Args:
            loader: DataLoader yielding `(tokens, targets)` batches.
            n_batches: Number of batches to collect.

        Returns:
            `(N, d)` tensor of side stream mean vectors.
        """
        self.eval()
        device = next(self.parameters()).device
        samples = []
        for i, (x, _) in enumerate(loader):
            if i >= n_batches:
                break
            _, side, _, _, _, _ = self.forward(x.to(device))
            samples.append(side[0].mean(dim=1).cpu())
        return torch.cat(samples, dim=0)  # type: ignore

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        top_p: float = 0.9,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate token ids autoregressively with top-p sampling.

        Writes to the CAM buffer at every `chunk_size` steps.

        Args:
            input_ids: Prompt token ids `(1, T)`.
            max_new_tokens: Maximum number of new tokens to generate.
            top_p: Nucleus sampling probability threshold.
            temperature: Sampling temperature.

        Returns:
            Token ids `(1, T + n)` including the prompt.
        """
        self.eval()
        device = next(self.parameters()).device
        ids = input_ids.to(device)
        buffer_tags = torch.zeros(1, self.cam.n_slots, self.cam.d_sig, device=device)  # type: ignore
        buffer_vals = torch.zeros(1, self.cam.n_slots, self.cam.d_c, device=device)  # type: ignore

        for step in range(max_new_tokens):
            ctx = ids[:, -self.chunk_size :]
            logits, side, buffer_tags, buffer_vals, _, _ = self.forward(
                ctx, buffer_tags, buffer_vals
            )
            if step % self.chunk_size == 0:
                buffer_tags, buffer_vals = self.write_cam(
                    side, buffer_tags, buffer_vals
                )

            logits_last = logits[:, -1, :] / temperature
            probs = F.softmax(logits_last, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)  # type: ignore
            cumprobs = torch.cumsum(sorted_probs, dim=-1)  # type: ignore
            sorted_probs[~((cumprobs - sorted_probs) < top_p)] = 0
            sorted_probs /= sorted_probs.sum()
            next_id = sorted_idx[0, torch.multinomial(sorted_probs[0], 1)]  # type: ignore
            ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)  # type: ignore

        return ids
