import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Memory(nn.Module):
    """Content-Addressed Memory.

    Provides a compressed memory tier above the side stream for cross-chunk coherence.
    Each buffer slot stores a tag (a low-dimensional signature of the chunk) and a value
    (a compressed representation of the chunk's side stream state). Writes happen at
    chunk boundaries; reads happen at every forward pass via ReLU-based retrieval.

    The tag projection `W_sig` is semi-isometric (`W_sig @ W_sig.T == I`),
    constructed from Householder reflections. This preserves distances in the tag
    subspace and prevents tag collapse. The threshold `theta` is initialized such
    that unrelated queries return zero weight and related queries return a sparse
    nonzero weight, giving a structural "not found" signal without supervision.

    Args:
        d: Hidden dimension of the main network.
        d_c: Compressed value dimension (`d_c << d`).
        d_sig: Tag signature dimension (`d_sig << d_c`).
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

        self.W_c = nn.Linear(d, d_c, bias=False)
        self.W_d = nn.Linear(d_c, d, bias=False)
        self.v_sig = nn.Parameter(torch.randn(d_sig, d))
        self.theta = nn.Parameter(torch.tensor(3.0 / math.sqrt(d_sig)))  # type: ignore[reportPrivateImportUsage]
        self.W_r1 = nn.utils.spectral_norm(nn.Linear(d * 3, d))
        self.W_r2 = nn.Linear(d, d)
        self.r_norm = nn.RMSNorm(d)
        nn.init.normal_(self.W_r2.weight, std=0.01)
        nn.init.constant_(self.W_r2.bias, -2.0)
        self.W_up = nn.Linear(d_c, d, bias=False)
        self._W_sig_cache: torch.Tensor | None = None

    def get_W_sig(self) -> torch.Tensor:
        """Return the semi-isometric tag projection matrix.

        Constructed as a product of `d_sig` Householder reflections over the
        unconstrained parameter vectors `v_sig`. Cached after each computation
        and invalidated after each optimizer step via `invalidate_cache()`.

        Returns:
            `(d_sig, d)` semi-isometric matrix satisfying
            `W_sig @ W_sig.T == I_{d_sig}`.
        """
        if self._W_sig_cache is not None:
            return self._W_sig_cache
        W = torch.eye(self.d, device=self.v_sig.device)  # type: ignore[reportPrivateImportUsage]
        for i in range(self.d_sig):
            v = self.v_sig[i]
            v = v / v.norm().clamp(min=1e-8)
            W = W - 2 * torch.outer(v, v) @ W  # type: ignore[reportPrivateImportUsage]
        self._W_sig_cache = W[: self.d_sig, :].detach().clone()
        return self._W_sig_cache

    def invalidate_cache(self) -> None:
        """Invalidate the cached `W_sig`.

        Must be called after each optimizer step since `v_sig` has been updated.
        """
        self._W_sig_cache = None

    def write(
        self,
        side_mean: torch.Tensor,
        buffer_tags: torch.Tensor,
        buffer_vals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write a chunk's side stream summary to the buffer.

        Rolls the buffer forward and writes the new tag and value to slot 0.
        Called at chunk boundaries after the side stream has converged.

        Args:
            side_mean: Mean of the side stream over token positions, `(B, d)`.
            buffer_tags: Current tag buffer, `(B, n_slots, d_sig)`.
            buffer_vals: Current value buffer, `(B, n_slots, d_c)`.
        Returns:
            Updated `(buffer_tags, buffer_vals)`, each with the new entry at slot 0.
        """
        W_sig = self.get_W_sig()
        tag = (W_sig @ side_mean.T).T
        val = self.W_c(side_mean)
        buffer_tags = torch.roll(buffer_tags, 1, dims=1)  # type: ignore[reportPrivateImportUsage]
        buffer_vals = torch.roll(buffer_vals, 1, dims=1)  # type: ignore[reportPrivateImportUsage]
        buffer_tags[:, 0, :] = tag.detach()
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

        Uses the last token of `fwd` as the query. Retrieval weights are computed
        via ReLA: `ReLU(sim - theta) / sqrt(d_sig)`. Slots with similarity below
        `theta` receive zero weight, giving a structural null space for unmatched
        queries. The retrieved content is blended into `fwd` via a recovery gate
        conditioned on `[side, fwd, retrieved]`.

        Args:
            fwd: Forward stream `(B, T, d)`.
            side: Side stream `(B, T, d)`.
            buffer_tags: Tag buffer `(B, n_slots, d_sig)`.
            buffer_vals: Value buffer `(B, n_slots, d_c)`.
        Returns:
            - Updated forward stream `(B, T, d)`.
            - Retrieval weights `(B, n_slots)`, detached.
        """
        W_sig = self.get_W_sig()
        q = (W_sig @ fwd[:, -1, :].T).T
        sim = torch.einsum("bd,bsd->bs", q, buffer_tags)  # type: ignore[reportPrivateImportUsage]
        w = F.relu(sim - self.theta) / math.sqrt(self.d_sig)
        retrieved_c = torch.einsum("bs,bsd->bd", w, buffer_vals)  # type: ignore[reportPrivateImportUsage]
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

    def recon_loss(self, side_mean: torch.Tensor) -> torch.Tensor:
        """Reconstruction loss.

        Used as an auxiliary loss during training to encourage the value compression
        to retain sufficient information for retrieval.

        Args:
            side_mean: `(B, d)`
        Returns:
            Scalar MSE loss.
        """
        return F.mse_loss(self.W_d(self.W_c(side_mean)), side_mean)

    def init_W_sig_from_pca(self, side_samples: torch.Tensor) -> None:
        """Warm-start `v_sig` from PCA over collected side stream samples.

        Initializes the tag projection subspace to the top `d_sig` principal
        directions of the side stream distribution. Called after a warmup period
        once the side stream has learned meaningful representations.

        Args:
            side_samples: `(N, d)` tensor of side stream mean vectors.
        """
        with torch.no_grad():
            sc = side_samples - side_samples.mean(0)
            U, _, _ = torch.linalg.svd(sc.T, full_matrices=False)
            self.v_sig.copy_(U[:, : self.d_sig].T)
        self.invalidate_cache()
