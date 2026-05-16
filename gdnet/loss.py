from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.fp8 import Precision
from .utils.fp8 import autocast as make_autocast


def gate_info_loss_from_vals(
    gate_vals: list[torch.Tensor],
    n_layers: int,
) -> torch.Tensor:
    """Compute the information austerity loss from pre-collected gate values.

    Takes the last cycle's gate values and returns their mean product across layers.
    Using pre-collected values avoids a redundant forward pass.

    Args:
        gate_vals: Gate tensors collected during a forward pass with `return_gates=True`.
        n_layers: Number of layers; selects the last cycle's gate values.

    Returns:
        Scalar loss penalizing the cumulative gate product.
    """
    last_cycle = gate_vals[-n_layers:]
    return torch.stack([g.mean() for g in last_cycle]).prod()


def build_cam_buffer(
    model: nn.Module,
    write_chunks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Populate the CAM buffer by processing write chunks sequentially.

    Must be called inside the same autocast context as the subsequent forward pass
    so that gradients flow through W_tag, W_c, W_pos, and rho.

    Args:
        model: GDNet model with `cam_enabled`, `cam`, and `write_cam` attributes.
        write_chunks: Write-chunk token ids `(B, n_write, T)`.

    Returns:
        `(buffer_tags, buffer_vals)` ready to pass into `model.forward`.
    """
    B, n_write, _ = write_chunks.shape
    device = write_chunks.device
    btags = torch.zeros(B, model.cam.n_slots, model.cam.d_sig, device=device)  # type: ignore
    bvals = torch.zeros(B, model.cam.n_slots, model.cam.d_c, device=device)  # type: ignore
    for i in range(n_write):
        _, side, _, _, _, _, fwd_last = model(write_chunks[:, i], btags, bvals)
        btags, bvals = model.write_cam(fwd_last, side, btags, bvals)  # type: ignore
    return btags, bvals


def projected_step(
    model: nn.Module,
    params: list[nn.Parameter],
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 0.1,
    scaler: torch.cuda.amp.GradScaler | None = None,
    precision: Precision = "bf16",
    write_chunks: torch.Tensor | None = None,
) -> float:
    """Single training step with projected gradient optimization.

    Enforces a task-primary hierarchy: the information austerity gradient is projected
    onto the nullspace of the task gradient, so it can only shape the solution within
    the task level set and never degrades task performance.

    Two forward passes are performed:
    - Pass 1: task loss + gate values collected for pass 2.
    - Pass 2: info loss (gate austerity + optional CAM reconstruction + optional
      capability loss), using the same gate values where possible.

    Args:
        model: The GDNet model. Must expose `n_layers`, `cam_enabled`, `trans_enabled`,
            `d`, `cam`, and `trans_ops` attributes.
        params: List of parameters to optimize (from `model.parameters()`).
        optimizer: Optimizer instance.
        tokens: Input token ids `(B, T)`.
        targets: Target token ids `(B, T)`.
        beta: Scale factor for the projected info gradient.
        scaler: Optional AMP grad scaler for bf16 training. Not used for fp8.
        precision: Training precision passed to `autocast`.
        write_chunks: Optional write-chunk token ids `(B, n_write, T)`. When provided
            and `model.cam_enabled`, the CAM buffer is populated via sequential writes
            before each forward pass so CAM parameters receive training gradients.

    Returns:
        Task loss value for this step.
    """
    optimizer.zero_grad()
    with make_autocast(precision):  # type: ignore
        btags, bvals = (
            build_cam_buffer(model, write_chunks)
            if write_chunks is not None and model.cam_enabled  # type: ignore
            else (None, None)
        )
        logits, _, _, _, gate_vals, _, _ = model(
            tokens, btags, bvals, return_gates=True
        )
        loss_task = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
    if scaler:
        scaler.scale(loss_task).backward()
    else:
        loss_task.backward()
    g_task = [
        p.grad.clone() if p.grad is not None else torch.zeros_like(p)  # type: ignore
        for p in params
    ]

    optimizer.zero_grad()
    with make_autocast(precision):  # type: ignore
        btags, bvals = (
            build_cam_buffer(model, write_chunks)
            if write_chunks is not None and model.cam_enabled  # type: ignore
            else (None, None)
        )
        logits, side, _, _, gate_vals, _, _ = model(
            tokens, btags, bvals, return_gates=True
        )
        loss_info = gate_info_loss_from_vals(gate_vals, model.n_layers)  # type: ignore

        if model.cam_enabled:
            loss_info = loss_info + 0.1 * model.cam.recon_loss(side[0].mean(dim=1))  # type: ignore

        if model.trans_enabled and tokens.shape[1] >= 2:
            mid = tokens.shape[1] // 2
            _, side1, _, _, _, _, _ = model(tokens[:, :mid])
            _, side2, _, _, _, _, _ = model(tokens[:, mid:])
            z_t = side1[0].mean(dim=1)
            z_t1 = side2[0].mean(dim=1).detach()
            loss_info = loss_info + 0.1 * model.trans_ops.loss(z_t, z_t1)  # type: ignore

    if scaler:
        scaler.scale(loss_info).backward()
    else:
        loss_info.backward()
    g_info = [
        p.grad.clone() if p.grad is not None else torch.zeros_like(p)  # type: ignore
        for p in params
    ]

    flat_task = torch.cat([g.flatten() for g in g_task])  # type: ignore
    flat_info = torch.cat([g.flatten() for g in g_info])  # type: ignore
    denom = flat_task.dot(flat_task).clamp(min=1e-8)
    proj_coef = flat_info.dot(flat_task) / denom
    flat_perp = flat_info - proj_coef * flat_task

    optimizer.zero_grad()
    idx = 0
    for p, gt in zip(params, g_task):
        sz = gt.numel()
        p.grad = gt + beta * flat_perp[idx : idx + sz].reshape(gt.shape)
        idx += sz

    if scaler:
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    return loss_task.item()
