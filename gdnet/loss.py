from __future__ import annotations

import contextlib

import torch
import torch.distributed as dist
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
    sp_group: dist.ProcessGroup | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Populate the CAM buffer by processing write chunks sequentially.

    Runs under no_grad: W_tag and W_c receive gradients through the main forward's
    read path and recon_loss respectively, so no write-path graph is needed.

    Args:
        model: GDNet model with `cam_enabled`, `cam`, and `write_cam` attributes.
        write_chunks: Write-chunk token ids `(B, n_write, T)`.
        sp_group: Sequence-parallel process group, or None for single-device.

    Returns:
        `(buffer_tags, buffer_vals)` ready to pass into `model.forward`.
    """
    B, n_write, _ = write_chunks.shape
    device = write_chunks.device
    btags = torch.zeros(B, model.cam.n_slots, model.cam.d_sig, device=device)  # type: ignore
    bvals = torch.zeros(B, model.cam.n_slots, model.cam.d_c, device=device)  # type: ignore
    with torch.no_grad():
        for i in range(n_write):
            _, side, _, _, _, _, fwd_last = model(
                write_chunks[:, i], btags, bvals, sp_group=sp_group
            )
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
    sp_group: dist.ProcessGroup | None = None,
    grad_clip: float = 0.0,
) -> float:
    """Single training step with projected gradient optimization.

    Enforces a task-primary hierarchy: the information austerity gradient is projected
    onto the nullspace of the task gradient, so it can only shape the solution within
    the task level set and never degrades task performance.

    Uses a single forward pass shared by both backward passes via `retain_graph=True`.
    The primary activation graph is freed immediately after the info backward (no
    retain_graph on the second call), avoiding GPU OOM without CPU offload. Projection
    dot products are accumulated parameter-by-parameter to avoid a full-gradient
    concatenation spike.

    Args:
        model: The GDNet model. Must expose `n_layers`, `cam_enabled`, `trans_enabled`,
            `cam`, and `trans_ops` attributes.
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
            build_cam_buffer(model, write_chunks, sp_group=sp_group)
            if write_chunks is not None and model.cam_enabled  # type: ignore
            else (None, None)
        )
        logits, side, _, _, gate_vals, _, _ = model(
            tokens, btags, bvals, return_gates=True, sp_group=sp_group
        )
        loss_task = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

    base_model = getattr(model, "module", model)
    no_sync = getattr(model, "no_sync", contextlib.nullcontext)

    with no_sync():
        if scaler:
            scaler.scale(loss_task).backward(retain_graph=True)
        else:
            loss_task.backward(retain_graph=True)
    g_task = [
        p.grad.clone() if p.grad is not None else torch.zeros_like(p)  # type: ignore
        for p in params
    ]

    optimizer.zero_grad()
    with make_autocast(precision):  # type: ignore
        loss_info_base = gate_info_loss_from_vals(gate_vals, base_model.n_layers)  # type: ignore
        if base_model.cam_enabled:
            loss_info_base = loss_info_base + 0.1 * base_model.cam.recon_loss(  # type: ignore
                side[0].mean(dim=1)
            )
    with no_sync():
        if scaler:
            scaler.scale(loss_info_base).backward()
        else:
            loss_info_base.backward()

    if base_model.trans_enabled and tokens.shape[1] >= 2:
        with make_autocast(precision):  # type: ignore
            mid = tokens.shape[1] // 2
            _, side1, _, _, _, _, _ = model(tokens[:, :mid], sp_group=sp_group)
            _, side2, _, _, _, _, _ = model(tokens[:, mid:], sp_group=sp_group)
            z_t = side1[0].mean(dim=1)
            z_t1 = side2[0].mean(dim=1).detach()
            loss_trans = 0.1 * model.trans_ops.loss(z_t, z_t1)  # type: ignore
        with no_sync():
            if scaler:
                scaler.scale(loss_trans).backward()
            else:
                loss_trans.backward()

    g_info = [
        p.grad.clone() if p.grad is not None else torch.zeros_like(p)  # type: ignore
        for p in params
    ]

    denom = torch.tensor(0.0, device=params[0].device, dtype=torch.float32)  # type: ignore
    proj_coef_num = torch.tensor(0.0, device=params[0].device, dtype=torch.float32)  # type: ignore
    for gt, gi in zip(g_task, g_info):
        denom += gt.float().pow(2).sum()
        proj_coef_num += (gi.float() * gt.float()).sum()
    if dist.is_initialized():
        dist.all_reduce(denom)
        dist.all_reduce(proj_coef_num)
    proj_coef = proj_coef_num / denom.clamp(min=1e-8)

    optimizer.zero_grad()
    for p, gt, gi in zip(params, g_task, g_info):
        p.grad = gt + beta * (gi - proj_coef * gt)

    if dist.is_initialized():
        world_size = dist.get_world_size()
        for p in params:
            if p.grad is not None:
                dist.all_reduce(p.grad)
                p.grad /= world_size

    if grad_clip > 0.0:
        if scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, grad_clip)

    if scaler:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    return loss_task.item()
