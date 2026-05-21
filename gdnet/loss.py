from __future__ import annotations

import contextlib

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .utils.fp8 import Precision
from .utils.fp8 import autocast as make_autocast


def sn_soft_penalty(layers: list, scale: float = 0.1) -> torch.Tensor:
    """Soft spectral norm penalty across all gate W1 matrices.

    Replaces hard SN constraint with a differentiable penalty: relu(sigma - 1)^2
    summed over all gate weight matrices, estimated via one power iteration step.
    Operates in float32 regardless of training precision for numerical stability.
    """
    device = next(layers[0].parameters()).device
    acc = torch.zeros((), device=device, dtype=torch.float32)  # type: ignore
    for layer in layers:
        for attr in ("gf_W1", "gb_W1", "rf_W1", "rb_W1"):
            W = getattr(layer, attr).weight.float()
            v = torch.randn(W.shape[1], device=device, dtype=torch.float32).detach()  # type: ignore
            v = v / v.norm()
            sigma = (W @ v).norm()
            acc = acc + F.relu(sigma - 1.0).pow(2)
    return scale * acc


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
    raw = getattr(model, "module", model)
    B, n_write, _ = write_chunks.shape
    device = write_chunks.device
    btags = torch.zeros(B, raw.cam.n_slots, raw.cam.d_sig, device=device)  # type: ignore
    bvals = torch.zeros(B, raw.cam.n_slots, raw.cam.d_c, device=device)  # type: ignore
    with torch.no_grad():
        for i in range(n_write):
            _, side, _, _, _, _, fwd_last = model(
                write_chunks[:, i], btags, bvals, sp_group=sp_group
            )
            btags, bvals = raw.write_cam(fwd_last, side, btags, bvals)  # type: ignore
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
    accum_steps: int = 1,
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
        accum_steps: Number of gradient accumulation micro-batches. Tokens and targets
            are split along the batch dimension; losses are scaled by 1/accum_steps so
            the effective gradient matches a single full-batch forward pass.

    Returns:
        Task loss value for this step.
    """
    base_model = getattr(model, "module", model)
    no_sync = getattr(model, "no_sync", contextlib.nullcontext)

    B = tokens.shape[0]
    assert B % accum_steps == 0, (
        f"batch size {B} not divisible by accum_steps {accum_steps}"
    )
    mb = B // accum_steps

    g_task = [torch.zeros_like(p) for p in params]  # type: ignore
    g_info = [torch.zeros_like(p) for p in params]  # type: ignore
    total_loss = 0.0

    optimizer.zero_grad()

    for i in range(accum_steps):
        tok_i = tokens[i * mb : (i + 1) * mb]
        tgt_i = targets[i * mb : (i + 1) * mb]
        wc_i = write_chunks[i * mb : (i + 1) * mb] if write_chunks is not None else None

        with make_autocast(precision):  # type: ignore
            btags, bvals = (
                build_cam_buffer(model, wc_i, sp_group=sp_group)
                if wc_i is not None and base_model.cam_enabled  # type: ignore
                else (None, None)
            )
            logits, side, _, _, gate_vals, _, _ = model(
                tok_i, btags, bvals, return_gates=True, sp_group=sp_group
            )
            loss_task = (
                F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt_i.reshape(-1))
                / accum_steps
            )

        with no_sync():
            if scaler:
                scaler.scale(loss_task).backward(retain_graph=True)
            else:
                loss_task.backward(retain_graph=True)

        for acc, p in zip(g_task, params):
            if p.grad is not None:
                acc.add_(p.grad)

        optimizer.zero_grad()

        with make_autocast(precision):  # type: ignore
            loss_info = (
                gate_info_loss_from_vals(gate_vals, base_model.n_layers) / accum_steps  # type: ignore
            )
            if base_model.layers:  # type: ignore
                loss_info = loss_info + sn_soft_penalty(base_model.layers) / accum_steps  # type: ignore
            if base_model.cam_enabled:  # type: ignore
                loss_info = (
                    loss_info
                    + 0.1
                    * base_model.cam.recon_loss(  # type: ignore
                        side[0].mean(dim=1)
                    )
                    / accum_steps
                )

        with no_sync():
            if scaler:
                scaler.scale(loss_info).backward()
            else:
                loss_info.backward()

        if base_model.trans_enabled and tok_i.shape[1] >= 2:  # type: ignore
            with make_autocast(precision):  # type: ignore
                mid = tok_i.shape[1] // 2
                z_t = side[0][:, :mid].mean(dim=1)
                z_t1 = side[0][:, mid:].mean(dim=1).detach()
                loss_trans = 0.1 * base_model.trans_ops.loss(z_t, z_t1) / accum_steps  # type: ignore
            with no_sync():
                if scaler:
                    scaler.scale(loss_trans).backward()
                else:
                    loss_trans.backward()

        for acc, p in zip(g_info, params):
            if p.grad is not None:
                acc.add_(p.grad)

        optimizer.zero_grad()
        total_loss += loss_task.item()

    denom = torch.zeros((), device=params[0].device, dtype=torch.float32)  # type: ignore
    proj_coef_num = torch.zeros((), device=params[0].device, dtype=torch.float32)  # type: ignore
    for gt, gi in zip(g_task, g_info):
        denom += gt.float().pow(2).sum()
        proj_coef_num += (gi.float() * gt.float()).sum()
    if dist.is_initialized():
        dist.all_reduce(denom)
        dist.all_reduce(proj_coef_num)
    proj_coef = proj_coef_num / denom.clamp(min=1e-8)

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

    return total_loss
