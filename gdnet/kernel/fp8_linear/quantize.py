from __future__ import annotations

import torch
import triton
import triton.language as tl

FP8_MAX: float = 448.0


@triton.jit
def _quantize_fp8_kernel(
    x_ptr,
    xr_ptr,
    xc_ptr,
    amax_ptr,
    scale,
    M,
    K,
    stride_xm,
    stride_xrm,
    stride_xcK,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)

    x = tl.load(
        x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    tl.atomic_max(amax_ptr, tl.max(tl.abs(x)))

    x_fp8 = tl.clamp(x * scale, -448.0, 448.0).to(tl.float8e4nv)

    tl.store(
        xr_ptr + offs_m[:, None] * stride_xrm + offs_k[None, :],
        x_fp8,
        mask=mask,
    )

    tl.store(
        xc_ptr + offs_k[:, None] * stride_xcK + offs_m[None, :],
        tl.trans(x_fp8),
        mask=tl.trans(mask),
    )


def quantize_fp8(
    x: torch.Tensor,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if torch.cuda.get_device_capability(x.device) < (9, 0):
        raise RuntimeError("quantize_fp8 requires SM90+ (fp8e4nv compute)")
    assert x.dtype == torch.bfloat16, f"expected bfloat16, got {x.dtype}"  # type: ignore
    shape = x.shape
    x = x.reshape(-1, shape[-1])
    if not x.is_contiguous():
        x = x.contiguous()
    M, K = x.shape

    x_row = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x.device)  # type: ignore
    x_col = torch.empty(K, M, dtype=torch.float8_e4m3fn, device=x.device)  # type: ignore
    amax = torch.zeros(1, dtype=torch.float32, device=x.device)  # type: ignore

    BLOCK_M, BLOCK_K = 64, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))

    _quantize_fp8_kernel[grid](
        x,
        x_row,
        x_col,
        amax,
        scale,
        M,
        K,
        x.stride(0),
        x_row.stride(0),
        M,
        BLOCK_M=BLOCK_M,  # type: ignore
        BLOCK_K=BLOCK_K,  # type: ignore
    )

    return x_row.reshape(shape), x_col, amax.squeeze()
