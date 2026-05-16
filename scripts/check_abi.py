"""Check CUDA / PyTorch / TransformerEngine ABI compatibility.

Prints version strings and flags mismatches that would cause silent
fp8 failures or import errors at runtime.

Usage:
    uv run python scripts/check_abi.py
"""

from __future__ import annotations

import importlib
import sys


def _section(title: str) -> None:
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print("=" * 50)


def check_python() -> None:
    _section("Python")
    print(f"  version : {sys.version}")


def check_torch() -> None:
    import torch

    _section("PyTorch")
    print(f"  torch         : {torch.__version__}")
    print(f"  CUDA (built)  : {torch.version.cuda}")
    print(f"  cuDNN (built) : {torch.backends.cudnn.version()}")
    print(f"  debug build   : {torch.version.debug}")

    if torch.cuda.is_available():
        rt = torch.version.cuda
        dev = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(f"  GPU           : {dev}  (sm_{cap[0]}{cap[1]})")
        print(f"  CUDA runtime  : {rt}")
        if cap < (8, 9):
            print("  [WARN] fp8 requires sm_89+ (H100/Ada). This GPU does not support fp8.")
        else:
            print("  [OK] GPU supports fp8 (sm_89+)")
    else:
        print("  [WARN] CUDA not available")


def check_te() -> None:
    _section("TransformerEngine")
    spec = importlib.util.find_spec("transformer_engine")
    if spec is None:
        print("  [MISSING] transformer_engine not installed")
        return

    import transformer_engine as te

    print(f"  version       : {te.__version__}")

    # TE exposes the CUDA version it was compiled against
    cuda_ver = getattr(te, "__cuda_version__", None) or getattr(
        te.pytorch, "__cuda_version__", None
    ) if hasattr(te, "pytorch") else None
    if cuda_ver:
        print(f"  CUDA (built)  : {cuda_ver}")

    import torch
    torch_cuda = torch.version.cuda
    if cuda_ver and torch_cuda and str(cuda_ver).split(".")[0] != str(torch_cuda).split(".")[0]:
        print(f"  [WARN] CUDA major version mismatch: TE={cuda_ver} vs torch={torch_cuda}")
    elif cuda_ver:
        print("  [OK] CUDA versions compatible")

    # Check fp8_autocast importable
    try:
        from transformer_engine.pytorch import fp8_autocast  # noqa: F401
        print("  [OK] fp8_autocast importable")
    except Exception as e:
        print(f"  [FAIL] fp8_autocast import: {e}")

    # Check recipe
    try:
        from transformer_engine.common.recipe import DelayedScaling, Format  # noqa: F401
        print("  [OK] DelayedScaling recipe importable")
    except Exception as e:
        print(f"  [FAIL] recipe import: {e}")


def check_triton() -> None:
    _section("Triton")
    spec = importlib.util.find_spec("triton")
    if spec is None:
        print("  [MISSING] triton not installed")
        return
    import triton
    print(f"  version       : {triton.__version__}")

    # Smoke-compile a minimal kernel to catch ABI issues
    try:
        import torch
        import triton.language as tl

        @triton.jit
        def _noop(x_ptr, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            x = tl.load(x_ptr + pid * BLOCK + tl.arange(0, BLOCK))
            tl.store(x_ptr + pid * BLOCK + tl.arange(0, BLOCK), x)

        x = torch.zeros(64, device="cuda")
        _noop[(1,)](x, BLOCK=64)
        print("  [OK] JIT smoke test passed")
    except Exception as e:
        print(f"  [FAIL] JIT smoke test: {e}")


def main() -> None:
    check_python()
    check_torch()
    check_te()
    check_triton()
    print()


if __name__ == "__main__":
    main()
