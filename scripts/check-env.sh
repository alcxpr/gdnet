#!/usr/bin/env bash
set -euo pipefail

ok() { echo "[OK]  $1"; }
warn() { echo "[WARN] $1"; }
fail() { echo "[FAIL] $1"; }

python --version
python -c "
import sys
v = sys.version_info
if v >= (3, 12) and v < (3, 13):
    print('[OK]  Python', sys.version.split()[0])
else:
    print('[FAIL] Expected 3.12.x, got', sys.version.split()[0])
    sys.exit(1)
"

python -c "
import torch
print('[OK]  torch', torch.__version__)
print('[OK]  CUDA', torch.version.cuda)
print('[OK]  CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('[OK]  Device:', torch.cuda.get_device_name(0))
"

python -c "
try:
    import triton
    print('[OK]  triton', triton.__version__)
except ImportError:
    print('[FAIL] triton not installed')
"

python -c "
try:
    import transformer_engine
    print('[OK]  transformer-engine', transformer_engine.__version__)
    from transformer_engine.common.recipe import DelayedScaling, Format
    print('[OK]  DelayedScaling importable')
    import inspect
    sig = inspect.signature(DelayedScaling)
    print('[OK]  DelayedScaling args:', list(sig.parameters.keys()))
except ImportError as e:
    print('[WARN] transformer-engine not installed:', e)
except Exception as e:
    print('[WARN] transformer-engine error:', e)
"

python -c "
import gdnet
print('[OK]  gdnet imported')
from gdnet import autocast, freeze_sn_iteration
print('[OK]  autocast, freeze_sn_iteration importable')
"

echo ""
echo "Done."
