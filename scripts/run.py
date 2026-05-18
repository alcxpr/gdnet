"""Single entry point: prepare data then launch training.

Auto-detects GPU count, derives tokenized file paths from phase metadata,
runs prepare_data.py for any missing files, then hands off to training.py.

Usage:
    python scripts/run.py                                    # all defaults
    python scripts/run.py --gpus 2                           # force 2 GPUs
    python scripts/run.py --data-dir /data/tokenized         # custom data dir
    python scripts/run.py --config configs/training.yaml --gpus 4
    python scripts/run.py --skip-prepare                     # assume data ready
    python scripts/run.py --prepare-only                     # tokenize, don't train
    python scripts/run.py --resume checkpoints/step_1000.pt  # resume checkpoint
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).parent.parent


def _derive_path(phase: dict, data_dir: Path) -> str | None:
    source = phase.get("dataset", "fineweb-edu")
    if source == "fineweb-edu":
        subset = phase.get("subset", "sample-10BT").replace("/", "_")
        lo = phase.get("min_int_score", 3)
        hi = phase.get("max_int_score", 5)
        return str(data_dir / f"fineweb_edu_{subset}_{lo}_{hi}.bin")
    if source == "nemotron":
        subset = phase.get("subset", "everything").replace("/", "_")
        return str(data_dir / f"nemotron_{subset}.bin")
    return None


def _prepare_cmd(phase: dict, path: str, encoding: str) -> list[str]:
    source = phase.get("dataset", "fineweb-edu")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "prepare_data.py"),
        "--dataset", source,
        "--encoding", encoding,
        "--out", path,
    ]
    if source == "fineweb-edu":
        cmd += [
            "--subset", phase.get("subset", "sample-10BT"),
            "--min-score", str(phase.get("min_int_score", 3)),
            "--max-score", str(phase.get("max_int_score", 5)),
            "--min-token-count", str(phase.get("min_token_count", 128)),
        ]
    elif source == "nemotron":
        cmd += [
            "--subset", phase.get("subset", "everything"),
            "--min-chars", str(phase.get("min_chars", 0)),
        ]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "training.yaml"))
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "tokenized"))
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", default=None, help="path to checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg: dict = yaml.safe_load(f) or {}

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    encoding = cfg.get("tokenizer", "cl100k_base")
    phases = cfg.get("phases", [])

    # Fill in paths for phases that don't have one
    for phase in phases:
        if not phase.get("path"):
            phase["path"] = _derive_path(phase, data_dir)

    # Prepare missing binary files
    if not args.skip_prepare:
        seen: set[str] = set()
        for phase in phases:
            path = phase.get("path")
            if not path or path in seen:
                continue
            seen.add(path)
            if Path(path).exists():
                size_gb = Path(path).stat().st_size / 1024**3
                print(f"[data] found {path} ({size_gb:.2f} GB)")
                continue
            source = phase.get("dataset", "fineweb-edu")
            if source not in ("fineweb-edu", "nemotron"):
                continue
            print(f"[data] preparing {path} ...")
            cmd = _prepare_cmd(phase, path, encoding)
            subprocess.run(cmd, check=True)

    if args.prepare_only:
        print("[run] --prepare-only: data ready, exiting.")
        return

    # Write a temp config with paths filled in
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=ROOT
    ) as tmp:
        yaml.dump(cfg, tmp)
        tmp_config = tmp.name

    # Detect GPU count
    n_gpus = args.gpus or torch.cuda.device_count()
    n_gpus = max(n_gpus, 1)

    print(f"[run] launching training on {n_gpus} GPU(s)")

    if n_gpus > 1:
        launch = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={n_gpus}",
            str(ROOT / "scripts" / "training.py"),
            "--config", tmp_config,
        ]
    else:
        launch = [
            sys.executable,
            str(ROOT / "scripts" / "training.py"),
            "--config", tmp_config,
        ]

    if args.resume:
        launch += ["--resume", args.resume]

    try:
        subprocess.run(launch, check=True)
    finally:
        Path(tmp_config).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
