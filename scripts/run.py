"""Single entry point: prepare data then launch training.

Prepares the first phase's data, starts training immediately, then tokenizes
remaining phases in a background thread. Training waits at each phase boundary
using inotify (watchfiles) -- no polling loops.

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
import signal
import subprocess
import sys
import tempfile
import threading
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


def _prepare_phase(phase: dict, encoding: str) -> None:
    path = phase.get("path")
    if not path:
        return
    if Path(path).exists():
        size_gb = Path(path).stat().st_size / 1024**3
        print(f"[data] found {path} ({size_gb:.2f} GB)", flush=True)
        return
    source = phase.get("dataset", "fineweb-edu")
    if source not in ("fineweb-edu", "nemotron"):
        return
    print(f"[data] preparing {path} ...", flush=True)
    subprocess.run(_prepare_cmd(phase, path, encoding), check=True)


def _prepare_remaining(phases: list[dict], encoding: str) -> None:
    seen: set[str] = set()
    for phase in phases:
        path = phase.get("path") or ""
        if path in seen:
            continue
        seen.add(path)
        try:
            _prepare_phase(phase, encoding)
        except Exception as e:
            print(f"[data] ERROR preparing {path}: {e}", flush=True)


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

    for phase in phases:
        if not phase.get("path"):
            phase["path"] = _derive_path(phase, data_dir)

    if args.skip_prepare:
        remaining_phases: list[dict] = []
    else:
        # Deduplicate by path so we don't double-prepare shared files
        seen: set[str] = set()
        unique_phases: list[dict] = []
        for phase in phases:
            path = phase.get("path") or ""
            if path not in seen:
                seen.add(path)
                unique_phases.append(phase)

        # Prepare the first phase synchronously so training can start immediately
        _prepare_phase(unique_phases[0], encoding)
        remaining_phases = unique_phases[1:]

    if args.prepare_only:
        _prepare_remaining(remaining_phases, encoding)
        print("[run] --prepare-only: all data ready, exiting.")
        return

    # Write temp config with paths filled in
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=ROOT
    ) as tmp:
        yaml.dump(cfg, tmp)
        tmp_config = tmp.name

    n_gpus = args.gpus or torch.cuda.device_count()
    n_gpus = max(n_gpus, 1)
    print(f"[run] launching training on {n_gpus} GPU(s)", flush=True)

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

    training_proc: subprocess.Popen | None = None
    try:
        training_proc = subprocess.Popen(launch)

        # Prepare remaining phases in background while training runs
        if remaining_phases:
            t = threading.Thread(
                target=_prepare_remaining,
                args=(remaining_phases, encoding),
                daemon=True,
            )
            t.start()

        def _on_signal(sig, frame):
            if training_proc and training_proc.poll() is None:
                training_proc.terminate()
            sys.exit(1)

        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)

        rc = training_proc.wait()
        if rc != 0:
            sys.exit(rc)
    finally:
        Path(tmp_config).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
