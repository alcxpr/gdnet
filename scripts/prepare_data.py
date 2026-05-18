"""Offline tokenization: stream a HF dataset, tokenize, write flat uint32 binary.

The output file is a raw numpy uint32 array that PackedTokenDataset mmaps at
training time -- zero tokenization overhead, just disk reads.

Documents are batched and tokenized in parallel across all available CPUs.
tiktoken is a Rust extension that releases the GIL, so ProcessPoolExecutor
scales linearly up to the number of CPU cores.

FineWeb-Edu (one score bucket):
    uv run python scripts/prepare_data.py \\
        --dataset fineweb-edu --subset sample-10BT \\
        --min-score 5 --max-score 5 \\
        --out /data/tokenized/fineweb_edu_5.bin

FineWeb-Edu (medium quality):
    uv run python scripts/prepare_data.py \\
        --dataset fineweb-edu --subset sample-10BT \\
        --min-score 3 --max-score 4 \\
        --out /data/tokenized/fineweb_edu_34.bin

Nemotron (all subsets):
    uv run python scripts/prepare_data.py \\
        --dataset nemotron --subset everything \\
        --out /data/tokenized/nemotron.bin
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

_DOCS_PER_BATCH = 512


def _tokenize_batch(args: tuple[list[str], str, int]) -> list[int]:
    texts, encoding_name, eos = args
    import tiktoken
    enc = tiktoken.get_encoding(encoding_name)
    out: list[int] = []
    for text in texts:
        out.extend(enc.encode_ordinary(text))
        out.append(eos)
    return out


def _chunk(it, size: int):
    it = iter(it)
    while batch := list(itertools.islice(it, size)):
        yield batch


def _stream_fineweb(subset: str, min_score: int, max_score: int, min_token_count: int):
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=subset,
        split="train",
        streaming=True,
    )
    for sample in ds:
        if not (min_score <= sample["int_score"] <= max_score):
            continue
        if sample["token_count"] < min_token_count:
            continue
        yield sample["text"]


def _stream_nemotron(subset: str, min_chars: int):
    from datasets import load_dataset  # type: ignore

    repo = "nvidia/Nemotron-Pretraining-Specialized-v1.1"
    if subset == "everything":
        from datasets import concatenate_datasets, get_dataset_config_names  # type: ignore

        configs = get_dataset_config_names(repo)
        ds = concatenate_datasets(
            [load_dataset(repo, name=c, split="train", streaming=True) for c in configs]
        )
    else:
        ds = load_dataset(repo, name=subset, split="train", streaming=True)

    for sample in ds:
        text = sample["text"]
        if min_chars > 0 and len(text) < min_chars:
            continue
        yield text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["fineweb-edu", "nemotron"], required=True)
    parser.add_argument("--subset", default="sample-10BT")
    parser.add_argument("--min-score", type=int, default=3)
    parser.add_argument("--max-score", type=int, default=5)
    parser.add_argument("--min-token-count", type=int, default=128)
    parser.add_argument("--min-chars", type=int, default=0)
    parser.add_argument("--encoding", default="cl100k_base")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 2),
        help="tokenizer worker processes (default: cpu_count - 2)",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Need eos token but can't send the full encoder to workers (large pickle).
    # Get it on the main process and pass as a plain int.
    import tiktoken
    eos = tiktoken.get_encoding(args.encoding).eot_token

    if args.dataset == "fineweb-edu":
        text_iter = _stream_fineweb(
            args.subset, args.min_score, args.max_score, args.min_token_count
        )
    else:
        text_iter = _stream_nemotron(args.subset, args.min_chars)

    total_tokens = 0
    t0 = time.perf_counter()
    max_pending = args.workers * 4

    print(f"workers={args.workers}  encoding={args.encoding}  out={out_path}")

    with open(out_path, "wb") as f, ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending: list = []

        def _flush_ready() -> None:
            nonlocal total_tokens
            # Drain completed futures in submission order to keep output deterministic.
            while pending and pending[0].done():
                tokens = pending.pop(0).result()
                arr = np.array(tokens, dtype=np.uint32)
                f.write(arr.tobytes())
                total_tokens += len(tokens)
                elapsed = time.perf_counter() - t0
                print(
                    f"\r{total_tokens / 1e9:.3f}B tokens  "
                    f"{total_tokens / elapsed / 1e6:.1f}M tok/s  "
                    f"pending={len(pending)}",
                    end="",
                    flush=True,
                )

        def _flush_one() -> None:
            # Block until the oldest pending future finishes.
            if pending:
                tokens = pending.pop(0).result()
                nonlocal total_tokens
                arr = np.array(tokens, dtype=np.uint32)
                f.write(arr.tobytes())
                total_tokens += len(tokens)

        for batch in _chunk(text_iter, _DOCS_PER_BATCH):
            # Back-pressure: if the queue is full, wait for the oldest batch.
            if len(pending) >= max_pending:
                _flush_one()

            pending.append(
                pool.submit(_tokenize_batch, (batch, args.encoding, eos))
            )
            _flush_ready()

        # Drain remaining futures in order.
        for fut in pending:
            tokens = fut.result()
            arr = np.array(tokens, dtype=np.uint32)
            f.write(arr.tobytes())
            total_tokens += len(tokens)

    elapsed = time.perf_counter() - t0
    size_gb = out_path.stat().st_size / 1024**3
    print(
        f"\ndone: {total_tokens / 1e9:.3f}B tokens  "
        f"{size_gb:.2f} GB  "
        f"{elapsed / 60:.1f} min  "
        f"-> {out_path}"
    )


if __name__ == "__main__":
    main()
