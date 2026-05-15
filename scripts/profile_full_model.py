import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time

import torch
import torch.profiler

from gdnet.model import GDNet

B, T = 2, 128
vocab_size = 512


def make_model(dtype):
    return (
        GDNet(
            vocab_size=vocab_size,
            d_embed=64,
            d=256,
            n_layers=4,
            n_cycles=2,
            chunk_size=T,
        )
        .cuda()
        .to(dtype)
    )


tokens = torch.randint(0, vocab_size, (B, T), device="cuda")  # type: ignore


def bench(model, n_warmup=5, n_steps=10):
    def step():
        logits, side, _, _, _, _ = model(tokens)
        logits.sum().backward()

    for _ in range(n_warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_steps * 1000


def profile(model, n_steps=3):
    def step():
        logits, side, _, _, _, _ = model(tokens)
        logits.sum().backward()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(n_steps):
            step()
    return prof


model = make_model(torch.float32)
ms = bench(model)
print(f"\nfloat32  {ms:.2f} ms/iter")
print("-" * 60)
prof = profile(model)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
del model
torch.cuda.empty_cache()
