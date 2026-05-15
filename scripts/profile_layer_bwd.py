import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from gdnet.layer import GDLayer

B, T, d, k = 4, 512, 512, 7

torch.manual_seed(0)
layer = GDLayer(d, k).cuda().float()
fwd  = torch.randn(B, T, d, device="cuda", requires_grad=True)
side = torch.randn(B, T, d, device="cuda", requires_grad=True)

for _ in range(10):
    fwd2, side2 = layer.fwd_step(fwd, side)
    (fwd2.sum() + side2.sum()).backward()
torch.cuda.synchronize()

for _ in range(3):
    fwd2, side2 = layer.fwd_step(fwd, side)
    (fwd2.sum() + side2.sum()).backward()
torch.cuda.synchronize()
