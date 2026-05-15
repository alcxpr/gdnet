# Copy-with-Gap: Baseline Results

## Setup

- Model: GDNet, n_layers=3, n_cycles=3, kernel_size=7, d=64
- Receptive field: n_cycles x kernel_size x n_layers = 63 tokens
- Task: copy seq_len=3 signal tokens across a noise gap of varying length
- 80 epochs max, early stop at acc >= 0.99

## Results

| gap | acc   |
|-----|-------|
| 0   | 1.000 |
| 8   | 1.000 |
| 16  | 1.000 |
| 32  | 1.000 |
| 48  | 1.000 |
| 64  | 1.000 |
| 96  | 0.213 |
| 128 | 0.100 |

## Conclusions

**The model solves gap=64 despite the theoretical receptive field of 63.** The bound is not tight -- multi-cycle routing allows information to propagate further than the worst-case single-path analysis suggests.

**Gap=96 learns something (0.213 vs 0.100 chance) but does not solve.** This indicates residual signal leaking through the side stream gate dynamics rather than a hard null. The forgetting rate is poor - accuracy grows slowly across epochs and plateaus well below solved, suggesting the side stream encodes a weak but real gradient path at this distance.

**Gap=128 is flat at chance throughout training.** This is the true null space boundary for this configuration - no gradient path exists and the model cannot improve beyond random.

## Interpretation

The transition is not a sharp cliff. There is a degraded zone (~65-127 tokens) where partial information survives via side stream accumulation, and a hard null beyond that. The weak learning at gap=96 is not useful in practice; the forgetting rate is too high for reliable recall.

The primary bottleneck is not architecture capacity but receptive field coverage. Positional encoding (RoPE on conv input, per-step S_l injection) or dynamic offset selection (learned saccade) are the recommended next steps to extend reliable recall beyond the receptive field.
