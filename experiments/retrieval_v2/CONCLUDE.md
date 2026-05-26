# Retrieval v2 - Entropy-Conditioned Recovery Gate

## Setup

- Model: ChunkEncoder (GDLayer, n_layers=1, n_cycles=2, kernel_size=7, d=32) + CAM
- n_slots=8, d_sig=8, d_c=8, chunk_size=16
- Three conditions, each with n_slots+1=9 chunks per sample:
  - C1 unique key: signal written once; query must retrieve from buffer
  - C2 ambiguous key: same key at two recencies; query must prefer newer
  - C3 missing key: key never written; answer embedded in query chunk itself
- All three conditions mixed equally in training (N/3 each)
- Single model, no baseline comparison

## Change from v1

v1 R input: [side, fwd, retrieved_e]  (d*3)
v2 R input: [side, fwd, retrieved_e, h]  (d*3 + 1)

where h = H(w) / log(n_slots) is the normalised retrieval weight entropy, scalar per token.
H=0 means the model committed to a single slot (confident retrieval).
H=1 means weight is uniform across slots (nothing matched).

C3 task redesigned: the answer is embedded in the query chunk at position 2 (same
structure as a signal chunk but with QUERY_TOKEN). The model can only get C3 right
by reading from fwd directly, which requires R to stay low and not blend in noise
from the buffer. This gives C3 a real gradient signal rather than treating it as
eval-only.

## Results

| condition | acc   | R (ep 60) | H (ep 60) |
|-----------|-------|-----------|-----------|
| C1        | 1.000 | 0.402     | 0.08      |
| C2        | 1.000 | 0.403     | 0.10      |
| C3        | 1.000 | 0.393     | 0.16      |

All three conditions solved from epoch 10. C3 R stays slightly below C1/C2 throughout,
and C3 H remains higher (0.16 vs 0.08-0.10), confirming the model is more uncertain
about slot selection on missing-key inputs and compensates by opening R less.

## Remaining Limitations

**R separation between conditions is modest.** C3 R (~0.39) vs C1/C2 R (~0.40) is a
real but small gap. The model solves C3 by reading from fwd, but not by strongly
suppressing retrieval -- it works despite a partially open gate rather than because
of a closed one. A stronger C3 signal (e.g. adversarial buffer content that actively
corrupts fwd if R stays open) would force sharper separation.

**H does not collapse to 0 for C1/C2.** Entropy stays around 0.08-0.10 even for
confident correct retrievals. The retrieval weight does not become fully peaked on
the correct slot, suggesting W_tag / W_slot have residual ambiguity at this scale.
May improve with larger d_sig.
