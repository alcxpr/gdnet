# CAM pos_gate Retrieval Experiment

## Setup

- Model: ChunkEncoder (GDLayer, n_layers=1, n_cycles=2, kernel_size=7, d=32) + pluggable CAM
- Receptive field: n_cycles x kernel_size x n_layers = 14 tokens, covers chunk_size=16
- n_slots=8, d_sig=8, d_c=8
- Three conditions, each with n_slots+1=9 chunks per sample:
  - C1 unique key: signal written once at oldest slot, query must retrieve it
  - C2 ambiguous key: same key written at two different recencies, query must prefer newer
  - C3 missing key: key never written, tests implicit not-found behavior
- Baseline: ContentCAM - softmax over raw tag similarity, no position signal
- Proposed: PosGateCAM - separate content and position similarity terms

## Mechanism

Tags and queries use a shared W_tag applied to fwd[:, -1, :] (last forward token, full RF coverage).
Values use W_c applied to side[0].mean over the chunk.

Retrieval:

    sim_content = einsum("bd,bsd->bs", q, buffer_tags)
    sim_pos     = einsum("bd,sd->bs",  gamma, e)
    w           = softmax(sim_content + alpha * sim_pos)

where gamma = sigmoid(W_pos @ q) is an input-conditioned position query,
e = W_slot(slot_ids) are learned slot embeddings with geometric decay init (decay=0.85),
and alpha = exp(rho) is a learned log-scale.

W_pos initialized to zero so gamma starts at 0.5, unbiased, no initial recency push.
W_slot geometric decay gives a soft recency prior that alpha can scale up or down.

## Results

| condition | baseline acc | pos_gate acc | notes                               |
|-----------|--------------|--------------|-------------------------------------|
| C1        | 1.000        | 1.000        | both solve by epoch 10-20           |
| C2        | ~0.48        | 1.000        | baseline coin-flips; pos_gate exact |
| C3 H      | ~1.9         | ~0.13        | see below                           |

pos_gate solves C2 perfectly from epoch 10. Baseline plateaus at ~0.48 -- effectively
random between the two same-content slots since it has no position signal.

## Issues Encountered and Fixes

**H=2.08 (uniform retrieval) for all conditions, acc stuck at chance.**
Root causes: (1) write used side[0].mean(dim=1) while read used fwd[:, -1, :] --
different inputs to W_tag, making tag-query matching structurally hard. (2) tags were
detached at write time, cutting W_tag gradient from the write side entirely.
Fix: use fwd[:, -1, :] for tags on both write and read (same W_tag, same position,
both chunks have the key token within RF). Remove tag detach so gradient flows through
both query and key paths.

**val detach limiting learning.**
W_c could not teach the encoder to write retrievable summaries end-to-end.
Fix: drop val detach. With the batched write, the full graph is already materialized
across all n_slots chunks; keeping the detach was just cutting gradient for free.
Effect: C1 jumped from ~0.34 to 1.000, convergence went from 40+ epochs to 10.

**Sequential chunk loop -- 8 Triton launches per batch step.**
At chunk_size=16, d=32, launch overhead dominated compute (~11s/epoch).
Fix: since the encoder reinitializes the side stream per chunk (stateless in this
experiment), all n_write chunks can be encoded in one batched call (B*n_write, T, d),
then flipped to assign slot ordering. Reduced to ~2.6s/epoch (~4x speedup).
Note: this optimization is experiment-specific. In LM the side stream carries over
between chunks so sequential processing is mandatory there.

**Previous pos_gate formulation: biased = buffer_tags + alpha * e * gamma.**
Position bias was mixed into the tag before the dot product with q, contaminating
content similarity. gamma collapsed to 1 (always position-biased) making retrieval
unconditionally recency-biased. C3 H collapsed to ~0 -- model retrieved slot 0
confidently even for missing keys.
Fix: separate the two terms. sim_content and sim_pos are computed independently and
added before softmax. Content similarity is unaffected by position; alpha is the sole
trade-off knob. With W_pos=0 init, the model starts with a pure content retrieval
and learns to dial in position bias only where useful.

## Remaining Limitations

**R does not differentiate by condition.** R ~= 0.43 for C1, C2, and C3 across both
models. The recovery gate learns a fixed operating point optimized for C1/C2 and C3
is dragged along. R sees [side, fwd, retrieved_e] but not the retrieval weights w,
so it cannot distinguish a confident correct retrieval from a confident wrong one.
Feeding max(w) or entropy(w) into R would be needed for a structural not-found signal.

**C3 H ~0.13 for pos_gate.** The recency prior (W_slot + alpha) concentrates on slot 0
even when no content matches. This is the inherent cost of having a position bias -
useful for disambiguation, overconfident for missing keys. Acceptable given the C2 gain.
