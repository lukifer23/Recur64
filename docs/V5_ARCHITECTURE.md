# V5 architecture: `counterfactual_relational_loop_v1`

Status: **FROZEN BEFORE IMPLEMENTATION.** Every named contract below is included
in the scientific configuration digest.

## Geometry

- width 256; 8 heads; FFN 768 with GELU; dropout 0; RMSNorm epsilon 1e-5
- root square encoder: four transformer blocks
- root candidate interaction: one transformer block
- returned-state encoder: two blocks shared across queried nodes
- returned-state compression: four learned attention-pooling slots
- relational core: one evidence block and one hypothesis block, distinct from
  each other and shared across all R iterations
- baseline and correction readout hidden width: 256
- residual scale alpha: 0.5

The implemented frozen graph contains **7,160,080 unique parameters**:
3,677,728 root/baseline; 1,633,808 returned-state encoder; 65,792 hypothesis
adapter; 136,960 evidence initializer; 789,872 shared evidence block; 789,872
shared hypothesis block; and 66,048 correction readout. No parameters were added
cosmetically. The parameter count is independent of Q and R.

For batch B, legal-candidate padded width W, and acquired-node padded width N,
the frozen tensor shapes are: root squares C `[B,64,256]`, root hypotheses H0
`[B,W,256]`, returned observations `[B,N,64,119]`, returned flags `[B,N,9]`,
payload anchors X and mutable evidence E `[B,4N,256]`, mutable hypotheses H
`[B,W,256]`, and candidate logits `[B,W]`. Evidence attention is
`[B,8,4N,4N+W+64]`; hypothesis attention is `[B,8,W,W+4N+64]`.
Legal-candidate and acquired-node masks exclude padding from memory attention,
candidate centering, normalization denominators and loss. These are dense
attention tensors; no custom sparse kernel or legal-candidate truncation exists.

## Root and baseline

`v5_root_frame_v1` encodes every current/history board in the root player's
physical frame. Piece ownership and castling are root-relative; en-passant and
actions are transformed through physical squares into that same frame. The
root/opponent turn role is explicit context.

The root observation is projected to 64 tokens, receives learned square identity
and 15x15 relative-displacement attention bias, and is encoded once. A legal move
hypothesis concatenates its from/to root tokens, pooled root context, normalized
from/to/delta geometry plus promotion one-hot (11 values), and the eight root
`CandidateFactsV1` values. One candidate block yields immutable `H0`.

The baseline MLP reads `[H0_i, pooled_root]` and produces `z0_i`. Q0 returns z0
directly: one root encoding and no state encoder, query, or relational block.
Stage B computes the complete baseline graph-free and lifts it as constants.

## Payload and graph

`v5_returned_payload_v1` contains only a queried non-root state's root-relative
ObservationV1 and payload flags: in-check, terminal, and the seven Rules Profile
termination reasons. Its state encoder is run once per forward. Four learned
queries pool the 64 square tokens; the flags are projected into the four outputs.
The four-token result is immutable anchor `X_e`.

Nulling occurs at this boundary: every anchor token and returned flag contribution
is exactly zero. Structure, path actions, turn roles, node existence and routing
remain matched. Thus the intervention is conditional on the acquired structure.
The composition diagnostic uses the same encoded-anchor boundary with an explicit
per-node four-slot mask; it never attempts to obtain a null anchor by passing a
zero raw observation through biased encoder layers.

`v5_acquired_graph_v1` is a path-specific tree with no transposition merging. It
records parent/child direction, siblings, four-slot membership, root-branch
ownership, turn role and depth 1..5. Node numbers and acquisition order are storage
only. No unqueried child state enters the reader.

## Shared relational loop

The evidence initializer is the same function for both streams and reads
`[X, structural context]`. The hypothesis initializer is a reasoner-only adapter
of H0. There is no iteration embedding, countdown, halting head, R-specific tensor,
intermediate loss, teacher forcing, or truncated gradient.

For evidence and then hypotheses at every iteration:

```text
U = RecallProjection(concat(RMSNorm(S_r), immutable_anchor))
A = MultiHeadAttention(U, normalized declared memory, relation_bias, mask)
T = S_r + 0.5 * A
S_(r+1) = RMSNorm(T + 0.5 * FFN(RMSNorm(T)))
```

Evidence memory is current evidence, current hypotheses and immutable root square
context. Hypothesis memory is hypotheses, newly updated evidence and root context.
The next evidence pass receives the previous hypothesis output.

Per-head additive relation features distinguish self/same node, parent, child,
sibling, same branch, different branch, evidence owned by a hypothesis, competing
hypotheses and common root context. Consistently permuting storage and remapping
these relations must preserve results.

## Paired readout

The factual and all-payload-null streams use the same parameters and cannot attend
to each other. A single correction MLP gives

```text
raw_delta_i = readout(H_real_R)_i - readout(H_null_R)_i
centered_delta_i = raw_delta_i - mean(raw_delta over valid legal candidates)
z_i = z0_i + centered_delta_i
```

Padding never enters attention, centering, softmax, or loss denominators. Both
streams remain in the autodiff graph. Identical null inputs must produce exact CPU
equality; CUDA qualification starts with maximum centered-logit error 1e-6.

Evaluation tracing executes the same reader computation and records, for factual
and null streams at every loop, valid-token state RMS, update RMS, attention
entropy and mean maximum attention weight. A parity test requires traced and
ordinary logits to be exactly equal on deterministic CPU execution.

The implementation emits separate full-payload and structure-only graph digests.
Payload shuffle changes the former while the latter must remain fixed. Shuffle
donor position/path mappings and deterministic composition partitions are stored
with per-position evidence.

## Versioned subcontracts

`v5_root_frame_v1`, `v5_root_hypotheses_v1`, `v5_returned_payload_v1`,
`v5_state_tokens4_v1`, `v5_acquired_graph_v1`, `v5_relational_loop_v1`,
`v5_input_recall_v1`, `v5_paired_null_readout_v1`,
`v5_correct_set_loss_v1`, `v5_fixed_graph_reader_pilot_v1`.
