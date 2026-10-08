# Recur64 V69 — executable model / training / evaluation specification

Companion to `CONTRACT.md` §9–§11 (which it refines, never contradicts) and
`DATA_PHASE_RESULTS.md`. Every item below is implemented in
`crates/recur64-v69-model` / `crates/recur64-v69` and was written before any
scientific fit. Owner decision: accept the ≈1.88 M model; no layer or width was
added to reach a parameter target.

Task: binary forced-mate classification of an immediate nonterminal child (defender to
move) given the remaining attacker-move budget. Not game-outcome prediction.

## 1. Inputs (`recur64_v69::features`)

- Every example is defender-to-move. **Attacker = the side not to move** (derived from
  this contract, never from root metadata). Validated per example; anything else fails
  visibly: FEN must have 6 fields and parse; defender has a lone king; attacker has a
  king plus exactly {2Q | Q+R | 2R}; no pawns/minors; halfmove ∈ {0,1}; no castling
  rights / en-passant; not terminal; budget ∈ {1,2}; rows reject unknown JSON fields and
  duplicate ids.
- Piece codes (13): 0 empty; 1–6 attacker P,N,B,R,Q,K; 7–12 defender P,N,B,R,Q,K
  (ownership is attacker/defender-relative, not White/Black).
- Orientation: if the attacker is Black the board is rank-flipped so the attacker always
  "plays up". This is a true symmetry of the pawnless domain; tests show a White-attacker
  position and its colour-swapped/flipped Black-attacker image give identical features.
  Square index = rank*8+file after the flip (a1 = 0).
- 8 rule scalars: `[attacker_to_move, defender_to_move, halfmove/100, att O-O, att O-O-O,
  def O-O, def O-O-O, en-passant]` (castling/ep are always 0 in this domain).
- Remaining budget: embedding index 1 or 2 (table has 4 rows; rows 0 and 3 unused).
- Attacker-relative-to-mover indicator: embedding index `attacker_to_move` (always 0 here;
  carries no information on this dataset).
- Not inputs: ids, row order, labels, root boards, root-depth, family labels, oracle data.
  `featurize(fen, budget)` has no label/id parameter; tests mutate labels/ids and observe
  identical features and predictions.

## 2. Architecture (D=192, 6 heads (dim 32), FFN 576, 64 board tokens, 8 workspace tokens)

- Attention (all variants): Q,K,V,O linear with bias; scale `1/sqrt(32)`; softmax over keys;
  no mask, no dropout (zero dropout everywhere). FFN: `Linear(192,576) → GELU(erf) → Linear(576,192)`.
  LayerNorm ε = 1e-5, learned γ/β. **Pre-norm residual sublayers with fixed scale 0.1:**
  `x ← x + 0.1·Sublayer(LN(x))` for every sublayer (encoder and recurrent blocks).
- Board tokens: `piece_emb[code] + square_emb[sq] + Linear(8→192)(scalars) + budget_emb[b] +
  att_emb[a]`; two encoder blocks (self-attention + FFN); `E = LN_enc(·)` ([b,64,192]).
- Workspace init: `slot_emb[8,192] + Linear(192→192)(LN(mean_over_tokens(E)))` broadcast.
- **Fast block** (parameters shared across its calls; updates the board): LN → self-attention
  whose queries are `LN(board)` and whose keys/values are over the concatenation
  `[LN(board); LN(E)]` (the frozen encoded-board refresh: E is the fixed encoder output,
  re-read at every fast call, read-only) → board-to-workspace attention (`LN_q(board)` queries,
  `LN_kv(workspace)` keys/values) → FFN. 
- **Slow block** (shared; updates the workspace): LN → workspace self-attention (keys over the
  workspace only) → workspace-to-board attention (`LN_kv(board)`) → FFN.
- Head: `mean over 8 workspace tokens → LayerNorm → Linear(192,192) → GELU → Linear(192,1)`.
- State (board = E, workspace init) is created from the example's own inputs at every
  forward; there is no carry across calls to forward, no detach, no adaptive halting, no
  equilibrium-gradient approximation. Full back-propagation through the unroll.

Arms (executable schedules, `model::Arm::schedule`):
- A: F S → read. 2 block calls, 1 readout.
- B: (F F S → read if cycle∈{1,2,4}) × 4 cycles. 12 calls, 3 readouts.
- C: (F S → read if pair∈{2,4,6}) × 6 pairs. 12 calls, 3 readouts.
B and C have equal block-call counts, **not** assumed equal FLOPs/latency (measured in qualification).

## 3. Initialization (reserved `model_init` stream, written once)

Host-generated from SplitMix64 + Box–Muller in traversal order (Burn's RNG unused):
Linear weight `[in,out] ~ N(0, 2/(in+out))`; embedding tables `~ N(0, 0.02²)`; Linear bias 0;
LayerNorm γ=1, β=0. File `init/canonical_init.bin` (JSON header + raw f32 LE), SHA-256 and
tensor hash recorded in `spec/param_inventory.json`. Every arm loads this exact file; the
hash of each arm's loaded tensors is checked against it. Qualification uses a different,
disposable stream (`qualification_init`) and never initializes a scientific fit.

**Parameters (MEASURED, 107 tensors, ceiling 4 M):** total **1,876,417** = piece_emb 2,496 +
square_emb 12,288 + scalar_proj 1,728 + budget_emb 768 + att_emb 384 + enc0 370,944 + enc1
370,944 + enc_ln 384 + slot_emb 1,536 + ws_pool_ln 384 + ws_pool_proj 37,056 + fast 519,936 +
slow 519,936 + head_ln 384 + head1 37,056 + head2 193. (CONTRACT estimate 1,876,801 assumed a
second final norm that the architecture does not contain; difference 384.) Weight-decay
group = the 41 rank ≥ 2 tensors; no-decay = the 66 bias/LayerNorm tensors. Full inventory:
`spec/param_inventory.json`.

## 4. Loss

Stable BCE-with-logits `max(z,0) − z·y + ln(1+exp(−|z|))`. Per readout: mean over the
examples in the microbatch; loss = mean over the arm's prescribed readouts. Final-readout
BCE is reported separately. Evaluation: f64 aggregation of serialized logits.

## 5. Training equations (update u = 0…599; eight accumulation steps of microbatch 2)

```
g_u   = Σ_{m=0..7} ∇( L_m / 8 )          # L_m: mean over 2 examples of mean-over-readouts BCE
n_u   = ‖g_u‖₂ over ALL parameters       # global, computed once, after accumulation
g_u  ← g_u · min(1, 1/(n_u + 1e-6))      # applied only when n_u + 1e-6 > 1; else untouched
θ    ← AdamW(lr_u, β=(0.9,0.999), ε=1e-8)(g_u)   # decoupled decay: θ ← θ(1 − lr_u·wd) − lr_u·m̂/(√v̂+ε)
```
Two AdamW instances (independent moments) implement the groups: decay group wd = 1e-4
(rank ≥ 2 tensors), no-decay group wd = 0 (biases, LayerNorm γ/β). Per-update decay at the
scheduled LR is ≤ 5e-8 relative, i.e. near f32 resolution (applied as AdamW prescribes; an
honest limitation of the frozen recipe, not a bug). Adam bias correction uses Burn's per-tensor
step counter.
LR (0-based update index u): `u<20: 5e-4·(u+1)/20`; `u≥20: 5e-5 + ½(5e-4−5e-5)(1+cos(π·(u−20)/579))`
→ update 19 = 5e-4, update 599 = 5e-5 exactly. The schedule position is the completed-update
count (saved in checkpoints).
Data order: epoch e = Fisher–Yates permutation of the 768 fit rows driven by
`train_order` stream index e; the sample stream is the concatenation of epochs (12.5 epochs =
9,600 samples); update u uses samples [16u,16u+16), microbatch m uses [2m,2m+2). Identical for
all arms; class-block serialization is not used. FP32 throughout; no validation during
training; no best-checkpoint selection; fixed update-600 endpoint.

## 6. Qualification (CUDA FP32; predefined tolerances in `qualify.rs`)

BCE value/gradient vs f64/analytic (|err| ≤ 1e-5, logits to ±88); readout mean; 8×2
accumulation == single batch-16 mean (per-tensor relative L2 ≤ 1e-4); clipping once on the
accumulated global gradient (post-clip norm = 1 ± 1e-4; untouched below 1; exactly one clip
operation per update); gradient inventory (every parameter tensor present, finite, nonzero);
**recurrent credit assignment**: zero-valued leaf probes added to board and workspace after
every block call, final-readout-only loss → every workspace probe and every board probe
except after the final call has nonzero gradient, the board after the final call has
exactly zero; negative control (detach after call m) zeroes exactly the probes ≤ m; shared
fast/slow gradients change > 1e-3 relative when early calls are cut; numeric central-difference
directional derivative (ε = 0.05, unit direction) vs analytic gradient through the unroll for
four tensors (≤ 5% + 2e-5); no state leakage (solo/reordered/repeated ≤ 1e-4 logits); label/id
mutation changes nothing; canonical-init identity across the three arms and optimizer
independence; weight-decay groups (zero-gradient step at lr=1: decay tensors scale by 1−1e-4
within 1e-6 relative, others bit-unchanged); checkpoint (weights, both optimizers' moments and step
counters, schedule position) restores and continued training agrees (≤ 1e-6 abs) with a
negative control (fresh optimizer) that must disagree; normal vs per-call-synchronized
("profile") execution parity (logits 1e-6, gradients 1e-5 relative); synchronized inference and
update latency; device memory (nvidia-smi sampled, ceiling 3,072 MiB) and host working set.
Limits: 30 min/arm, 8 GiB host. Any failure stops before scientific fitting.

## 7. Evaluation

Fixed update-600 endpoints; fit and validation partitions only. Reports: final-readout BCE,
balanced accuracy at logit 0, accuracy, Brier, confusion matrices, per-(family,budget,class)
metrics, all readouts, training traces/gradient norms/clip frequency, synchronized latency,
exposure counts, memory.
- **Derangement** (frozen before results, `intervention/map.json`, SHA-256 in
  `intervention/map.sha256`): per partition and (family, budget) cell, ids are sorted, shuffled by
  stream `intervention/<partition>/<family>/<budget>` and mapped to the next id cyclically
  (label-independent; no fixed points; bijection). A recipient is evaluated on the donor's
  complete board features (including board-derived rule features) with its own task budget
  (identical within a cell). Reports: accuracy / balanced accuracy vs recipient labels and vs donor
  labels, donor-map label-agreement rate, and verification that each deranged input's logits equal
  the donor's ordinary-input logits within 1e-3 (different batch composition).
- **Board erasure** (OOD diagnostic): all 64 squares set to empty; retained legitimate task
  metadata: remaining budget, attacker-to-move indicator and the constant rule scalars. It tests
  reliance on board content; it is not in-distribution evidence.
- **Uncertainty:** cluster bootstrap over connected `group_id`s (validation), 2,000 resamples,
  percentile 95% intervals, stream `bootstrap/<arm>` (domain-separated, frozen before results). Rows are not
  resampled independently.
- **Gates (unchanged):** fit balanced accuracy ≥ 95%; validation balanced accuracy ≥ 75%;
  validation final BCE ≤ 0.55; validation balanced accuracy (real) − balanced accuracy of
  deranged inputs against **recipient** labels ≥ 15 percentage points (balanced accuracy is
  used for "recipient-label accuracy"; plain accuracy is reported too); all engineering/custody
  gates pass.
- Independent aggregation: `v69-aggregate` recomputes every metric and gate in f64 from the
  serialized prediction rows and fit/val metadata with no shared code with the model crate;
  results are cross-checked against the in-process f32 summary.

## 8. Access policy and provenance

Roles (`recur64_v69::access`): learner = seed + `data/fit.jsonl`; evaluator = fit + val
model rows; metric aggregator = fit/val metadata + predictions; model code cannot reach
`sealed/`, `pool/` or root-bearing metadata (runtime refusal tested; a source scan test forbids
those paths in the model crate). Read-only file attributes are not treated as a barrier.
Every launch/checkpoint/prediction records the consumer source id (git head, dirty count,
digest of sources + lock), the distinct **data-producer** head
`36a81508b456ede1cb682f2f03fe678fd08db70f`, input hashes, spec/init/intervention hashes,
streams, arm schedule, completed updates. The sealed test is never opened by a model.
