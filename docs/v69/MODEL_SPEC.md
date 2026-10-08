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

## 9. Qualification findings (MEASURED; evidence in `docs/v69/evidence/qualification/`)

Final qualification (`qual/qual_summary.json`, run under `run_limited.ps1`: 8 GiB host, 3,072 MiB
device ceiling, exit 0, no orphans) qualified A, B and C on all 15 checks; every failed attempt
that preceded it is preserved. Honest account of how the suite reached its final form (no scientific
fit existed at any point):
1. Attempt 1: stack overflow on the 1 MiB Windows main thread inside Burn checkpoint recording
   → binaries now run on a 512 MiB worker thread.
2. Attempt 2 (arm A, original checks) exposed four problems: (a) a real defect — `clamp_min`
   has a one-sided gradient at exactly z = 0, so BCE gradients were off by 1/2 at that single point;
   fixed (`max(z,0)` written as `(z+|z|)/2`; same loss function); (b) per-tensor relative accumulation
   error was dominated by tensors whose true gradient is ~0 (attention key biases: softmax is
   shift-invariant) — check redesigned to scale by `max(‖tensor‖, 1e-3·global norm)`; (c) the
   credit-assignment control skipped tensors that lose all gradient when early calls are cut — those
   now count as zero, and the expectation is block-specific (a block called only after the cut must
   not change); (d) fp32 finite differences of the GPU loss were useless because of the finding below —
   replaced by exact f64 finite differences of an independent host reference model.
3. **Precision finding (not removable):** Burn 0.21.0 / cubek-matmul 0.2.0 autotune may execute f32
   matmuls on **TF32 tensor cores** (`adjust_dtypes`: f32 inputs are staged as tf32 when the device
   supports it and the selected kernel is accelerated). This GPU supports it. The graph is therefore
   "CUDA, f32 storage and accumulation, matmul inputs possibly TF32" — **not validated as strict FP32**.
   Measured against the independent f64 host reference over the 16-example panel: max |logit error|
   5.7e-4 (A), 9.3e-4 (B), 9.1e-4 (C) at logit scales 0.02–0.3 (tolerance 2e-3), i.e. TF32-class,
   not 1e-6-class. Gradients agree with exact f64 finite differences to ≤ 1.2e-3 relative (typically
   1e-4). Forcing strict FP32 matmuls would require custom kernels or a different backend, which
   this task forbids; the recipe was therefore left unchanged. Kernel selection is by timing, so
   bit-exact reproducibility across processes is not claimed.
Final measured values: gradient accumulation 8×2 vs batch-16 agree to ≤ 1.2e-6 (error/scale), global
norms to ≤ 1e-8; checkpoint continued-training disagreement 0.0 (negative control with fresh optimizer:
3.9e-4); normal vs per-call-synchronized execution identical (0.0) — "profile" parity is interpreted as
this synchronization mode (no Burn profiler API was used); zero-valued gradient probes are nonzero for
every workspace state and every board state except after the final call (exactly zero, as the head reads
only the workspace); cutting the unroll changes the shared fast/slow gradients (B: 1.24 / 0.93,
C: 0.86 / 1.03 relative); cutting after call 1 in A changes fast (1.0) but not slow (0.0), as required.
Latency (synchronized, medians, qualification weights): inference batch 2 / 16 — A 5.7 / 8.1 ms,
B 21.7 / 28.6 ms, C 22.8 / 25.5 ms; full update (8 microbatches + clip + AdamW) — A 241 ms,
B 868 ms, C 859 ms. B and C block-call counts are equal (12) and their measured update latency agrees
within ~1%; no FLOP estimate is made. Memory: sampled device memory (whole GPU, nvidia-smi, 100 ms)
peaked at 771 MiB for A alone and 2,403 MiB after B/C ran in the same process (allocator pool growth);
host working set 284–291 MiB point samples, launcher peak 2,683 MiB (includes the f64 reference);
ceiling 3,072 MiB device / 8 GiB host respected. Allocator high-water values are not available through
this interface (limitation).
Weight decay note: at the scheduled learning rates the decoupled decay factor (1 − lr·1e-4 ≤ 5e-8)
is near f32 resolution; the zero-gradient lr = 1 test shows the groups behave exactly as specified.
