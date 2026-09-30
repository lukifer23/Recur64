# HP X1 experiment ledger

Fast-falsification ledger for X15 / Chimera. Labels: MEASURED, INFERRED,
POST-HOC, NOT RUN. Each entry: QUESTION, PRE-REGISTERED RULE, CONFIG, WALL TIME,
MEASURED, INTERPRETATION, DECISION, NEXT ACTION.

Starting SHA for this phase: `c60103bd95b4f535b381f2f1cd985ea9e2998984`
(branch `experiment/hp-r15-h3-integration`, clean tree, remote in sync).

## Time policy
Fast falsification. Cells of minutes, no experiment over 60 minutes, about 2
hours cumulative GPU time, then stop and report.

## E0 - P0 source fixes (no GPU)
- QUESTION: is the checkout sound enough to run science on?
- MEASURED: all seven P0 findings confirmed and fixed (D60, D61). Workspace
  tests (arena tests skipped) and clippy pass; fmt clean.
- DECISION: proceed to CUDA qualification.

## E1 - first CUDA execution of X15 (P1.1 / P1.2)
- QUESTION: does the actual X15 graph run on the RTX 2050, with finite outputs
  and gradients?
- RULE (stated before running): finite outputs, legal policy mass 1, exactly
  neutral fresh WDL, non-zero finite gradient in every enabled subsystem, 0 CUDA
  errors, D57 device check passes.
- CONFIG: `configs/x15_cuda.toml` (native_v1, mate_depth 1, 64px visual), release
  build `--features cuda,fusion,autotune`, CUDA 12.9.1 user-space, driver 616.92.
- MEASURED:
  - `x15 sanity` T=1/2/4: PASS (repeated). Entropy about 3.41, WDL logit 0.
  - `x15 grads` T=4 batch 4: PASS, all 8 subsystems non-zero and finite.
    grad norms: symbolic 1.7e-2, core 1.5e-2, output 3.2e-3, latents 1.2e-2,
    compute 8.6e-4, visual 1.8e-3, heads 39.7 (synthetic uniform target), gates 9.5e-5.
  - `x15 thoughts` T=4 (DIAGNOSTIC forward): four rows. |Z| 0.87 -> 1.77,
    pathway magnitudes square 0.25 / compute 0.18 / visual 0.27, gates 0.500,
    consecutive KL 7.9e-4 -> 3.9e-4 -> 1.2e-4 (untrained).
- FAILURES (visible, not hidden):
  - CLI `sanity` read a device bool tensor with `to_vec::<bool>`; CUDA returns
    `Bool(U8)`. CLI bug, fixed (float readback). An unproven model-side change
    was reverted.
  - Two runs failed the D57 device check because NVRTC could not be loaded
    (kernels returned zeros; the check refused, as designed). Both happened
    while orphaned `nvidia-smi` sampler loops from my own earlier commands were
    running. Not reproduced in 6 later runs after those were killed. Cause
    UNPROVEN.
- INTERPRETATION: X15 genuinely executes on CUDA. Fresh pathway magnitudes are
  the same order of magnitude, so the ComputeBankV1 byte scale is not obviously
  swamping the trunk at init (measured, not tuned). NOT concluded: anything about
  learning.
- DECISION: proceed. Gate init and compute-feature scale left unchanged.
- COLD START (MEASURED): first-time kernel JIT/autotune per new shape costs
  minutes of CPU with the GPU mostly idle (fwd/bwd warm-up 172-202 s, forward
  93 s). Once per shape; steady state is fast. Timed cells must follow a warmup
  on the same shapes.

## E2 - latency and training layout (P1.3)
- QUESTION: what does T cost, and what training batch fits in 4 GB?
- RULE: choose the largest layout with peak VRAM <= ~3.2 GB at effective batch
  about 128.
- MEASURED, NORMAL forward (`x15 bench --mode infer`, batch 32, warm):
  T=1 91.5 ms (350 pos/s), T=2 108 ms (295 pos/s), T=4 142.5 ms (225 pos/s);
  peak VRAM 517 MiB. The diagnostic forward is NOT used for these numbers.
- MEASURED, train step (fwd+bwd+accum+AdamW, final_only, synthetic targets,
  cost not learning):
  - 16x2 (eff 32): T=2 365 ms/update 88 ex/s 1.23 GB; T=4 442 ms 72 ex/s 1.58 GB.
  - 32x4 (eff 128): T=4 1473 ms/update, 86.9 ex/s, peak 2379 MiB, GPU ~97% busy.
  - Loss falls on a fixed batch in every cell (e.g. 4.49 -> 4.16 at T=4).
  - T=1 train cells are NOT valid (autotune still settling in the timed steps).
- DECISION: physical 32 x accumulation 4 (effective 128). T=8 not run.
- NEXT: lifecycle plateau, owner-pool / D55 checks, then ReasoningTargetsV1.

## E3 - lifecycle (P1.4)
- MEASURED (`x15 bench --mode lifecycle`, D55 build, T=4, 5 build/forward/drop
  cycles): post-drop VRAM 483 MiB every cycle, growth 0 MiB. Plateau, no
  per-owner leak.
- NOT RUN: plain-CUDA (no fusion) lifecycle. The raw model has no owner thread;
  the owner-thread lifecycle (D44) is only relevant once X15 gets an inference
  owner (P7).

## E4 - ReasoningTargetsV1 generation (P2)
- QUESTION: can we build an exact, deterministic, self-teacher target set?
- CONTRACT (frozen before generating): teacher = Train1 final trainer
  (`runs/hp-r15-train1-r1/checkpoints/trainer`, model_id `073357a98882...`,
  0.633 vs M), probe_v1 architecture, recurrence 1; deterministic PUCT, c_puct
  1.0, 1 leaf in flight, NO root noise; ladder 16/32/64/128 simulations; seed
  20260929. Positions = exact `start_fen + action-id prefix` from that run's 288
  self-play games; no external engine, tablebase, book or human data.
- Selection: seeded hash order, equal quota over opening / middlegame / endgame /
  tactical (side to move in check) / material_advantage (|diff| >= 3); train/val
  split by source game (about a quarter of games in val).
- MEASURED:
  - 32-position smoke: 94 s single thread, 70 s at 4 threads; digest identical
    (`7b6bba18...`), so labels are independent of thread count.
  - 128 positions (25/26/26/26/25 by category; train 96 / val 32): 249 s at 6
    threads. Digest `8c9237ff6a0c...`. Every position reconstructs move for move
    (FEN, legal list and observation digest all reproduce).
  - RTX during labelling: peak 52% util, 43% busy-mean, 321 MiB. The GPU is used
    but latency-starved: batch-1 search is CPU-bound.
- Evidence: `docs/evidence/x1/reasoning-targets-v1-128.json`.
- INTERPRETATION: good enough to proceed. Labelling throughput is an
  engineering limit, not a science limit; 256/512 positions would need batched
  labelling.

## Outstanding (as of this entry)
1. Fixed-data trainer (`x15 train-probe`): checkpoint save/resume + optimizer
   state, supervision modes on ReasoningTargetsV1, value target. NOT BUILT.
2. Micro-overfit, then LR screen (3e-5 / 7.5e-5 / 1.5e-4). NOT RUN.
3. P4 screen: same weights at T1/T2/T4 (diagnostic forward) vs the deepest
   teacher rung. Pre-registration to be committed BEFORE running. NOT RUN.
4. Loss-scale normalization before any supervision A/B (addendum item 1):
   `thought_loss` weights final 1.0 + intermediates 0.25 each, i.e. 1.75 total at
   T=4 vs 1.0 for final_only. Must normalize by total active weight (with a unit
   test) or compensate LR. NOT DONE.
5. Experiment provenance hash (addendum item 6): git SHA, geometry, X15 config,
   thought count, dataset digest, teacher, ladder, supervision mode, weight, LR /
   optimizer / updates, seed. NOT DONE.
6. Throughput: batch the teacher labelling through the existing
   InferenceOwner/BatchedEvaluator instead of batch-1 SyncEvaluator; overlap host
   batch building with GPU steps in training; re-measure T=1 train cells warm
   (current T=1 train numbers are invalid). Cold start (kernel JIT/autotune)
   costs minutes per new shape; keep shapes fixed. NOT DONE.
7. P1.5 owner-pool (1 vs 2) and D55-vs-plain on X15: NOT RUN (X15 has no
   inference owner yet; plain-CUDA build not built).
8. P5 ablations, P6 tactical suite, mate-in-2 rule-exactness audit: NOT RUN.
9. P7 tiny self-play: not earned yet.

## E5 - trainer mechanics and micro-overfit (P3.1)
- QUESTION: can X15 learn from ReasoningTargetsV1, and do save / resume / eval
  work?
- RULE (mechanics, stated before): loss falls, parameters move, everything stays
  finite, `eval-reasoning` on a saved checkpoint reproduces the trainer's own
  evaluation, resume continues the update counter.
- CONFIG: 12 train positions, T_train=4, `final_only_v1`, lr 1e-4 (5-update
  warmup), 60 updates, full batch, seed 1, eval T=1..4 (one diagnostic pass; a
  test proves diagnostic thought k equals a T=k run's final readout).
- MEASURED: loss 3.40 -> 2.09; grads finite; peak VRAM 1.26 GB; whole run 92 s.
  `eval-reasoning` reproduced the val table exactly. Resume 60 -> 70 continued
  (loss 2.096 -> 2.089). Loss normalization by total active weight is now in
  (`thought_loss_with`, unit-tested; every supervision mode has unit total
  weight).
- POST-HOC (seen before pre-registration; NOT evidence): on the 32 held-out
  positions, same weights, KL to the deepest teacher: T1 0.430, T2 0.364,
  T3 0.354, T4 0.350 (T4 - T1 = -0.079, bootstrap 95% CI [-0.152, -0.018]).
  Monotone with T2 intermediate, from 12 training positions and one seed.
- DECISION: mechanics pass. The reasoning claim must be decided by the
  pre-registered screen below on fresh seeds, not by this run.
- KNOWN LIMIT: warmup restarted on resume (fixed in the next commit).

## E6 - PRE-REGISTERED: LR screen (P3.2)
Written and committed BEFORE running.
- QUESTION: which LR trains X15 stably on the fixed data?
- CELLS: lr in {3e-5, 7.5e-5, 1.5e-4, 3e-4 (high-LR control)}; all else fixed:
  96 train positions (full batch), T_train=4, `final_only_v1`, 40 updates,
  5-update warmup, seed 1, `targets-128` (digest `8c9237ff6a0c...`).
- SELECTION RULE: among cells with finite loss and finite gradient norms, pick
  the lowest held-out (val, 32 positions) mean KL to the deepest teacher at T=4;
  a cell within 0.01 KL of the best takes the smaller LR. Training loss alone
  never selects. A cell with any non-finite value is disqualified.
- BUDGET: about 2 min per cell.

## E7 - PRE-REGISTERED: core reasoning screen (P4)
Written and committed BEFORE running.
- QUESTION: does the SAME trained network at T4 move closer to the deeper
  Recur64 search target than at T1, and is T2 intermediate?
- CONFIG: targets-128; train on the 96 train positions, T_train=4,
  `final_only_v1`, LR = the E6 selection, 80 updates, two independent seeds
  (S1 = 1, S2 = 2). Evaluate the same weights at T=1..4 with the diagnostic
  forward on the 32 val positions. PRIMARY METRIC: mean KL(deepest-rung teacher
  policy || model policy) over legal moves. Secondary: CE, top-1 agreement with
  the teacher's best move, |p_win - p_loss - teacher root value|, entropy,
  latent delta, consecutive-thought KL.
- PROMISING iff ALL hold:
  1. mean KL at T4 < at T1 in both seeds;
  2. the paired-bootstrap 95% CI of (T4 - T1) on the pooled val positions
     excludes 0;
  3. the sign reproduces on both deterministic val halves (even / odd position
     index) in both seeds;
  4. T2 does not regress: mean KL at T2 <= mean KL at T1 in both seeds;
  5. each seed's gain (T1 - T4) exceeds the seed-to-seed spread of the T1 KL
     (|KL_T1(S1) - KL_T1(S2)|), the only variance estimate available.
- SECONDARY (reported, not decisive): a training-matched control trained at
  T_train=1 with identical settings; compare its T1 val KL against the T_train=4
  network's T4 val KL; and the untrained (update 0) baseline.
- IF NOT PROMISING (T4 ~= T1): run ONE bounded comparison,
  `final_only_v1` vs `progressive_search_v1` (ladder of 4 rungs = 4 thoughts),
  same LR / updates / seed S1, loss normalized to unit total weight. If
  recurrence still does not help, mark LATENT REASONING = NO SIGNAL for X1.
- NOT CLAIMED even if PROMISING: playing strength, or that latent reasoning
  (rather than repeated shared-core depth) is the cause; the symbolic-only
  control and ablations (P5) separate those.
