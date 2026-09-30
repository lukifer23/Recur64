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
