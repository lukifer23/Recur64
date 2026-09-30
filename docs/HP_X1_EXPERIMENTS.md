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

### E6 results - LR screen (MEASURED, rule applied as pre-registered)
Same weights, val (32 positions), KL to the deepest teacher rung; 40 updates,
full batch of 96, T_train=4, `final_only_v1`, seed 1.

| lr | val KL T1 | T2 | T3 | T4 | T4-T1 (95% CI) | final train loss |
|---|---|---|---|---|---|---|
| 3e-5 | 0.4192 | 0.2656 | 0.2425 | 0.2258 | -0.193 [-0.406,-0.041] | 1.889 |
| 7.5e-5 | 0.3062 | 0.3092 | 0.2798 | 0.2189 | -0.087 [-0.215,+0.005] | 1.845 |
| 1.5e-4 | 0.3861 | 0.4248 | 0.3232 | 0.2225 | -0.164 [-0.305,-0.062] | 1.831 |
| 3e-4 | 1.0661 | 0.8663 | 0.4966 | 0.2814 | -0.785 [-1.247,-0.391] | 1.858 |

- All cells finite; gradient norms finite; no CUDA errors.
- SELECTION (rule from E6): lowest T=4 val KL is 7.5e-5 (0.2189); 3e-5 and 1.5e-4
  are within 0.01 KL of it, so the smaller LR is taken: **lr = 3e-5**.
- OBSERVATION (not a screen result): val KL at T4 is below T1 in all four cells,
  and 3 of 4 have a CI excluding zero. This is a single seed on 96 training
  positions and was not the pre-registered test; E7 decides.
- ENGINEERING FINDING (MEASURED): the full-batch (96) T=4 step peaked at
  **3.9 GB** VRAM, above the ~3.2 GB safety limit (the 2.38 GB figure belongs to
  micro-batches of 32). Nothing failed, but the trainer now accumulates gradients
  over micro-batches (`--micro-batch`, default 32; chunk-size-weighted, so the
  objective is unchanged). E7 uses this.
- WALL TIME: about 8 min for the first cell (cold kernel compilation for new
  shapes), about 3.6 min for each later cell.

## AMENDMENT A1 (2026-09-29) - committed BEFORE any confirmation-set label exists
Reason: the 32 `val` positions of `targets-128` (digest
`8c9237ff6a0ca34d5a9d7b131a7052f4c20e127ab4369f419db725e3d479afca`) were
observed in E5 (POST-HOC T1 -> T4 signal). They can no longer be a clean
confirmation set. E5/E6/E7 text above is NOT rewritten; this amendment
reclassifies and tightens. It does not change the hypothesis.

1. **Roles.** `targets-128` train 96 = training set (unchanged). `targets-128`
   val 32 = **TUNING set** (LR selection E6 and exploratory replicates).
   E7's primary confirmation moves to a NEW unseen set (A2).
2. **What already ran on the tuning set (honest ordering).** Before this
   amendment existed, E6 and an E7-configured replicate had already been run and
   evaluated on the tuning set (numbers below). They are reclassified as
   **tuning-set replicates / exploratory**, and are NOT confirmation. The E7
   *training* runs (train 96, lr 3e-5, T_train=4, `final_only_v1`, 80 updates,
   seeds 1 and 2, 32x3 accumulation) never touched the new set, so their
   checkpoints are legitimate inputs to confirmation on it.
3. **Execution amendment (not a science change).** "Full 96-position batch" is
   executed as the same full-data mean objective via deterministic 32 x 3
   gradient accumulation: chunk order = position order in the targets file, no
   shuffling, no reuse, each chunk's loss weighted by chunk_size / total before
   accumulation, one optimizer update per 96 positions. Regression test
   `accumulated_micro_batches_match_the_full_batch_objective` (CPU) compares
   full-batch vs accumulated loss and per-subsystem gradient norms. E6 cells were
   run as physical batch 96 (peak 3.9 GB, above the safety limit); E7 runs used
   32 x 3.
4. **Guards fixed.** `train_thoughts` and `eval_thoughts` must be in 1..=8 (1
   when reasoning is disabled); the config's `thought_steps` is the designed T,
   not a cap, so a `T_train=1` control on a T=4 config is legal and recorded as
   `train_thoughts`.
5. **Provenance.** `experiment.json` now records `initialization` (fresh+seed, or
   resume with parent model_id / parent experiment hash / resume update counter)
   and `start_update`, `segment_updates`, `final_update`. Fresh-start runs (all
   of E6/E7) are unaffected.

## A2 - fresh confirmation set: selection contract (PRE-STATED before generation)
- Teacher: Train1 final trainer, model_id
  `073357a9888256ec28be40649e6f1d67bb103341223f091ff5d67c50239676ae`
  (`runs/hp-r15-train1-r1/checkpoints/trainer`), `probe_v1`, recurrence 1.
- Search: c_puct 1.0, 1 leaf in flight, root noise OFF, ladder 16/32/64/128.
- Selection seed **20260930**; 64 positions; same replay
  (`runs/hp-r15-train1-r1/replay`).
- **Source-game exclusion:** every replay game used by `targets-128` is removed
  BEFORE selection (`--exclude-targets`), and disjointness is re-verified after
  generation as a hard error (`audit-targets --disjoint-from`). Different plies
  from the same games would not count.
- Same category selector (equal quotas across opening / middlegame / endgame /
  tactical / material_advantage; remainder to middlegame). All positions carry
  split label `confirm`.
- The set is NEVER used for LR selection, training, early stopping or
  architecture tuning. Its digest is recorded here after generation.
- Descriptive ladder-informativeness report (entropy by rung, adjacent-rung JS,
  shallow-vs-deep JS, best-move flip rate, mean |root value change|) is produced
  by `audit-targets --report` for both datasets. It is descriptive, not a gate.

## A3 - E7 (amended): confirmation on the fresh set
Same question. Training unchanged (train 96, lr 3e-5 per E6 rule, T_train=4,
`final_only_v1`, 80 updates, seeds 1 and 2). PRIMARY evaluation: the fresh 64
`confirm` positions only, same weights at T=1..4, metric KL(teacher_128 ||
X15_T). Unchanged criteria: (1) T4 < T1 in both seeds; (2) CI of T4 - T1
excludes 0; (3) direction holds in both deterministic halves (even / odd
position index); (4) T2 <= T1 in both seeds; (5) each seed's gain exceeds the
seed-to-seed spread of T1 KL on the confirmation set.
- **Bootstrap (exact, stated before results):** the experimental unit is the
  POSITION. Both seeds are evaluated on the same 64 positions, so seeds are NOT
  concatenated into 128 observations. For each position i,
  `delta_i = mean_over_seeds(KL_T4,i) - mean_over_seeds(KL_T1,i)`; the CI is a
  deterministic paired bootstrap of the mean of `delta_i` over the 64 positions
  (2000 resamples). Reported separately: seed-1 effect + CI, seed-2 effect + CI
  (each bootstrapped over positions), and the pooled position-clustered effect
  + CI. Same computation for T2 and T3 vs T1.
- **Secondary (not decisive):** the training-matched `T_train=1` Chimera control
  (2 seeds), evaluated at its own T=1 on the same confirmation set, compared with
  the T_train=4 network at T4. This control is a Chimera executing ONE thought,
  NOT the symbolic-only architecture; the symbolic-only ablation belongs to P5.
- **Not claimed even if PROMISING:** chess strength, conversion, visual or
  compute benefit, that the latent scratchpad (rather than repeated square-core
  depth) is the cause, or superiority to spending the same compute on PUCT. If
  PROMISING: STOP and document before any broad ablation.
- If NOT PROMISING: the single preregistered `progressive_search_v1` rescue.

## Tuning-set replicates observed before A1 (exploratory, NOT confirmation)
`targets-128` val 32, lr 3e-5, 80 updates, KL to the deepest rung
(KL at T1 / T2 / T3 / T4):
| run | T1 | T2 | T3 | T4 | notes |
|---|---|---|---|---|---|
| T_train=4, seed 1 | 0.4024 | 0.2484 | 0.2437 | 0.2040 | T4-T1 -0.198 [-0.416,-0.028]; halves -0.02/-0.38 |
| T_train=4, seed 2 | 0.3947 | 0.2258 | 0.1954 | 0.1911 | T4-T1 -0.204 [-0.429,-0.048]; halves -0.03/-0.37 |
| T_train=1 control, seed 1 | 0.1960 | 0.2372 | 0.3939 | 0.5109 | worsens with T |
| T_train=1 control, seed 2 | 0.1770 | 0.2473 | 0.4329 | 0.5546 | worsens with T |
| untrained (update 0) | 0.6323 | 0.6250 | 0.6205 | 0.6162 | baseline |
| progressive_search_v1, seed 1 | 0.5003 | 0.3062 | 0.2130 | 0.1890 | per-thought rung supervision |
| symbolic-only (T=1, no latents), seed 1 | 0.2029 | - | - | - | separate ablation config |
- OBSERVATION: the T_train=4 network is best at T=4 and the T_train=1 network is
  best at T=1; each is best at the depth it was trained at, and the T_train=4
  network's T4 (0.191-0.204) is NOT better than the T_train=1 network's T1
  (0.177-0.196) or the symbolic-only T1 (0.203). Extra thought in a network
  trained for it did not beat one pass on this tuning set. Whether that holds on
  fresh positions is what A3 asks. Peak VRAM: 2.4 GB (32x3, T4), 2.9 GB
  (progressive), 1.4 GB (T1).

## A2 result - fresh confirmation set (MEASURED)
`runs/x1/targets-confirm64.json` (evidence copy
`docs/evidence/x1/reasoning-targets-v1-confirm64.json`), digest
`2aad034191fe1daef4c73257d7cd43b4c7cfd22854f16ae403b72ac641775e6b`, seed
20260930, 64 positions (opening 12 / middlegame 13 / endgame 13 / tactical 13 /
material_advantage 13), split label `confirm`. 100 source games of `targets-128`
excluded before selection (188 remained); disjointness re-verified after
generation (`audit-targets --disjoint-from`: zero shared source games); every
position audited move for move. 2 min 4 s, RTX peak 51% util.

Ladder informativeness (descriptive, from `audit-targets --report`):
| dataset | positions | entropy 16/32/64/128 | JS 16-vs-128 | JS 32-vs-128 | JS 64-vs-128 | best move differs from 128 (16/32/64) | mean abs root-value change vs 128 |
|---|---|---|---|---|---|---|---|
| targets-128 all | 128 | 1.34/1.62/1.76/1.78 | 0.117 | 0.071 | 0.034 | 45/37/23 | 0.031/0.030/0.023 |
| targets-128 train | 96 | 1.25/1.52/1.68/1.72 | 0.117 | 0.074 | 0.035 | 32/25/16 | 0.032/0.032/0.023 |
| targets-128 val (tuning) | 32 | 1.61/1.93/1.99/1.96 | 0.117 | 0.059 | 0.032 | 13/12/7 | 0.029/0.026/0.020 |
| confirm64 | 64 | 1.58/1.93/2.10/2.11 | 0.129 | 0.072 | 0.035 | 32/24/17 | 0.029/0.024/0.021 |
The ladder carries progressively different policy information (deeper search is
flatter and disagrees with shallow search on the best move in roughly 25-50% of
positions at 16 simulations); root value barely moves across rungs.

## E7 RESULT (amended A3, fresh confirm set) - NOT PROMISING (MEASURED)
Same weights, T_train=4, `final_only_v1`, lr 3e-5, 80 updates, seeds 1 and 2;
KL(teacher_128 || X15_T) on the 64 confirmation positions.
| | T1 | T2 | T3 | T4 |
|---|---|---|---|---|
| seed 1 | 0.4336 | 0.3163 | 0.3384 | 0.2934 |
| seed 2 | 0.4073 | 0.3395 | 0.3427 | 0.3180 |
Pooled position-clustered (mean over seeds per position, then paired bootstrap
over the 64 positions, 2000 resamples): T2-T1 -0.093 [-0.180, -0.007]; T3-T1
-0.080 [-0.217, +0.070]; **T4-T1 -0.115 [-0.239, +0.023]**. Per seed: S1 T4-T1
-0.140 [-0.275, -0.003]; S2 -0.089 [-0.210, +0.057]. Even/odd halves (pooled)
-0.119 / -0.110.
Criteria: (1) T4 < T1 both seeds PASS; (2) pooled CI excludes 0 **FAIL** (upper
bound +0.023); (3) halves consistent PASS; (4) T2 <= T1 both seeds PASS; (5) each
gain (0.140, 0.089) exceeds the T1 seed spread (0.026) PASS.
By the pre-registered rule (ALL must hold): **NOT PROMISING**. The direction is
consistent and the point estimate is about half the tuning-set estimate; the
evidence is insufficient at n=64.
SECONDARY (T_train=1 Chimera controls, one thought, same set): T1 KL 0.2981 and
0.3323 (mean 0.315) versus the T_train=4 networks at T4 0.2934 and 0.3180
(mean 0.306). The thought-trained network at its trained depth is not better
than a one-pass network trained identically, and the T_train=1 networks get
steadily worse with more thoughts (T4 0.524 / 0.589). Each network is best at
the depth it was trained at.
INTERPRETATION: the within-network T1 -> T4 improvement is largely "the network
is best at its trained depth"; it is not evidence that extra thought beats one
pass. Not established: any strength, conversion, visual or compute benefit.

## E8 - PRE-REGISTERED: the single progressive_search_v1 rescue (before evaluation)
Stated and committed BEFORE the `progressive_search_v1` checkpoint is evaluated
on the confirmation set. The checkpoint (`runs/x1/e7-prog-s1`: train 96, lr 3e-5,
T_train=4, 4 rungs = 4 thoughts, seed 1, 80 updates, loss normalized to unit
total weight, 32x3 accumulation) was trained before A1 and evaluated only on the
tuning set (T4 KL 0.189 there); it has never seen the confirmation set.
- QUESTION: does per-thought search supervision make recurrence beat one pass?
- RULE: RESCUED iff the paired position-bootstrap 95% CI (64 confirmation
  positions) of `KL(prog T4) - KL(T_train=1 control seed 1, at T=1)` has an upper
  bound below 0. Otherwise LATENT REASONING = NO SIGNAL for X1 at this scale.
- Also reported (not decisive): the same comparison for the `final_only_v1`
  networks against the pooled T_train=1 controls, and the prog network's T1..T4.
- No other rescue attempts (no LR / update / architecture tuning for this).

## E8 RESULT - progressive_search_v1 rescue: NOT RESCUED (MEASURED, rule applied as pre-registered)
`runs/x1/e7-prog-s1` on the 64 confirmation positions (KL to the 128-simulation
teacher): T1 0.4877, T2 0.3857, T3 0.3353, T4 0.2997 (monotone, all differences
vs T1 have CIs below 0; per-thought supervision makes T2 and T3 useful). But the
rescue rule compares against a one-pass network: `KL(prog T4) - KL(T_train=1
control s1 at T1)` = 0.2997 - 0.2981 = **+0.0016, 95% CI [-0.0392, +0.0462]**,
position-clustered. Upper bound is not below 0, so it is **NOT RESCUED**.
Formal secondary (final_only networks, seeds 1+2, T4, vs T_train=1 controls at
T1): 0.3057 vs 0.3152, diff -0.0095 [-0.0338, +0.0187]: a tie.
- **DECISION (per the pre-registered E7/E8 rules): LATENT REASONING = NO SIGNAL
  for X1 at this scale** (96 training positions, 80 updates, 2 seeds, one
  teacher). Not "proved useless": the data is small and the training short. What
  is established is that, on fresh held-out positions, spending 4 thoughts did not
  beat spending 1, under either final-only or per-thought supervision.
- EXPLORATORY (one seed, not pre-registered): symbolic-only network (no latents,
  compute or visual; T=1) on the confirmation set: KL 0.3002 (top-1 0.516),
  against 0.298 / 0.332 for the Chimera T_train=1 controls and 0.293 / 0.318 for
  the T4 networks. All variants are within noise of each other. Neither compute,
  visual nor recurrence shows a measurable benefit at this data scale.
- NEXT (owner decision, per stop condition "fixed-data result clear enough to
  need an owner decision"): options C/D. Cheap, decisive follow-ups if wanted:
  (a) scale the training set (needs batched teacher labelling, see outstanding
  item 6) so that a capacity-bound variant could separate from the symbolic
  control; (b) P6 tactical / conversion suite, where exact compute tokens could
  matter even if teacher-KL does not; (c) pivot. No self-play was earned.
- GPU experiment time used so far: about 1.5 hours of the 2-hour budget.

## E9 - tactical / conversion suite (P6), first reading (MEASURED)
- SUITE (`x15 gen-tactics`, seed 20261001, 0.5 s, evidence
  `docs/evidence/x1/tactics-v1.json`): 80 deterministic fixtures. Mates in five
  material sets (KQvK, KRvK, KQQvK, KQRvK, KRRvK; 8 each; every mating move is
  correct), `material_gain` (8; best 2-ply material outcome, margin >= 3),
  `promotion` (8; margin >= 2), and 24 `hp_repetition_fen` positions (final FENs
  of Train1 arena games that ended by threefold repetition, material lead >= 3,
  repetition history LOST: they carry no move label). Synthetic kinds are FEN-only
  with fresh clocks and no history, so mate labels are decided by the board alone.
  The 2-ply labels are bounded material claims, not "best move".
- Mate-in-2 rule-exactness: NOT USED as a target. The coprocessor's bounded
  mate-in-2 uses a plain cozy-chess board without Recur64's repetition / fifty-move
  history, so it is at most "board-forced mate"; it stays diagnostic only
  (`mate_depth = 1` everywhere in X1).
- TEACHER CEILING (Train1 trainer + deterministic PUCT, `x15 teacher-tactics`,
  55 s): top-1 correct / mass on correct moves. Mates at 16 simulations 0.38-0.62,
  at **64 and 128 simulations 1.00** (mass 0.66-0.91). `material_gain` 0.62-0.88,
  `promotion` 0.25-0.50 (this network is not strong there). So search converts
  depth into mate-finding; the fixtures are solvable.
- X15 VARIANTS (trained on the 96-position set, 80 updates; T1 -> T4 shown for
  networks trained at T=4): top-1 correct on mates 0.00-0.25 (chance level; mass on
  correct moves 3-7%), promotion 0.12 (chance), `material_gain` 0.75-1.00 for
  ALL variants including symbolic-only (0.88). No variant separates from another.
  P(win) rises with T only in T-trained nets (0.39 -> 0.46 for `final_only` T4;
  0.42 -> 0.44 progressive) while correct-move rate does not. The value sign
  agrees with the material lead on 96% of the repetition-draw FEN positions.
- INTERPRETATION (MEASURED + INFERRED): at this training scale the networks have
  not learned tactics at all, so the suite cannot yet separate recurrence, compute
  or vision. It is a useful, solvable target: the teacher reaches 100% on mates
  with 64 simulations. INFERRED design lead for the reasoning work: mate-in-1
  needs one ply of look-ahead plus terminal detection. `ComputeBankV1` reports
  whether a mate exists (a global token) but not WHICH candidate mates, so the
  network cannot read the answer off the bank. A candidate-level exact-fact channel
  (per legal move: gives mate / gives check / captures value / hangs the mover) is
  the natural next contract, and it belongs to a versioned bank change.
- NOT RUN: playing fixtures out (conversion rollouts), which needs fixed candidate
  widths to avoid per-shape kernel compilation; the ~2-hour GPU budget is essentially
  spent (about 1.7 h).

## E10 - scaled training set + candidate-facts channel (engineering, MEASURED)
- **Batched labelling (D63):** 960 positions in 6 min 8 s (2.6 positions/s vs 0.5),
  up to 91% GPU utilization, 0 inference errors. Owner batches were capped at 16 in
  practice (`batch_size_max` 16), a tuning knob for later.
- **Training set `targets-train960`** (digest
  `39a136c266cebf51f8b1aa45a73aa669f8ee928bee862826b2d1b739ba644911`, evidence copy in
  `docs/evidence/x1/`): 400 replay positions (exact history; 80 each of opening /
  middlegame / endgame / tactical / material_advantage) drawn from the 207 replay
  games NOT used by the tuning-val or the confirmation set (hard-verified disjoint,
  81 games excluded), plus 560 synthetic tactic positions (7 kinds x 80, seed
  20261002, FENs of the evaluation suite excluded). Teacher and ladder as in A2.
  All positions carry split `train`.
- **Candidate facts (D62):** `candidate_facts_v1` implemented, 8 exact fields per
  legal move; 5 unit tests (mate flagged on exactly the mating move, capture value,
  promotion, attacked destination, stalemate, padding, determinism) and 3 model
  tests: neutral at init, inert when disabled, gradient when enabled, and a tiny
  symbolic network with facts learns to play the mating move (mass on the mating
  move > 0.5 and > 3x its initial value). Workspace clippy 0 warnings.

## E11 - PRE-REGISTERED: scaled ablation - facts and thought (committed BEFORE running)
- QUESTIONS: (Q1) do candidate facts let a network find mates? (Q2) does extra
  thought (T_train=4) beat one pass (T_train=1), with and without facts?
- DATA: `targets-train960` for training. Evaluation sets never used for training or
  tuning: the tactical suite `tactics-v1` (80 fixtures; its FENs were excluded from
  the synthetic training positions) and a NEW replay confirmation set `confirm2`
  (64 positions from games disjoint from train960, targets-128 and confirm64;
  contract as A2, seed 20261004, generated before any E11 result and its digest
  recorded here after generation).
- VARIANTS (all fresh start, lr 3e-5 is NOT reused blindly: the LR is fixed by
  rule to 1e-4 because the set is 10x larger, chosen before results and not tuned):
  S = symbolic-only (T=1); SF = symbolic + facts (T=1); C1 / C4 = Chimera without
  facts, T_train = 1 / 4; C1F / C4F = Chimera with facts, T_train = 1 / 4.
  `final_only_v1`, seeds 1 and 2, 100 updates, each update = 96 positions (3 x 32
  micro-batches, accumulated) taken in a seeded fixed order over train960 (about 10
  epochs), 5-update warmup, value weight 1. Training positions per update and order
  are deterministic and recorded.
- METRICS: (a) mate top-1 accuracy on the 40 `tactics-v1` mate fixtures (all five
  material sets pooled); (b) `material_gain` and `promotion` top-1 (reported); (c)
  mean KL(teacher_128 || model) on `confirm2` at the trained T. Cost: peak VRAM and
  seconds per update for every variant.
- Q1 decision: FACTS WORK iff SF (mean of 2 seeds) mate top-1 exceeds S by >= 0.30
  absolute AND each SF seed exceeds each S seed.
- Q2 decision: THOUGHT SIGNAL iff, in at least one facts setting (no facts: C4 vs C1;
  facts: C4F vs C1F), C4x beats C1x with a paired position-clustered bootstrap 95% CI
  (mean over seeds per position, 2000 resamples; fixtures for (a), confirm2 positions
  for (c)) that lies wholly on the favourable side on BOTH (a) and (c), and the sign
  agrees in each seed separately. Anything else: NO THOUGHT SIGNAL at this scale.
- Interpretation limits: a facts win is a tool, not reasoning. A thought signal here
  would still not establish that the latent scratchpad (rather than repeated
  square-core depth) is responsible; that needs the latent-only / compute / visual
  ablations. No strength claim. Conversion rollouts remain NOT RUN.

### E11 addendum - confirm2 generated (before any E11 result)
`targets-confirm2` (evidence copy `docs/evidence/x1/reasoning-targets-v1-confirm2.json`),
digest `89a39e189f0790a8adceb45d7c182e958bc2cffd58d5014366d270ad1ce57c9d`: 64
positions (opening 12 / middlegame 13 / endgame 13 / tactical 13 /
material_advantage 13), seed 20261004, from the 27 replay games used by NO other set
(targets-128, confirm64 and the replay part of train960 all excluded; hard
disjointness verified), split label `confirm`, teacher and ladder as A2, batched
labelling (identical to batch-1, D63). 56 s.
Execution details fixed before results: one fixed candidate width (next standard
bucket at or above the widest legal list) for the whole run so the GPU sees one
shape; per-update data = 96 positions = 3 micro-batches of 32 taken in seeded
hash order over train960, cycling; evaluation inputs are built from each
checkpoint's own experimental contract. Tooling: `x15 compare` computes the
pre-registered statistics (fixture / position-clustered paired bootstrap of the
seed-averaged difference, 2000 resamples, with per-seed signs).

## E11 RESULT - scaled ablation (MEASURED, pre-registered rules applied as written)
12 fresh-start runs on `targets-train960` (lr 1e-4, 100 updates x 96 positions,
seeds 1 and 2), evaluated on the frozen sets: `tactics-v1` (40 mate fixtures = 5
material sets x 8) and `confirm2` (64 positions; teacher KL at the trained T).
| variant | wall | peak VRAM | final train loss (s1 / s2) |
|---|---|---|---|
| S symbolic (T=1) | 48-51 s | 1.35 GB | 2.181 / 2.098 |
| SF symbolic + facts (T=1) | 47-50 s | 1.35 GB | 2.194 / 2.041 |
| C1 Chimera (T=1) | 86-103 s | 1.51 GB | 2.145 / 2.087 |
| C4 Chimera (T=4) | 144 s | 2.44 GB | 2.184 / - |
| C1F Chimera + facts (T=1) | 86-110 s | 1.51 GB | 2.140 / - |
| C4F Chimera + facts (T=4) | 140-142 s | 2.44 GB | 2.172 / 2.240 |
Training is cheap now: 100 updates in under 2.5 minutes at T=4.

Q1 (facts let a network find mates?): SF mate top-1 0.325 vs S 0.300, diff +0.025,
95% CI [-0.038, +0.088], per-seed diffs [0.00, +0.05]. Rule required >= +0.30 with
both seeds above: **FAILS. Facts did not help at this optimization budget.**
Q2a (no facts) C4 (T=4) vs C1 (T=1): mate top-1 0.200 vs 0.288, diff -0.088 [-0.175,
0.000]; teacher KL 0.3250 vs 0.2882, diff +0.037 [+0.004, +0.073] (C4 worse).
Q2b (facts) C4F vs C1F: mate 0.263 vs 0.300, diff -0.038 [-0.125, +0.038]; KL
0.3142 vs 0.2874, diff +0.027 [-0.004, +0.062]. Promotion top-1 is also worse at T4
(0.19 vs 0.69; 0.44 vs 0.69).
**Rule outcome: NO THOUGHT SIGNAL** (no facts setting has C4 wholly favourable).
Further descriptive rows: C1F vs C1 mate +0.013 [-0.025, +0.050], KL -0.0009: the
facts channel is inert here in the latent architecture too.
Per mate set (mean of 2 seeds, top-1): KQvK 0.00-0.06 for every variant (never
solved), KRvK 0.44-0.56, KQQvK/KQRvK 0.12-0.38, KRRvK 0.19-0.56. Teacher with 64
simulations: 1.00 everywhere.
INTERPRETATION: more thought did not help and cost 1.6x wall time and 1.6x VRAM; at
equal updates the T=4 network is, if anything, slower to fit. The facts channel
should have made mates easy (a unit test shows a small network learning it in 60
steps), so its failure here needs a diagnosis, not a conclusion about facts.

## AMENDMENT A4 (after the E11 result, BEFORE any gain result) - facts scale
- **Diagnosis (MEASURED, `x15 facts-probe` on the E11 checkpoints):** the learned
  fact bias separates mating moves from the other legal moves by only **+0.014
  logits** (SF s1 +0.0144, SF s2 +0.0177, C1F s1 +0.0143, C4F s1 +0.0144). AdamW moves
  each weight by about the learning rate per step (1e-4), and the fact MLP starts
  at zero, so in 100 updates it cannot reach the several logits needed to single
  out one move among ~20-30. The unit test that learned the mate used lr 2e-2. The
  E11 Q1 failure is therefore an optimization-scale failure of the channel, not
  evidence about facts.
- **Change:** `[experimental.candidate_facts] gain` (default 1.0, in the identity):
  bias = gain * MLP(facts). Default behaviour is unchanged.
- **Tuning set** `tactics-tune`: `gen-tactics` seed 20261005, 8 per kind, no
  repetition-draw positions; hard-verified to share no FEN with `tactics-v1` and no
  start / current FEN with `targets-train960`. It is used ONLY for choosing the gain.
- **Screen:** SF (symbolic + facts), seed 1, gains {8, 32, 128}, all other E11
  settings unchanged. Metric: mate top-1 on the 40 `tactics-tune` mate fixtures.
  Select the highest; ties go to the smaller gain; a run with a non-finite value is
  disqualified. Reference: E11 S seed 1 on the same set.
- **E11b (amended rerun):** SF, C1F, C4F with the selected gain, seeds 1 and 2, all
  else as E11; S, C1 and C4 are unchanged (they do not use facts) and are reused.
  Judged by the SAME E11 rules on `tactics-v1` and `confirm2`: Q1 (SF vs S mate top-1
  >= +0.30 absolute, both seeds above) and Q2b (C4F vs C1F wholly favourable on
  mates and KL, seeds agreeing). **The original E11 outcome (Q1 fail, no thought
  signal) stands and is reported alongside; E11b does not replace it.**
- Still NOT claimed: facts are a tool, not reasoning; strength and conversion.

### A4 result - gain screen on `tactics-tune` (MEASURED, rule applied as written)
`tactics-tune` (seed 20261005, 56 fixtures, verified to share no FEN with `tactics-v1`
or `targets-train960`). SF seed 1, 100 updates, mate top-1 over the 40 mate fixtures;
reference S seed 1 = 0.250.
| gain | mate top-1 | diff vs S (95% CI) | learned bias gap (correct - other) | final loss |
|---|---|---|---|---|
| 1 (E11) | - | - | +0.014 logits | 2.194 |
| 8 | 0.275 | +0.025 [-0.050, +0.100] | +0.107 | 2.160 |
| 32 | 0.450 | +0.200 [+0.075, +0.325] | +0.428 | 2.013 |
| 128 | **0.875** | **+0.625 [+0.475, +0.775]** | +1.661 | 1.672 |
Monotone in the gain, exactly as the diagnosis predicts. **Selected gain = 128**
(highest; the rule's grid was {8, 32, 128}). Caveats: it is the edge of the grid, so a
larger gain may be better still; one seed; the tuning set only. KQQvK stays hard even
at gain 128 (0.50).
Configs: `configs/x15_facts_g128_cuda.toml`, `configs/x15_symbolic_facts_g128_cuda.toml`.

## E11b RESULT - amended rerun with facts gain 128 (MEASURED, E11 rules applied as written)
SF, C1F, C4F at gain 128, seeds 1 and 2, all else as E11; S, C1, C4 reused from E11.
Evaluated on the frozen `tactics-v1` (40 mate fixtures) and `confirm2` (64 positions).
| variant | wall | peak VRAM | final train loss (s1 / s2) |
|---|---|---|---|
| SF g128 (T=1) | 48-49 s | 1.35 GB | 1.673 / 1.727 |
| C1F g128 (T=1) | 86 s | 1.51 GB | 1.681 / 1.732 |
| C4F g128 (T=4) | 136-139 s | 2.44 GB | 1.707 / 1.744 |
**Q1 - facts work: PASS.** SF g128 mate top-1 **0.950** vs S 0.300, diff **+0.650**, 95% CI
[+0.513, +0.775], per-seed diffs [+0.65, +0.65] (rule: >= +0.30, both seeds). Per set:
KQvK 1.00, KRvK 1.00, KRRvK 1.00, KQQvK 0.88, KQRvK 0.88 (S: 0.06 / 0.50 / 0.50 / 0.25 /
0.19). Teacher with 64 simulations: 1.00. Teacher KL on `confirm2` is unchanged
(0.2871 vs 0.2821, CI includes 0); `material_gain` and `promotion` are unchanged or
slightly worse: the facts help exactly where they carry the answer.
**Q2b - thought beats one pass with facts: FAIL (no thought signal).** C4F 0.9125 vs
C1F 0.9000 on mates (+0.0125 [-0.038, +0.063]); teacher KL 0.2987 vs 0.2846 (+0.014
[-0.012, +0.044]); `material_gain` worse at T=4 (0.31 vs 0.56, CI wholly below 0).
E11 (gain 1) outcome stands as recorded above: Q1 fail (facts inert because the module
could not learn a useful scale), Q2a no thought signal.
INTERPRETATION (MEASURED + INFERRED): (1) the candidate-fact channel is a real, cheap
capability gain: mate-in-1 goes from chance to ~95% in ~50 s of training. (2) With that
answer available from a one-ply fact, mate-in-1 can no longer discriminate anything
about thinking: the benchmark is at its ceiling for the one-pass network, so a null
here says little. (3) Across E7, E8, E11 and E11b, extra recurrent thoughts have
never beaten a one-pass network trained the same way on any fixed-data metric, and
cost 1.6x wall time and VRAM.
NEXT (needs an owner decision): a problem family where the one-ply facts do NOT reveal
the answer and lookahead is required, i.e. mate-in-2 and multi-step captures. Under
the fresh-clock, no-history fixture convention a 4-ply forced mate is exact under
Recur64 rules (no repetition or fifty-move state can arise), so the labels can be made
rule-exact by exhaustive bounded search. The teacher's ability at those depths is the
ceiling to establish first.

## E12a - the teacher cannot do mate-in-2 (MEASURED)
`mate2-eval` (`x15 gen-mate2`, seed 20261010, 60 fixtures = 5 material sets x 12,
each with NO mate in one and at least one first move that forces mate in two;
labelled by exhaustive board-rule search; 9 sampled positions cross-checked against
the full GameState rules path; hard-disjoint from `tactics-v1`, `tactics-tune` and
`targets-train960`). Exactness assumption, stated: fixtures are fresh-clock and
no-history, so a 4-ply sequence cannot reach threefold repetition or the fifty-move
claim; this is a property of the fixture convention, not of arbitrary positions.
Teacher (Train1 trainer + deterministic PUCT, `teacher-tactics`), top-1 / mass on
forcing moves, mean over the five sets:
| simulations | 16 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| top-1 | 0.19 | 0.03 | 0.07 | 0.18 | 0.33 |
Per set at 512: KQvK 0.42, KRvK 0.58, KRRvK 0.33, KQRvK 0.25, KQQvK 0.08. The
teacher is near chance up to 128 simulations and only starts to find these at 512
(it took 7.5 min for this table). PUCT-at-128 targets therefore carry almost no
information about this skill.
**Consequence:** for this family the teacher is EXHAUSTIVE BOUNDED RULES SEARCH (allowed:
"internal bounded rules search"), not PUCT: the target is uniform over every first move
that forces mate in two, root value +1. It is recorded as such in the teacher contract
(`exact_rules_search_v1`) and every variant receives identical supervision.

## E12 - PRE-REGISTERED: does thought help when lookahead is required? (before training)
- QUESTION (Q3): with one-ply facts available (they cannot reveal a mate-in-2), does a
  network that can think 4 (or 8) times find the forcing first move better than a
  one-pass network trained identically?
- DATA: `targets-train960` (as E11) + `targets-mate2-exact-train` (5 material sets x 300 =
  1500 positions, seed 20261011, exact labels, hard-disjoint from `mate2-eval`,
  `tactics-v1`, `tactics-tune` and `targets-train960`). Merged in file order; single-rung
  files are extended by repeating their last rung (only matters for progressive
  supervision, not used).
- VARIANTS (fresh start, facts gain 128 as selected in A4, `final_only_v1`, lr 1e-4, 5
  update warmup, 600 updates x 96 positions in a seeded hash order, seeds 1 and 2):
  SF (symbolic + facts, T=1), C1F (T=1), C4F (T=4), C8F (T=8). C8F uses micro-batch 16
  (memory); the accumulated objective is identical.
- METRICS: PRIMARY mate-in-2 top-1 on `mate2-eval` (60 fixtures), evaluated at the
  trained T; retention: mate-in-1 top-1 on `tactics-v1`, teacher KL on `confirm2`.
- DECISION: THOUGHT HELPS LOOKAHEAD iff C4F beats C1F on mate-in-2 top-1 with a
  fixture-clustered paired bootstrap 95% CI (mean over seeds per fixture, 2000
  resamples) wholly above 0 and BOTH seeds individually positive. Descriptive: C8F vs
  C4F vs C1F trend, SF vs C1F.
- FLOOR RULE: if the best variant's mate-in-2 top-1 is below 0.30 (chance is about
  0.1-0.2), the run is labelled FLOOR (uninformative) and I make ONE pre-stated
  extension, 1200 updates, before concluding anything; if still at the floor, stop and
  report.
- NOT CLAIMED: strength or conversion; that any gain is due to the latent scratchpad
  rather than repeated shared-core depth; that thought is useful outside this family.

## E13 - PRE-REGISTERED (written during E12, before any E12 or E13 result is analysed)
- QUESTION: E12 uses 1,500 exact mate-in-2 positions, a fixed dataset that a network can
  partly memorise. Exact labels are essentially free (1,500 positions in 2 s), so data
  need not be a limit. Does thought help on mate-in-2 when the network sees tens of
  thousands of distinct positions and cannot be data-limited?
- DATA: `targets-train960` + `targets-mate2-exact-15k` (3,000 exact positions per
  material set, seed 20261012, digest
  `948b1d1a7d2b8a8dae5eed618a1920b18789c81a94dfe2f95098a7896b7dc042`), 15,960
  positions, hard-disjoint from `mate2-eval`, `tactics-v1`, `tactics-tune`,
  `targets-train960` and the E12 exact set. (6,000 per set was infeasible: KQvK has too
  few distinct mate-in-2 positions to sample that many.)
- TRAINING: `--stream` (micro-batches built on demand; identical objective), fixed
  candidate width, 96 positions per update in a seeded hash order (1,200 updates is
  about 7 passes over the data), lr 1e-4, gain 128, 5-update warmup, seeds 1 and 2.
- VARIANTS: SF (T=1), C1F (T=1), C4F (T=4). C8F is added only if C4F shows a signal.
- METRIC AND DECISION: identical to E12 (mate-in-2 top-1 on `mate2-eval`; C4F beats C1F
  with a fixture-clustered paired 95% CI wholly above 0 and both seeds positive).
  Retention metrics reported: mate-in-1 top-1 on `tactics-v1`, teacher KL on `confirm2`.
- DESCRIPTIVE: E12 vs E13 accuracy for SF and C1F shows whether data was limiting.

## E12 RESULT - mate-in-2, thought vs one pass (MEASURED; pre-registered rule applied as written)
Data `targets-train960` + 1,500 exact mate-in-2 positions; facts gain 128; 600 updates x
96 positions; seeds 1, 2. Evaluated on `mate2-eval` (60 fixtures).
| variant | wall | peak VRAM | final loss (s1 / s2) | mate-in-2 top-1 (mean of 2 seeds) |
|---|---|---|---|---|
| SF (T=1) | 294 s | 1.45 GB | 1.39 / - | 0.600 |
| C1F (T=1) | 399-427 s | 1.99 GB | 1.41 / - | 0.575 |
| C4F (T=4) | 706-727 s | 2.67 GB | 1.40 / - | 0.542 |
| C8F (T=8, micro-batch 16) | 1255-1285 s | 3.82 GB | 1.59 / - | 0.300 |
Not at the floor (best 0.60 > 0.30), so the test is informative. **Q3: C4F vs C1F = -0.033,
95% CI [-0.108, +0.042], per-seed [0.00, -0.067]: THOUGHT DOES NOT HELP LOOKAHEAD**
(rule needed a CI wholly above 0 and both seeds positive). Descriptive: C8F vs C1F -0.275
[-0.358, -0.192] (both seeds negative), C8F vs C4F -0.242 [-0.317, -0.167]; SF vs C1F
+0.025 [-0.058, +0.108]. Per set (C4F / C1F): KQvK 0.71/0.88, KQQvK 0.29/0.46, KQRvK
0.62/0.50, KRvK 0.42/0.38, KRRvK 0.67/0.67. The one-pass symbolic-plus-facts network
finds a forced mate in two 60% of the time, against 3-7% for the teacher at 128
simulations and 33% at 512.
INTERPRETATION: unrolling the shared core more times made fitting WORSE (T=8 has higher
training loss and much lower accuracy), so the failure looks like an optimisation
problem of the deep unroll under final-only supervision, not evidence that thought is
useless in principle. Deep supervision is the standard remedy and is already
implemented, hence E14. E13 (15k positions, streaming) is running as pre-registered.

## E14 - PRE-REGISTERED (written before any deep-supervision result exists)
- QUESTION: does deep supervision (`same_target_v1`, every thought trained against the
  exact target, intermediates weighted 0.25, loss normalised to unit total weight) rescue
  the recurrent network, so that more thoughts help find the forcing first move?
- DESIGN: identical to E12 in every respect (data, gain 128, lr 1e-4, 600 updates x 96,
  seeds 1 and 2, non-streaming) except `--supervision same_target_v1`: C4F-DS (T=4) and
  C8F-DS (T=8, micro-batch 16). Reuses E12's C1F (T=1) as the one-pass baseline.
- DECISION: DEEP SUPERVISION RESCUES RECURRENCE iff C4F-DS beats E12 C1F on mate-in-2 top-1
  with a fixture-clustered paired 95% CI wholly above 0 and both seeds positive.
  Reported: the per-thought curve of each DS network (T=1..T_train from one diagnostic
  pass; monotone improvement with T is the signature of useful thought), C8F-DS vs
  E12 C8F (does supervision fix the T=8 collapse), retention (mate-in-1, teacher KL).
- NOT CLAIMED even if it succeeds: strength, conversion, or that the latent scratchpad
  (rather than repeated core depth) is responsible.

## E13 RESULT - mate-in-2 with 15,960 training positions, streamed (MEASURED; rule as written)
`targets-train960` + `targets-mate2-exact-15k`, 1,200 updates x 96 (about 7 passes), seeds 1, 2,
gain 128, evaluated on `mate2-eval` (60 fixtures).
| variant | wall | peak VRAM | mate-in-2 top-1 (mean of 2 seeds) |
|---|---|---|---|
| SF (T=1) | 444-449 s | 2.0 GB | 0.792 |
| C1F (T=1) | 765-766 s | 1.5 GB | 0.775 |
| C4F (T=4) | 1376-1384 s | 2.4 GB | 0.742 |
**Q3: C4F vs C1F = -0.033, 95% CI [-0.108, +0.042], per-seed [0.00, -0.067]: THOUGHT DOES
NOT HELP LOOKAHEAD** (identical difference to E12, at 10x the data). SF vs C1F +0.017
[-0.058, +0.092]. Data scale (descriptive): SF 0.792 (15k) vs 0.600 (1.5k), +0.192
[+0.092, +0.300], both seeds positive: data WAS limiting for the one-pass network, and
removing that limit did not make thought useful. Per set (C4F / C1F): KQvK 0.92/0.88,
KQQvK 0.50/0.54, KQRvK 0.58/0.71, KRvK 1.00/1.00, KRRvK 0.71/0.75.
Retention (C4F vs C1F): mate-in-1 1.00 vs 0.95 (+0.05, CI [0.00, +0.11], both seeds
+0.05); promotion 0.63 vs 0.44 (seeds disagree); teacher KL on `confirm2` 0.540 vs 0.529
(tie) - both far worse than E11b's 0.29 because the mate-in-2 positions now dominate
the training mixture (replay positions are 6% of it): forgetting, not a thought effect.
INTERPRETATION: with exact labels, plenty of data and one-ply facts, the one-pass
network reaches 0.79 on forced mates in two and the recurrent network does not beat it,
at 1.8x the training time. E14 (deep supervision) is the remaining pre-registered
attempt to make recurrence useful on this task.

## E14 status note and a MEASURED stability finding (recorded while E14 trains)
`x15 thoughts --checkpoint` on trained E12 networks (batch 16, evaluated to T=8): the
latent state `Z` (mean |Z|) grows about 1.3x per thought in EVERY network, with nothing
bounding it: T_train=4 net 1.01 -> 6.20 by thought 8; T_train=8 net 0.95 -> 3.63 (it
learned to damp it); T_train=1 net 0.90 -> 8.78 (its p(win) saturates to 0.965). `Z`
only ever receives residual additions (cross-attention, latent FFN) and is normalised
once, at initialisation. Unbounded growth across a shared-parameter loop is a standard
optimisation hazard, and it fits the T=8 collapse in E12 (higher training loss, lower
accuracy). This is a measurement plus a hypothesis, not a proof.

## E15 - PRE-REGISTERED: does bounding the latent rescue recurrence? (before any result)
- CHANGE (versioned, default off): `[experimental.reasoning] normalize_latent = true`
  re-applies the existing `latent_norm` RMSNorm to `Z` after every thought. Off is the
  historical loop, bit-identical. Test: with it on, mean |Z| stays in (0.4, 1.3) for all 8
  thoughts, with it off it grows > 1.5x. Config: `configs/x15_facts_g128_norm_cuda.toml`.
- DESIGN: identical to E12 (train960 + 1,500 exact mate-in-2 positions, gain 128,
  `final_only_v1`, lr 1e-4, 600 updates x 96, seeds 1 and 2, non-streaming) except
  `normalize_latent = true`: C1F-N (T=1), C4F-N (T=4), C8F-N (T=8, micro-batch 16).
- DECISION: NORMALISATION RESCUES RECURRENCE iff C4F-N beats E12's C1F (the standing
  one-pass baseline) on mate-in-2 top-1 with a fixture-clustered paired 95% CI wholly
  above 0 and both seeds positive AND the same holds against C1F-N (matched one-pass).
  Reported regardless: C8F-N vs E12 C8F (is the T=8 collapse fixed), the per-T curves, and
  retention.
- If it does not rescue: recurrence stays NO SIGNAL at this scale under final-only
  supervision (E12), deep supervision (E14) and a bounded latent (E15).
