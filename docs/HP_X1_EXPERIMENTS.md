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
