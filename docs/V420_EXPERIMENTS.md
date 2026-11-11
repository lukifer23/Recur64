# V420 — experiment ledger

Plan and frozen gates: `docs/V420_RESEARCH_PLAN.md` (approved 2026-11-11; amendment A1). Branch `experiment/workstation-v420-recurrent-depth`.
Every entry below is written BEFORE the result it precedes. Status words: MEASURED, DETECTED, TESTED, INFERRED, NOT RUN.

## V420-E0 — P0: plan approved and frozen
- **Date:** 2026-11-11. Owner approved arms, gates, thresholds, the 3 GPU-hour ceiling, "a failed S1 ends V420", and "option 2 stays unbuilt
  unless S1 passes". Commit `1137061`. No code, data or run existed.

## V420-E1 — P1: resolved layouts, implementation, CPU correctness, CUDA engineering smoke (NO science result)
- **Date:** 2026-11-11. **Status:** TESTED (CPU correctness), MEASURED (engineering timings only). Nothing here feeds any gate.
- **Deviation found and fixed in the plan (owner decision, before any run):** the original S4 compared a 1-block R1 against V4's 8-block
  Stage A band and could never be met. S4 is now a BASE8 pipeline check; the recipe is V4 Stage A's for every arm (plan section 10, A1).
- **Implementation (new, `crates/recur64-v420`; no V3/V3.5/V4 crate changed):** `model.rs` re-states V4's base path with explicit
  board blocks (public `recur64_model::model::Block` and `CandidateInputs`; the crate-private candidate block is re-stated) minus the
  unused WDL head; `train.rs` arms/recipe/trainer/eval; `diag.rs` the plan-section-6 gradient-health check and its abort rule; `gates.rs`
  the frozen gates as pure functions. CLI: `recur64 v420 {model-info, smoke, train, measure}`. Data: `V4Data::load` (the frozen P25 TRAIN
  artifact only; every sealed set refused).
- **Resolved arm layouts (MEASURED parameter counts, `recur64 v420 model-info`):**

| arm | unique board blocks | applications at training | parameters |
|---|---|---|---|
| R1 | 1 | 1 | 4,736,266 |
| RK2 | 1 | 2 | 4,736,266 |
| RK4 | 1 | 4 | 4,736,266 |
| RV | 1 | uniform 1..4 per update (one K per update) | 4,736,266 |
| WIDE | 4 | 4 | 14,590,696 |
| BASE8 (S4 pipeline check, seed 5101 only) | 8 | 8 | 27,729,936 |

  Shared-arm parameter counts are identical (K is not a parameter). BASE8 = V4's base 27,731,859 minus the 1,923-parameter WDL head
  (`640*3+3`), which V420 omits because only root logits are used.
- **Resolved choices:** headline cell **KQRvK M3** (the hardest V4_TRAIN_DEV cell in V4 Stage A: top-1 0.637 on seed 5101; the data has
  cells M1..M3 only, so M3 is the longest proof depth; 507 dev positions). Seeds 5101/5102/5103 (BASE8: 5101). Evaluation depths: shared
  arms K in {1,2,4,8,16}; unshared arms only at their block count. RV is evaluated at K=4,8,16 for S2. Bootstrap seeds `0x74200001` (S1),
  `0x74200002` (S3); 20,000 resamples (the V4 `paired_bootstrap`). Recipe: 2,000 updates, peak LR 3e-4, warmup 80, batch 128, micro 64,
  `per_update_cell_balanced_v1` on V4_TRAIN_FIT, CUDA FP32, shared AdamW contract; every arm sees the same batches for a given seed.
  The run directories hold final weights and a history only; they are **not resumable** (no optimiser state) and say so.
- **S0 operational definition:** every required run has `run-summary.json` with all losses finite, the health rule passed and 2,000
  updates, and no `run-aborted.json`. A missing, aborted or unhealthy run stops `measure` before any evaluation. Aborted runs write
  `run-aborted.json` (reason, health records) so a failure stays visible.
- **TESTED (CPU, tiny geometry; `cargo test -p recur64-v420`, 11 passed, 0 failed):** parameter count independent of K, a wider
  unshared model has more; the shared block is actually applied K times (K=2 differs from K=1); the shared block receives a gradient on
  every tensor at K=1,2,4 and its gradient norm changes with K (the AGENTS.md recurrence rule); an unshared model refuses a mismatched K;
  autodiff and inference copies agree; the abort rule matches the frozen thresholds; the five gate scenarios behave as pre-registered
  (flat curve stops, a gain in every seed passes, one non-gaining seed fails, collapse at K=16 fails S2, a failed pipeline check and S0 take precedence).
- **MEASURED (engineering, RTX 2000 Ada, CUDA FP32, `docs/evidence/v420/v420-cuda-smoke.json`, 4 updates per arm, not a result):**
  steady update wall seconds and sampled peak VRAM — R1 about 0.14 s (first updates include kernel compilation), RK2 0.135 / 879 MB,
  RK4 0.223 / 1,135 MB, RV 0.152 / 1,135 MB, WIDE 0.236 / 1,327 MB, BASE8 0.427 / 2,351 MB. The first-update loss of every arm is
  about 3.46, matching V4 Stage A's first-update loss (3.45–3.46), as a pipeline consistency indication only.
- **Compute estimate for P2 (INFERRED from the smoke):** about 270 s (R1), 270 s (RK2), 450 s (RK4), 300 s (RV), 480 s (WIDE) per seed
  = about 1,770 s per seed, 5,300 s for three seeds, plus BASE8 about 860 s and evaluation: about 1.8 GPU-hours, inside the 3-hour ceiling.
  Stop and ask if the first full run implies more than 3.
- **NOT RUN:** any training run, any evaluation on dev, any gate. HOLDOUT_C and `v4_tune_v1` untouched.
- **P2 launcher:** `scripts/v420_run.sh` (fail-stop; BASE8 first, so the pipeline check S4 is known after about 15 minutes).

## V420-E2 — P2: result `STOP_DEPTH_DOES_NOT_HELP`
- **Date:** 2026-11-11. **Status:** MEASURED, applying the rules frozen in the plan and amendment A1 once.
- **Commands:** `scripts/v420_run.sh` (BASE8 seed 5101, then R1/RK2/RK4/RV/WIDE for seeds 5101-5103, then `recur64 v420 measure`).
  Started 12:05, finished 13:52 local (about 1.8 h, inside the 3 GPU-hour ceiling). No run was aborted, repeated or discarded.
- **Result:** S0 PASS; S4 PASS (BASE8 0.8474 vs 0.8403); S1 FAIL (RK4 − R1 mean +0.0217, CI [+2.7e-18, +0.0427], per seed 0.0000 / +0.0335 / +0.0316,
  below the +0.03 floor); S2 passes but is moot; S3 OK. Verdict `STOP_DEPTH_DOES_NOT_HELP`. Details: `docs/V420_RESULTS.md`.
- **Consequence:** per the plan, V420 stops. No retry, no tuning, P3 (halting) and option 2 not authorised. Decision V420-D1.
