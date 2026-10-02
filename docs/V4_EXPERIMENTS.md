# Recur64 V4 - Experiment Ledger (append-only)

Labels: PRE-REGISTERED / MEASURED / INFERRED / NOT RUN. Never edit past entries; append corrections as new entries.

## V4-E0 - branch creation and pre-registration

- **Date:** 2026-10-02
- **Status:** MEASURED (repository facts); no science.
- **Commands:** `git checkout -b experiment/workstation-v4-evidence-belief 40916509936e83468e2fe58c455a33a246123545`
- **Result:** branch created from a clean tree. V3 and V3.5 branches unchanged.
- **Pre-registered here (docs/V4_RESEARCH_PLAN.md):** laws A-D; module contracts; staged protocol; mechanism questions A-G
  and pass rules; TRAIN-dev partition (`sha256("v4_train_dev_v1|"+canon)` u64 mod 10 == 0 -> DEV); `V4_TUNE_V1` rule
  (KQRvK/KRRvK x M1/M2/M3, 1,000/cell, seed `0x7A40_0001`, disjoint by FEN and canonical class from all prior sets).
- **NOT RUN:** any V4 model, data generation, training, or evaluation. V4_TUNE_V1 not generated. HOLDOUT_C not evaluated.

## V4-E1 - V4-B build: architecture, checkpoint identity, invariants (engineering, no science)

- **Date:** 2026-10-02
- **Status:** MEASURED (CPU, FP32, tiny test geometry for behaviour; full geometry for parameter count). No training data
  was used and no V4 mechanism question was answered.
- **Commands:** `cargo test -p recur64-v4`, `cargo test --workspace` (before the V4 custody additions), `cargo test -p recur64-runtime --lib v4`.
- **Result:**
  - New crate `recur64-v4`; new identity `evidence_belief_v4` (`Architecture::EvidenceBeliefV4`, `EvidenceContracts`,
    `CheckpointMeta.evidence_contracts`); `ModelConfig::refuse_evidence_v4` is called from `bench`, `cuda-smoke`, the mainline
    runtime config, and `proof train/eval`, `v25-qual`, `model-info` bail on the new variant.
  - Full-geometry parameters: **30,023,684** (base 27,731,859 = `root.*` + `base.readout`; evidence 820,328; utility path 1,471,497;
    breakdown printed by `parameter_count_is_independent_of_the_budget_and_groups_sum_exactly`). Budget-independent.
  - 19 invariant tests pass (`crates/recur64-v4/tests/invariants.rs`): B0 independent of budget and bitwise `z0`; zero
    content => bitwise-zero message => bitwise identity update for any routing metadata; permutation invariance (acquisition
    order and ledger slot order, tolerance 1e-5); real vs zero vs shuffled content differ; deltas finite and bounded
    (extreme content x1e4); exact StateQuery accounting incl. leak detection; utility scoring executes no query and the
    `FrontierEdgeView` has no child field; probe forks leave the real manager/tree/ledger/belief bit-identical and the
    final trajectory identical with and without probes, `probe_queries` counted separately; V4 refused by every historical
    identity and vice versa (metadata level + checkpoint round trip + tampered contract); frozen base gets no gradient,
    evidence gets finite non-zero gradient, B0 bit-identical after optimiser steps; zero-content replay along a deep
    reply chain returns exactly to B0.
  - 5 training-plumbing tests pass (`tests/training.rs`).
  - Workspace regression: 513 tests passed, 0 failed (`cargo test --workspace`, run before the V4 custody edit).
- **Defect found and fixed during this entry:** with `Freeze::BASE_AND_EVIDENCE` the pooled root context
  (`BaseStage::root_node`) was still attached to the base graph, so a utility loss could have trained the base root
  encoder. Caught by `utility_losses_train_only_the_utility_path`; fixed by detaching it with the tokens and `z0`.
- **NOT RUN:** CUDA (no GPU test yet), the CLI boundary test for V4, any training on P25 TRAIN, `v4_tune_v1` generation.

## V4-E2 - PRE-REGISTERED: TRAIN-only mechanism-study protocol (frozen before any V4 training run)

- **Date:** 2026-10-02
- **Status:** PRE-REGISTERED. Written and committed before any Stage A/B/C run. The pass rules for questions A-G are those of
  `docs/V4_RESEARCH_PLAN.md` section 7 and are not repeated or loosened here.
- **Data:** P25 TRAIN only, split by the frozen `v4_train_dev_v1` rule into `V4_TRAIN_FIT` (training) and `V4_TRAIN_DEV`
  (measurement). `V4_TUNE_V1` is never evaluated; HOLDOUT_C is never loaded. Seeds 5101, 5102, 5103.
- **Common:** AdamW (`OPTIMIZER_CONTRACT`), linear warmup + cosine (`learner::lr_at`), FP32, per-update cell-balanced
  sampler over FIT (`sampler = per_update_cell_balanced_v1`: a fresh `CellSampler` seeded `mix(seed, update)` each update, so
  resuming needs no sampler state), query budgets never above 8, no job over 2 hours.
- **Stage A (base):** 2,000 updates, warmup 80, peak LR 3e-4, batch 128 (micro 64), budget 0 only. Recorded, not gated.
- **Stage B (evidence), base detached (`Freeze::BASE`):** init from the seed's Stage A final weights, fresh optimiser.
  Budget cycles `[2, 4, 8]` per update; selection is `fixed_bfs_actionid_v1` on even updates and `Random(mix(seed, update))` on
  odd updates (label-independent). Batch 128 (micro 32), 1,200 updates, warmup 80.
  **LR screen (TRAIN-only):** peak LR in {3e-4, 1e-3, 3e-3}, seed 5101, 400 updates each, chosen by the lowest mean
  `V4_TRAIN_DEV` CE at B8 averaged over the FIXED and RANDOM evaluation schedules; ties go to the lower LR. The chosen LR
  is then used for all three seeds. No other tuning.
- **Stage C (utility), base and evidence detached (`Freeze::BASE_AND_EVIDENCE`):** init from Stage B finals. Probed
  partial states: a label-independent `Random` prefix of 0, 1, 2 or 3 real queries (cycled per update), session budget 8,
  K = 8 probes per state (the head's argmax over the whole frontier first, then a hash-deterministic fill, rule
  `choose_picks`). 300 updates, batch 32 (micro 16), warmup 30, peak LR 3e-4. **Loss selection (bounded study):** train
  `ranking` and `regression` with identical data and seeds; the loss with the higher pooled DEV mean per-state Spearman is
  frozen as the V4 selector contract; if the two differ by less than 0.02 the `ranking` loss is chosen (scale-free). No
  sign auxiliary, no stacking.
- **Measurement:**
  - Stage B on all of `V4_TRAIN_DEV`, schedules FIXED and RANDOM (`mix(0x7A40E001, ...)`), budgets 2/4/8, with replays of the
    recorded FIXED/RANDOM paths under zero and shuffled content. A, B, G must hold under BOTH schedules (the stricter reading);
    C is bitwise equality of per-position CE between zero-content replay and B0.
  - Stage C on `DEV1000` = the 1,000 `V4_TRAIN_DEV` positions with the smallest `fnv1a(id)`; prefixes 0-3; probe seed
    `0x7A40E002`. Answers D, E, F. Label noise is checked by re-probing 50 states (must be exactly equal).
  - Bootstrap: 20,000 resamples of positions (states for Stage C), seeds kept together, SplitMix64, ranks 499 / 19,499.
    Seeds: A `0x7A400101`, B-zero `0x7A400102`, B-shuffled `0x7A400103`, G-top1 `0x7A400104`, E-spearman `0x7A400105`,
    E-pairwise `0x7A400106`, F `0x7A400107`.
- **Stage D:** a smoke-scale report only (no gate): `Selection::Utility` vs FIXED vs RANDOM on `V4_TRAIN_DEV` at B2/B4/B8, plus
  the B0 bit-identity check across all stages.
- **NOT RUN:** everything above. Multi-step (BMPS complementary-computation) utility diagnostics are NOT planned for this
  pass and will be reported as NOT RUN.

## V4-E3 - `v4_tune_v1` generated, audited and sealed (custody only; NEVER evaluated)

- **Date:** 2026-10-02
- **Status:** MEASURED (generation and custody). The rule, the seed `0x7A40_0001` and the 1,000-per-cell size were committed
  in V4-A (`docs/V4_RESEARCH_PLAN.md`, commit `37d2139`) before this run; `generate_v4_tune` and its constants were committed
  in V4-B before this run. No count was reduced.
- **Command:** `recur64 v4 tune-gen --inventory-dirs runs/v25/p25/data,runs/v25/p25/holdouts,runs/v25/proof,runs/v25/proof-v2,runs/v25/proof-v2-check,runs/v3/data --data-dir runs/v4/data --manifest docs/evidence/v4/v4-tune-v1-manifest.json --seal-output docs/evidence/v4/v4-tune-v1-seal.json --holdout-c runs/v25/p25/holdouts/proof-holdout_c.json --threads 12`
- **Result:**
  - 6,000 unique positions, **1,000 in every cell** (KQRvK and KRRvK x M1/M2/M3); digest
    `b83624e4495d6586622aad3b523789dd065e1c398f391403743468796f7aaf77`; independent audit 6,000 checked, 0 failures.
  - Hard-disjoint by exact FEN and canonical class from every inventoried dataset: 15 files (P25 TRAIN/TUNE, HOLDOUT_A/B/C,
    the V2/V2.5 proof sets, `proof-v2-check`, and V3_TUNE_V1); 75,124 excluded canonical classes and exact FENs; exclusion
    manifest digest `8c2be0e0528d0442da6bd497dc375b031c849a7f30471f6477ce6ce160c64014`; 0 overlap; every pair also checked
    with `check_disjoint`. Every cell had a large eligible pool (for example KQRvK M1: 102,639 available of 111,273), so the
    frozen rule was feasible with no relaxation.
  - Deterministic regeneration from the seed gave the identical digest. Wall 110 s.
  - Sealed: `sealed = true`, `evaluated = false` (`docs/evidence/v4/v4-tune-v1-seal.json`). `recur64 v4 tune-verify`
    re-verified the seal and HOLDOUT_C's frozen digest `4ab951c6...d87d5` (`docs/evidence/v4/v4-custody-verify.json`):
    HOLDOUT_C sealed, unevaluated.
  - The dataset file itself is in git-ignored `runs/v4/data/`; every V4 training and mechanism path loads only P25 TRAIN
    (`V4Data::load` refuses any other digest) and `load_sealed_v4_tune` needs a `V4TuneAuthorization` for phase `V4-FINAL`.
- **Limitation (inherited):** HP/X1/X2 exact datasets are not present on this workstation in a compatible form, so
  disjointness from them was NOT verified (same limitation as V3_TUNE_V1).
- **NOT RUN:** any evaluation on `v4_tune_v1`; HOLDOUT_C was only digest-verified.

## V4-E4 - CUDA correctness smoke and cost measurements (engineering, no science)

- **Date:** 2026-10-02
- **Status:** MEASURED on the RTX 2000 Ada (16 GB), `recur64` release build with `--features cuda`, FP32 (no TF32).
- **Commands:** `recur64 v4 cuda-smoke --train runs/v25/p25/data/proof-train.json --output docs/evidence/v4/v4-cuda-smoke.json`;
  `recur64 v4 bench --train runs/v25/p25/data/proof-train.json --device cuda --batch 16 --repeats 10 --output docs/evidence/v4/v4-bench-cuda.json`.
- **TESTED (the real graph ran on the GPU, 30,023,684 parameters):**
  - CPU/CUDA forward parity from identical weights (saved on the CPU backend, loaded on CUDA), 16 TRAIN positions: max
    |logit difference| 5.4e-5 at B0 and 5.5e-5 at B4 (tolerance 2e-3), argmax equal everywhere.
  - A CUDA backward + AdamW update with the base detached (3 updates, finite losses): B0 per-position CE bit-identical before
    and after; the evidence delta is non-zero. A CUDA checkpoint round trip (`Trainer::save` -> `Trainer::load`) restores B0 exactly.
  - Not substituted: `verify_device` errors visibly if CUDA cannot run kernels.
- **Cost (batch of 16 positions, `fixed_bfs` selection, per batch):** B0 18.7 ms; B2 24 ms; B4 53 ms; B8 41 ms (timings are
  noisy at this size; the marginal cost per query step is about 3-9 ms per batch, dominated by host-side StateQuery and the
  per-step host synchronisation); utility selection B4 113 ms; counterfactual probes about 940 probe queries per second.
  GPU busy mean 29-60% (host-bound at this batch size, as for V3).
- **Training steps (full geometry):** Stage A update (batch 128, micro 64) 0.5-1.1 s after a 2.5 s cold start, peak VRAM 2.5 GB;
  Stage B update (batch 128, micro 32, base detached) 0.8-0.9 s after a 3.5 s cold start, peak VRAM sampled 10.6 GB (in a
  process that had already run Stage A; cubecl's allocator does not return memory).
- **Defect found and fixed during this entry (cost, not correctness):** the first benchmark showed 14.8 GB peak for Stage B
  because every query step also ran the utility path's parent/action state encoder under autodiff even when the selection
  (FIXED/RANDOM) never reads it. That path is now opt-in (`RunOptions::with_state`; required for utility selection, utilities
  and probes), the belief is bit-identical with it on or off (test), and the Stage B peak fell to 10.6 GB.
- **NOT RUN:** B16, sustained-load VRAM plateau, TF32/BF16 (not part of V4).
