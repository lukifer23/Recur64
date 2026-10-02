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
