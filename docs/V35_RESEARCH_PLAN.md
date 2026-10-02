# V3.5 plan: on-policy information-acquisition rescue (`on_policy_proof_relabel_v1`)

**Status: PRE-REGISTERED (2026-10-02). Frozen before any V3.5 checkpoint is evaluated on V3_TUNE_V1.**
Decisions: V35-D1 to V35-D5 (`docs/DECISIONS.md`). Experiment record: V35-E1 (`docs/V3_EXPERIMENTS.md`).
The resolved physical layout and the final per-seed recipe digests are frozen in commit V35-B (TRAIN-only preflight),
still before any TUNE evaluation of a V3.5 model.

## 1. The question

> Can the existing `active_search_v3` architecture learn to select and USE useful queried state information when trained
> entirely on its own query trajectories, with the proof oracle used only to label learner-visited states and never to
> choose a query?

This is the one disciplined DAgger-style rescue motivated by V3. It is not an architecture redesign and not V4. If it
fails, the lineage stops and a V4 design memo (no implementation) is the output.

## 2. Lineage

- Repository `https://github.com/lukifer23/Recur64`. Source branch `experiment/workstation-v3-active-search`, source HEAD
  `7e508df8d369fb34c58990774213bc51c762c504` (verified against the remote before branching).
- New branch `experiment/workstation-v35-onpolicy`, created from exactly that commit. Nothing is merged into it (no main,
  V2.5, HP or unrelated work). Scientific V3.5 work is never committed to the V3.0 branch.

## 3. Accepted V3 evidence (immutable, not reinterpreted)

- P4: KQRvK M3 `C_8 = 2168/5000 = 0.4336` vs threshold 0.25: qualified for the B8 primary experiment.
- P5: contract `105ac313...009d2`; selected peak LR `3.0e-4`; selected recipe (no seed)
  `a069ba9d18befed65f970aca253b47780365fd7019be38283f270d79d6c1db33`; final selected-recipe checkpoints for seeds 5101, 5102
  (screen) and 5103 (P6 baseline replication).
- P6 Gate I: PASS, `Delta = +0.2351`, CI [0.2044, 0.2671], threshold +0.20 (raw exact future-state information is sufficient
  for this model family). ALL-INFO weights and outputs are NOT used by V3.5 in any way.
- P5.2: the existing ACTIVE model does not productively use queried-state content; teacher-forced accuracy was query-pattern
  leakage. This is why V3.5 exists.

## 4. Data and custody (unchanged)

TRAIN P25_DATA_V1 `3b25dc85...f2e6` (44,332 positions; trace manifest `8160734e...c488`). TUNE V3_TUNE_V1
`c6601865...1b53` (4,500 positions; trace manifest `acb786fa...6d25`). HOLDOUT_C `4ab951c6...d5d` stays SEALED; no code path
of V3.5 loads it (the loaders call `load_working_split`, which refuses it).

## 5. Architecture (frozen)

`active_search_v3` with no geometry change and no new learned module, head, RL, engine/tablebase label or descendant
CandidateFacts. The only model-crate change is an API separation (section 7); it adds no parameter and does not alter any
existing selection mode.

## 6. Training contract `on_policy_proof_relabel_v1`

- **B0:** root encoder once, no queries, root-policy loss only.
- **B2/B4/B8:** the learner selects EVERY query: current frontier from the live real `Tree`; ACTIVE selector logits; ACTIVE
  argmax (lowest index on ties) picks the edge; ProofTrace computes the oracle target over the CURRENT learner-induced
  queried set `S`; the learner's chosen edge is executed through StateQuery; the real state is integrated; continue.
- **Target at learner states:** `A(S) = A_proof(S) U A_refute(S)`, uniform over every admissible frontier edge, all ties
  retained, a function of `S` and the proof structure only (no serialization dependence). Every target index is a legal edge
  of the live frontier (mapping failure is an error). Incomplete proof: target non-empty. Complete proof (`residual = 0`):
  target empty.
- **The oracle never changes `follow`.** No expert action probability, no teacher/student mixing, no switch to the proof
  teacher, no correction of an off-target action, no scheduled expert continuation, no completion latch, no filler schedule.
  Proof completion only masks the selector loss; ACTIVE keeps spending the forced budget exactly as at evaluation.

## 7. API separation: choice vs supervision

`recur64_model::active::QueryTargetProvider` (returns frontier indices only; no handle on the choice) and
`Selection::ActiveLabelled(&mut dyn QueryTargetProvider)`. The learner chooses exactly as `Selection::Active`; the provider is
called after the choice is fixed and cannot influence it. `ActiveOutput::chosen` records the learner's frontier index per
step. The existing `QueryScript` (which lets one object choose AND label) is not used by V3.5 training; it remains only for
the replay of recorded learner edges (Pass B) and for historical P5 code. Tests prove: ACTIVE with a provider is identical
to ACTIVE without one (chosen edges, query records, final policy bit-for-bit); arbitrary labels leave every forward value
unchanged and change only the selector loss/gradient.

## 8. Two-pass update

Per optimizer update, weights frozen:

- **Pass A (detached ACTIVE rollout)** on the inference copy of the model (`model.valid()`): real StateQuery and contents;
  records every learner-selected edge, every prefix target, accounting, queried-tree statistics and the final policy.
  `N_supervised` for the whole update is known exactly after Pass A.
- **Pass B (autodiff replay)** replays the recorded learner edges (`Selection::Script` with the recorded follow and targets)
  through the autodiff model with real content. Before any backward the update asserts the replay reproduces Pass A: same
  query records (parent slot, action, branch, depth, terminal, frontier size), same successful-query counts and depths, same
  tree accounting, same selector-decision count, and the final policy within a frozen absolute log-prob tolerance of `1e-4`.
  A violation refuses the update (visible error, never a silent fallback). The old model-free proof-teacher dry simulation is
  not used.

## 9. Loss

`L = mean root-policy CE + 1.0 * mean selector NLL`, no WDL. Policy: exact uniform root target over the correct set, final
forced-budget readout only, mean over every example of the optimizer update. Selector: mean over every supervised
learner-visited decision of the optimizer update (never per microbatch; B0 does not dilute it). Each microbatch is
back-propagated once with the final weights and accumulated, so the accumulated gradient is the monolithic gradient.

## 10. Initialisation, optimizer, schedule

- Each seed initialises from the WEIGHTS ONLY of its corresponding selected P5 final checkpoint: seed 5101 ->
  `runs/v3/p5/v3-p5-run-lr3e-4-seed5101/final` (P5 recipe digest `c9d90281...61a1`); 5102 -> `.../seed5102/final`
  (`20b30924...0802`); 5103 -> `runs/v3/p6/v3-p5-run-lr3e-4-seed5103/final` (`92384763...eab0f4`). The loader verifies the
  sidecar recipe digest, that the recipe with its seed cleared equals the selected digest `a069ba9d...`, the seed, the peak
  LR, 800 completed updates, the model config and the checkpoint metadata. NOT from ALL-INFO; no checkpoint averaging; no
  ensemble.
- **Fresh `adamw-v1`** (the discarded optimizer state is never loaded). Reason: the optimisation distribution and objective
  presentation change materially; V3.5 inherits P5 weights, not P5 optimizer momentum.
- Peak LR `3.0e-4`, warmup 80, 800 updates, linear warmup + cosine (`learner::lr_at`), FP32, health checks on. No LR screen,
  no tuning after TUNE results, no early stopping, no best checkpoint.

## 11. Budgets and sampling

Training budgets `{0,2,4,8}`; B16 is never trained. Preferred layout micro16 x accum8 (effective batch 128), budget
sequence `[0,2,4,8,0,2,4,8]`, 32 examples per budget per update, four independent `cell_balanced_v1` samplers with the
P5 seed rule (so each seed draws the same position sequence per budget as its P5 run; recorded, not hidden). A TRAIN-only
preflight may establish the execution-only fallback micro8 x accum16 BEFORE any V3.5 TUNE evaluation. The layout is never
chosen from TUNE.

## 12. Gates (all at update 800; primary cell KQRvK M3; seeds 5101/5102/5103; the SAME checkpoint for every arm)

All estimators: for each M3 position `i` and seed `s` a paired value `v[s][i]`; `d_i = mean_s v[s][i]`; point estimate
`mean_i d_i`; 95% percentile bootstrap CI over 20,000 resamples of positions with all seed pairs kept together, SplitMix64,
ranks 499 / 19,499 of the sorted resample means (the Gate I machinery, `p6::gate`). New fixed seeds: Gate II `0x7A350002`,
Gate III `0x7A350003`, Content-Use `0x7A350004`.

- **Gate II (same-weight compute):** `v = top1(ACTIVE_B8) - top1(B0)`. PASS iff pooled delta >= +0.10 AND CI lower > 0 AND
  every seed delta > 0. Also reported: top-1, correct mass, CE, entropy.
- **Gate III (learned selection):** `v = top1(ACTIVE_B8) - top1(FIXED_B8)` with FIXED exactly `fixed_bfs_actionid_v1`. PASS iff
  pooled delta >= +0.05 AND CI lower > 0 AND every seed delta > 0. Secondary comparators are diagnostic only.
- **Content-Use (new, required):** `query_content_ablation_v1` replayed on the exact ACTIVE B8 query paths;
  `v = CE_ablated(i) - CE_normal(i)` (positive: the real queried content improved the log-likelihood of the correct target).
  PASS iff pooled mean > 0 AND CI lower > 0 AND every seed mean > 0; no magnitude threshold. Also reported: normal vs ablated
  top-1, correct mass, entropy, positions whose top-1 changes, B2/B4 secondary content effects, and the attribution ratio
  `(normal_B8 - ablated_B8)/(normal_B8 - B0)` (top-1) where `normal_B8 > B0`. The ablation replay must reproduce the ACTIVE
  source CE to < 1e-4 or the evaluation is refused.
- **Gate VI (stability):** finite outputs and gradients over all 800 updates; health checks on (bounded workspace/branch RMS,
  exact query counts, no duplicate edges, root encoder once, query-encoder rows = successful queries, enforced by the model's
  per-run invariants); detached-rollout/autodiff-replay parity within 1e-4 for every update; parameter count independent of
  budget (one model); resident lifecycle / VRAM stable: on CUDA, VRAM sampled at every checkpoint, the maximum after update
  100 must be within 5% of the maximum up to update 100. An unmeasured criterion (CPU run, no VRAM samples) is NOT TESTED and
  is not a pass.
- Gate I (historical) is PASS and immutable. These gates do not rewrite historical V3 Gates II/III.

Gate II/III/V had no code on the V3 branch (only Gate I did); their estimators were written and unit-tested in this
commit before any V3.5 measurement.

## 13. Evaluation protocol

TUNE (all 4,500 positions) at updates 0/200/400/600/800: diagnostics at 0-600 (ACTIVE B0/2/4/8 with selector diagnostics), no
early stopping, no best checkpoint, no recipe change after any observation. At update 800: ACTIVE B0/2/4/8, FIXED B2/4/8,
`query_content_ablation_v1` on ACTIVE B2/4/8, selector diagnostics (proof-admissible, refute-admissible, off-target share,
first-query correct-root rate, depth histogram, distinct root branches, residual decrease per query, fraction of queries
decreasing residual, proof completion fraction, final residual, ideal `Q* <= B` ceiling; pre-completion where appropriate),
compared to the accepted P5.2 numbers without creating a post-hoc gate. HOLDOUT_C is not evaluated.

## 14. B16 / Gate V

B16 is never trained. If and only if Content-Use + Gate II + Gate III + Gate VI all PASS, the same checkpoints are run at
B16 and B0/2/4/8/16 are reported (top-1, mass, CE, entropy, wall, query depths, branch coverage, VRAM; ablation at B16 if
cheap). Gate V (`ACTIVE_B16 - ACTIVE_B8 > 0`, CI > 0, every seed positive) supports an extrapolation claim only and is not
required for FULL GO. A tie is no extrapolation claim. If the primary gates fail, stop before expanding.

## 15. Outcome classification (pre-registered, exact)

- **V3.5 FULL GO:** Gate I (historical PASS) + Content-Use PASS + Gate II PASS + Gate III PASS + Gate VI PASS.
- **PARTIAL - COMPUTE:** Gate II passes, Gate III fails (and Content-Use passes). Stop before HOLDOUT_C.
- **PARTIAL - CONTENT:** Content-Use passes, Gate II fails. Stop before HOLDOUT_C.
- **PATH/COMPUTE EFFECT WITHOUT INFORMATION USE:** Gate II and/or III pass, Content-Use fails. No claim of selective
  information acquisition; recommend a V4 content-gated architecture.
- **V3.5 NO-GO:** otherwise (no content use and no useful compute). No second DAgger iteration, no second LR screen, no extra
  400 updates.
- **Gate VI failure** (Content-Use + II + III pass but stability fails): the pre-registration did not enumerate this case.
  It is classified `StabilityFailure` (not a GO) and the owner decides; recorded here before any measurement.

## 16. HOLDOUT_C

Not opened. Only if V3.5 reaches FULL GO on TUNE: stop and report; a later explicit owner instruction may authorise one
P8-equivalent confirmation, pre-registered before exposure (three final checkpoints, B0, ACTIVE B8, FIXED B8, the same-path
content ablation, the Gate II/III calculations and the Content-Use calculation with these exact rules). No tuning after
HOLDOUT_C exposure.

## 17. If V3.5 fails

Return a V4 design memo (not an implementation): path identity may route evidence but must not update belief; query content
must generate an explicit evidence message; zero/ablated content should give an exact or near-exact identity planner update;
structurally content-gated planner/workspace updates; training on learner-induced trajectories from the start with the oracle
only labelling visited states; query-content counterfactual tests as first-class qualification; consider predicting query
utility / residual reduction directly; keep one-edge StateQuery semantics and exact accounting.

## 18. Sequence

V35-A (this commit): plan, API, trainer, estimators, CLI, tests. V35-B: TRAIN-only throughput/duplication pass,
parity-proven execution-only optimisations, CUDA preflight, resolved layout, final recipe digests; pushed before any TUNE
evaluation. V35-C: three runs, gates applied once, compact evidence and results, then STOP. The final statement of V35-C is
`HOLDOUT_C NOT EVALUATED - awaiting owner review/approval.`
