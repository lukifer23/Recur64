# Workstation V2.5 - summary and review

Branch `experiment/workstation-v25` (base main `fef1ffc`; main is untouched). Read this first. Every
number below is taken from a committed evidence file under `docs/evidence/v25/`; the pre-registered
rules are in `WORKSTATION_V25_EXPERIMENTS.md` and `WORKSTATION_V25_P25_PLAN.md`, the run-by-run records
in `WORKSTATION_V25_EXPERIMENTS.md` and `WORKSTATION_V25_P25_RESULTS.md`. MEASURED means produced by a
command in this repo; INFERRED means reasoning that was not tested.

## 1. The question and the one-paragraph answer
Mainline Recur64 learns but does not convert won endgames (self-play drifts to 70-80% draws; the value
head is calibrated to its data, so the bad value follows bad trajectories). V2.5 asked whether a
substantially stronger ONE-PASS model can learn the technique if it has more capacity, legal moves as
first-class tokens, exact per-candidate facts, and exact near-mate policy supervision before any self-play.

**Answer.** Exact CandidateFacts are the one ingredient that clearly helped: they solve mate-in-one
(1.000 vs ~0.63) and lift deeper mates, most of all through the candidate-token architecture. Candidate
tokens without facts did not beat the matched-capacity legacy head. Five times more unique exact data
closed the train/held-out gap without raising held-out accuracy. No configuration reached the pre-registered
M2 gate (0.75): the best one-pass model plateaus near 0.66-0.69 on M2 and 0.60-0.66 on M3. P3 (conversion)
was therefore never authorized, and the lineage stops here for review and V3 design.

## 2. What was built (all on this branch)
| area | what | where |
|---|---|---|
| architectures | `candidate_v25` (C0/CF, 27,469,204 params), `legacy_facts_v25` LF (26,810,584), control L (26,809,944, the existing `ProbeModel` at width 640), historical F10 (9,805,672) | `recur64-model/src/{candidate,legacy_facts,net}.rs` |
| identity | architecture id + per-architecture contracts in checkpoint metadata; 3x3 cross-refusal; historical hashes unchanged | `config.rs`, `checkpoint.rs` |
| facts | `CandidateFactsV1` (8 exact one-ply facts, legal-action order), evaluator/inference plumbing | `recur64-core/src/candidate_facts.rs`, `recur64-search`, `inference.rs` |
| proof data | `ProofTargetsV1`, exact mate solver, independent audit, exhaustive pools, holdouts, scaled set | `recur64-runtime/src/proof/` |
| training/eval | policy-only trainer, cell_balanced_v1 sampler, macro metrics, paired bootstrap, seed-identity compare, interaction analysis | `proof/{train,sampler,compare}.rs`, `accum.rs` |
| tooling | `proof ...` CLI, `v25-qual` (CUDA qualification), `model-info` | `recur64-cli/src/{proof_cli,v25_qual,model_info}.rs` |

## 3. Engineering gates (all MEASURED)
- fmt clean, clippy 0 warnings, 302 workspace release tests pass; the frozen P4.5 hash and Probe/F10
  identity tests are unchanged.
- CandidateFacts: field-identical to an independent `apply()`+FEN-material reference on 41,353 random-game
  positions plus perft/edge fixtures; 131,725 positions/s; 1.4% of a batch-32 evaluation wall.
- CUDA (RTX 2000 Ada, FP32, no fusion/autotune/TF32): device guard, finite forward/training/lifecycle;
  forward ~1,250 positions/s at batch 32-64; learner 0.92 s/update at effective batch 256 (64x4), ~2.6 GB
  peak; one 400-update run ~6 min. GPU utilization averaged ~85% with dips (deferred to P4 scheduling).
- Exact data: every proof position used (about 74,000 across the retired and replacement splits, the holdouts and the scaled set) was
  re-derived by an independent implementation with 0 disagreements; splits/holdouts are disjoint by
  canonical class and exact FEN; generation is deterministic across thread counts.

## 4. Results (heavy-family numbers are top-1; chance is ~0.06)
**P2** (replacement CONFIRM, all families, seed-averaged):
| | M1 | M2 | M3 |
|---|---:|---:|---:|
| L (legacy head) | 0.661 | 0.650 | 0.631 |
| C0 (candidate tokens, no facts) | 0.629 | 0.622 | 0.604 |
| CF (candidate tokens + facts) | **1.000** | **0.689** | **0.659** |
Gates: Q1 (facts path) PASS; Q3 NOT MET (M2 0.689 < 0.75; M3 0.659 passes 0.55). Extension untriggered.

**P2.5-F** (HOLDOUT_A, heavy only, never used for training/selection; seed-averaged top-1):
| | no facts | facts |
|---|---|---|
| legacy policy | L: M1 .658 M2 .603 M3 .564 | LF: M1 .744 M2 .612 M3 .576 |
| candidate-token policy | C0: M1 .614 M2 .554 M3 .543 | CF: M1 1.000 M2 .686 M3 .601 |
M2+M3 top-1: L .583, LF .594, C0 .548, CF .643. Paired effects (B minus A, 95% CI, both seeds agree in sign):
architecture without facts C0-L **-0.035** [-0.046, -0.024]; facts in candidate CF-C0 **+0.095** [+0.081, +0.108];
facts in legacy LF-L +0.010 [+0.003, +0.017]; architecture with facts CF-LF **+0.050** [+0.036, +0.064];
interaction (CF-C0)-(LF-L) **+0.085** [+0.070, +0.100]. Pre-registered selection rule: CF.

**P2.5-D** (HOLDOUT_B; CF, matched seeds, 1k vs 5k unique heavy positions per cell; P25_DATA_V1 = 44,332 positions):
| | M1 | M2 | M3 |
|---|---:|---:|---:|
| CF, 1k heavy data | 1.000 | 0.662 | 0.624 |
| CF, 5k heavy data | 1.000 | 0.662 | 0.634 |
Scaled minus 1k, M2+M3 top-1: **+0.005 [-0.004, +0.015]**, both seeds positive but the CI includes 0: the
pre-registered unique-data signal is NOT met. The absolute gate fails on M2. No catastrophic KQ/KR regression.

## 5. What the evidence says (by hypothesis)
- **Exact CandidateFacts (supported, strong).** They turn M1 from ~0.63-0.69 to 1.000 and add a
  significant, seed-consistent gain on M2/M3. Facts alone are not enough to reach the M2 gate.
- **Candidate-token representation (supported only in combination).** Without facts it is slightly WORSE
  than the legacy head (-0.035). With facts it is clearly better than the legacy-plus-facts variant
  (+0.050), and the interaction is large and positive. So the candidate architecture's value is specific to
  integrating facts. **Caveat (MEASURED, unresolved):** LF reached M1 of only 0.74 and was UNDERFIT (M1 about
  equal on TRAIN/TUNE/holdout and still rising at update 400), so LF learned facts far too slowly through its
  scalar-delta path. The interaction therefore compares CF against LF-as-parameterized-and-trained-here, not
  against "legacy plus facts" in principle.
- **Unique data (not the limiter).** With 1k data the model fit its heavy training positions better than
  held-out ones (M2 gap ~+0.07, partly memorization); with 5k the gap closes (~+0.03) and held-out accuracy
  does not move. More unique data removed memorization without raising the level.
- **Optimization horizon and capacity (NOT tested).** The data result implies the plateau comes from what the
  one-pass model learns in this budget (optimization length/schedule and/or capacity and depth), but nothing here
  separates those. The test that would (an 800-update run) was deliberately removed from scope.
- **Conversion, value learning, and the original H4 (NOT tested).** P3 was never run, so nothing here shows
  whether exact-proof competence transfers to converting real games, or whether better conversion trajectories
  would restore value learning. F10 was never scored on the proof targets, so the capacity effect (27M vs 9.8M)
  is not isolated either.

## 6. Caveats a reader should keep in mind
1. Two model seeds per cell. The paired CIs are over positions; seed variance is shown per seed but is thinly sampled.
2. The LR (3e-4) is the top edge of the screened grid in both P1a and P1b; it was not expanded by design.
3. The replacement CONFIRM is "sealed" as a set, but 47 of its 1,208 positions (3.9%) were in a retired CONFIRM that a
   disposable toy model scored once (shared pools; unavoidable). It is used only for P2 and descriptive checks.
4. KQvK/KRvK have no fresh holdout (their exact pools are fully partitioned), and their per-cell CONFIRM counts are small
   (e.g. KRvK M1 = 18); per-family numbers there are noisy.
5. KQQvK M3 has only 1,831 unique TRAIN positions (pool limit), so the "5k vs 1k" contrast is diluted in that one cell.
6. Heavily oversampled small cells (KRvK M1 ~45 local epochs at 400 updates) can memorize; TRAIN-vs-TUNE gaps were small.
7. The pooled depth metrics are the gates; macro-cell metrics are diagnostics (macro CF ~0.79 vs pooled 0.77 on CONFIRM).

## 7. Integrity ledger (every deviation, none hidden)
- Toy-model pipeline smoke touched the first CONFIRM -> the whole split assignment was retired and regenerated.
- A P2 launch began before an owner addendum arrived -> stopped, outputs quarantined unread, void.
- A runner-script error (PowerShell stderr handling) killed the first frozen P2 attempt before any result -> fixed, same cells rerun.
- A determinism bug in exhaustive generation (thread-order-dependent FEN) -> caught by a repeat run, fixed, guarded by a test.
- `SyncEvaluator` silently fell back to a uniform policy on bad mass -> now refuses.
- LF contract 1 had an inert final bias -> removed (contract 2); the contract-1 training runs (started before an owner
  addendum arrived) are quarantined unread and void; HOLDOUT_A/B/C had been generated (never evaluated) before that addendum.
- The LF same-seed regression test failed when parallel tests shared Burn's global RNG -> moved to its own process.
- The P2.5-O optimization-horizon test was cancelled by the owner before any P2.5 science; HOLDOUT_C is reserved and unused.

## 8. Inputs for the V3 decision (options, NOT actions taken)
The evidence supports building on exact facts integrated through a richer interface than a scalar logit, and it
rules out "just add unique exact data" as the next lever for a one-pass model. Untested levers, each a separate
pre-registered experiment if pursued: a longer/differently scheduled optimization run; more capacity or depth; a
faster-learning fact-delta parameterization (to resolve the LF confound); and, only if exact-technique competence is
judged sufficient, a conversion test (P3) with a corrected exploration contract. The HP-branch planner/tool line and
the mainline conversion/value evidence remain the other inputs.

## 9. Reproduction
Commands are listed in the README ("Workstation V2.5"). Datasets, checkpoints and logs live in the untracked `runs/`
(`runs/v25/{proof,proof-v2,p2,p25}`); the digests that identify them are in the ledger and evidence files, and every
dataset regenerates bit-identically from its seed. Start points: `configs/v25/*.toml`, `docs/evidence/v25/`.
