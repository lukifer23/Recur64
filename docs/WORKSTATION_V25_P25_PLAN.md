# Workstation V2.5 - P2.5 plan (PRE-REGISTERED before any P2.5 science)

Starting SHA: `729c1e5183f2b36bc86d68288ef2efea51bd2d09` (verified after `git fetch`; main
untouched at `fef1ffc`).

P2.5 is a NEW owner-authorized research lineage motivated by P2 evidence. It is NOT the old
P2 extension and NOT a P2 rescue. P2 remains a formal Q3 NO-GO; its numbers, thresholds and
decisions are not altered. P3 stays blocked until P2.5 produces a new P3' justification.

Subphases: P2.5-F (complete the 2x2 factorial), P2.5-D (unique exact data scaling),
P2.5-O (optional optimization-horizon test). No self-play, conversion, recurrence, planner,
world model, visual path, engine, human data, reward shaping or new losses. GPU performance
is NOT optimized during this sequence.

## Questions
A. Did we choose the wrong action-head inductive bias (C0 < L)?
B. Can the stronger legacy head be combined with the useful exact CandidateFacts (LF)?
C. Is the remaining M2 gap primarily a unique-data / generalization problem?
D. Only after those: does a longer optimization horizon help?

## LF model (`legacy_facts_v25`)
Board encoder identical to L (width 640, 10 heads, FFN 1280, 8 blocks, one pass,
recurrence 1). The legacy head-v2 source/destination/promotion logit is the BASE. Facts add
a candidate-local policy delta:
`final_logit_i = base_logit_i + Linear(64->1)(GELU(Linear(8->64)(CandidateFactsV1_i)))`,
then the usual masked legal softmax. No gain multiplier, no zero-gated path; all facts
parameters receive gradient from update 1. The final fact-delta layer uses a small NONZERO
normal init (std 0.01, the same structural rule as CandidateV25's policy scorer; fixed before
any result, never tuned). Facts do NOT feed WDL (legacy WDL path unchanged). Distinct
architecture/checkpoint identity with its own contract versions; cross-refused in every
direction against probe_v1 and candidate_v25 by architecture id, not by tensor shape. The
historical probe_v1 identity must remain hash-compatible. Expected size: L + 641 parameters.

## Holdouts (heavy families only)
HOLDOUT_A / B / C: families KQQvK, KQRvK, KRRvK; depths M1, M2, M3; target 500 positions per
family x depth cell (4,500 per holdout). Same exact solver, shortest-correct-move target,
<= 15% correct fraction, CandidateFacts ambiguity for M2/M3, symmetry-canonical dedup,
fresh_no_history contract, independent audit. Exclusion is by CANONICAL CLASS of: the retired
P1/P2 TRAIN/TUNE/CONFIRM, the replacement P2 TRAIN/TUNE/CONFIRM, the other holdouts, and every
P2.5 training-extension position. Roles: A = P2.5-F factorial evaluation; B = P2.5-D
confirmation (NOT evaluated before the P2.5-D final endpoints); C = P2.5-O confirmation
(NOT evaluated unless P2.5-O is triggered). Holdout split seeds: A 0x7A130001, B 0x7A130002,
C 0x7A130003. Holdout evaluation prints the exposure guard and is logged.

### Pool-limit rule (written BEFORE the pools are measured under these exclusions)
The exact eligible pool of KQQvK M3 is only 4,409 classes, and two earlier split assignments
already consumed part of it, so that cell may not be able to supply 1,500 holdout positions
plus 4,000 training-extension positions. Rule, for every heavy cell:
1. Holdouts take priority: A, B, C each get 500 (evaluation integrity first).
2. The training extension then takes ALL remaining eligible, non-excluded classes, up to 4,000.
3. A cell whose remaining pool is below 4,000 is reported with its exact unique count; its
   TRAIN total is 1,000 (existing P2 TRAIN) + what is available. Nothing is relaxed: no filter,
   no exclusion, no reuse of a holdout/TUNE/CONFIRM position.
4. If a cell cannot supply even the three 500-position holdouts, STOP and report.
cell_balanced_v1 equalizes per-cell EXPOSURE regardless, so a smaller cell is oversampled
more; this is reported per cell (local epochs).

## P2.5-F
Train LF, seeds 1 and 2, under EXACTLY the final P2 contract: replacement P2 TRAIN
(`1e5e121b...`), TUNE (`f59d744a...`), cell_balanced_v1, LR 3e-4, 400 updates, warmup 40,
cosine over 400, effective batch 256 (64x4), policy-only, FP32. Evaluate on HOLDOUT_A:
L, C0, CF (existing P2 checkpoints, both seeds) and LF (both seeds); also LF on the old P2
CONFIRM for DESCRIPTIVE continuity only (never a selection set).
Factorial effects (paired, seed-averaged per-position bootstrap; top1, mass, -CE; groups
M1, M2, M3, M2+M3, all; per seed):
- no-facts architecture effect: C0 - L; facts effect in candidate: CF - C0;
  facts effect in legacy: LF - L; architecture effect with facts: CF - LF;
- interaction: (CF - C0) - (LF - L) = (CF - LF) - (C0 - L).
LF health expectation: LF M1 >= 0.95; if both seeds fail badly, treat as an
implementation/optimization anomaly and do not select LF until understood.
### Architecture selection rule (primary group: HOLDOUT_A M2+M3 top-1, LF vs CF)
- LF - CF CI wholly > 0 AND both seeds' mean differences positive: SELECT LF.
- CF - LF CI wholly > 0 AND both seeds positive: SELECT CF.
- otherwise: INCONCLUSIVE, and P2.5-D scales BOTH.

## P2.5-D
P25_DATA_V1: the exact existing small-family TRAIN (KQvK 1,570; KRvK 931) unchanged; heavy
cells = the existing 1,000 P2 TRAIN positions + new positions per the pool-limit rule above
(target 5,000 per heavy cell). Every added position audited; excludes every evaluation set.
TUNE stays the replacement P2 TUNE. Keep cell_balanced_v1 (400 updates = 102,400 examples
~ 6,827 per cell), so unique data changes while per-cell exposure stays ~fixed.
Train the selected architecture(s) (both if inconclusive), seeds 1 and 2, from scratch, LR 3e-4,
400 updates, warmup 40, cosine over 400, effective batch 256, policy-only, FP32.
Matched baseline = SAME architecture and seed on the 1k data (LF: the P2.5-F LF runs; CF:
the original P2 CF runs). Both baseline and scaled checkpoints are evaluated on HOLDOUT_B
(never used for training, architecture/LR/threshold selection).
- UNIQUE-DATA SCALE POSITIVE iff (scaled - baseline) on HOLDOUT_B M2+M3 top-1 has mean > 0,
  paired 95% CI wholly > 0, and both seeds positive. No arbitrary minimum size.
- ABSOLUTE GATE (best scaled architecture, HOLDOUT_B, seed-averaged pooled heavy metrics):
  M2 top-1 >= 0.75 AND M3 top-1 >= 0.55; neither seed more than 0.03 below a floor; all
  gradients finite; no severe train/eval collapse; M1 >= 0.95 for a facts-enabled architecture.
- KQ/KR SECONDARY HEALTH (descriptive; old P2 CONFIRM KQvK+KRvK subset; there is no fresh
  KQ/KR holdout because the exact small pools are fully partitioned): CATASTROPHIC iff the
  pooled KQvK+KRvK M2+M3 top-1 drops by MORE THAN 10 percentage points versus the same-
  architecture baseline in BOTH seeds. Not the primary gate.
- Decision: signal positive AND absolute gate passes => "GO FOR P3' QUALIFICATION" (stop and
  report; P3 is not run). Otherwise consider P2.5-O by the trigger below.

## P2.5-O (only if triggered)
TRIGGER (fixed before HOLDOUT_B is read). The best 400-update scaled model fails M2 >= 0.75
and/or M3 >= 0.55 (on HOLDOUT_B) but satisfies ALL of: (1) it is within 0.10 absolute top-1 of
EACH missed floor; (2) training finite/stable; (3) pooled evaluation CE does not show
catastrophic overfit: HOLDOUT_B pooled CE minus the pooled CE over the HEAVY-family TRAIN
positions (from the final full-TRAIN evaluation) <= 0.30; (4) macro-cell CE: the mean HOLDOUT_B
CE over the 9 heavy cells minus the mean heavy-cell TRAIN CE over the same 9 cells <= 0.40.
If it misses a floor by > 0.10 or overfits severely: do NOT train longer; stop.
TRAINING (fresh, NOT a resume, NOT a restarted scheduler): same architecture, seeds 1 and 2,
P25_DATA_V1, cell_balanced_v1, peak LR 3e-4, 800 updates, warmup 80, cosine over all 800,
effective batch 256, policy-only, FP32. HOLDOUT_C stays untouched until the final 800-update
endpoints; then the 400-update scaled model and the 800-update model are both evaluated on it.
SIGNAL: (800 - 400) M2+M3 top-1 mean > 0, CI wholly > 0, both seeds positive.

## P3' justification gate (NEW; does not make P2 a pass)
Satisfied only if the FINAL selected model on its untouched final holdout (B if P2.5-O was not
run, else C) has seed-averaged M2 >= 0.75 and M3 >= 0.55 with neither seed more than 0.03
below a floor, M1 >= 0.95 (facts-enabled), healthy finite training, and no catastrophic KQ/KR
descriptive regression. If satisfied: STOP, commit, report; do NOT run conversion, P3, M4/M5,
self-play or a curriculum. If it fails: STOP; add no layers, facts, successor boards,
recurrence, world model, parameters, losses or reward shaping within this lineage.

## Stop conditions
Historical probe_v1 identity changes; LF facts order disagrees with legal actions; LF facts
gradient zero/non-finite; any checkpoint cross-loads; a holdout canonical overlap; HOLDOUT_B
read before P2.5-D final evaluation or HOLDOUT_C before P2.5-O final evaluation; data-scaled
dataset overlaps A/B/C; non-finite model output; CUDA guard failure; a run with the wrong
seed/LR/sampler/digest; science configuration differing between seeds; a failure that cannot
be cleanly rerun under the same frozen contract. Infrastructure failures may be repaired only
with an identical scientific configuration, and are documented.
