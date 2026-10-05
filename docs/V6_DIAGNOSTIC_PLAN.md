# Frozen V5 closure diagnostics for V6 design

Status: diagnostic preregistration; no V6 implementation or fitting authorized.
Evidence snapshot: publication HEAD `3efce66f62c120df4bf599796c27f66d2ec194d6`,
Stage-B producer `3db24a926815159d592e93b60a8ae51852abad13`, existing evaluator
library `7fb7461e94fa4819404913d9a51e91d4d154c167`. Production code stays unchanged.

## Selection frozen before model execution

Use only V5_HP_TRAIN_V2. Nine cells, 24 IDs each, 216 positions total. Sort each
cell by SHA256(`recur64.v6_diagnose_train_panel_v1` + NUL + little-endian seed
`0x7A60_D001` + UTF-8 position ID), then ID; take first 24. No prediction/loss
selection. Exact IDs and indices: [panel](evidence/v6/train-panel.json). The
first two selected IDs in each cell define nine gradient pairs; no substitution
for a pair that happens to be baseline-right. Report actual wrong/right support.

## Fixed execution and bounds

Independent scratch executable outside production scientific code, linked to
the existing CUDA V5 library. FP32, physical microbatch 2, RTX2050, unchanged
process-local CUDA12.9.1 environment. Load immutable Stage-B updates0 and800
through the normal integrity loader. No Trainer, optimizer construction or steps.
Before/after parameter digests must match; all checkpoint/raw/failure hashes
preserved. Native exits and scratch-source/binary/library hashes recorded.

One coverage/forward invocation per checkpoint, each timeout20 minutes; one
gradient invocation per checkpoint, each timeout20 minutes. Total diagnostic
model budget80 minutes; terminate visibly on nonfinite values, mismatch or
timeout. No repeat for a preferred outcome. Compilation correction before any
measurement is permitted; failed compile logs remain evidence. A failed model
invocation stops model diagnostics and is reported, not automatically retried.

Acquire Q8 once per panel root and each existing schedule, with seed
`0x7A60_D002`, occurrence0 and frozen B0 logits. Reuse the identical graphs
for updates0/800, R1/R4 and controls. New TRAIN diagnostic episode identity,
not a replica of sampled training or frozen DEV evaluation episodes.

## A: information coverage and conservative partial proof

Reconstruct only states along the acquired paths using repository rules;
enumerate legal actions at those states for denominators. Never query unseen
child states or call MateSolver in this diagnostic. Report by cell/schedule and
baseline wrong/right: branches covered/root legal count, depths, terminal/mate/
draw nodes, defender nodes' acquired children/legal reply count, fully covered
defender nodes, and baseline-best/correct-set branch coverage. Correct labels
are used only for offline stratification, not supplied to reader inputs.

Partial proof values are WIN/NOT_WIN/UNKNOWN from root attacker perspective.
Observed terminal checkmate resolves WIN iff loser is defender; observed draws
resolve NOT_WIN. Nonterminal attacker: any WIN child => WIN; all legal children
acquired and NOT_WIN => NOT_WIN; otherwise UNKNOWN. Defender: any NOT_WIN child
=> NOT_WIN; all legal children acquired and WIN => WIN; otherwise UNKNOWN.
Unqueried root actions/replies are UNKNOWN. No missing-reply-as-win inference.
Report root candidates proved winning/refuted/unknown and decision relevance.
This diagnostic is deliberately weaker than learning from nonterminal boards;
zero terminal-proof coverage cannot prove that returned states are useless.

## B: trained-base learnability and C: integration

At both updates, Q8/R1 and Q8/R4, both schedules: real, null and deterministic
TRAIN-only shuffle. Same widening rule as v3: different root/same cell/same
turn/exact depth else nearest same-turn depth; stable structural candidate order,
seed `0x7A50_E002`. Any unresolved pool stops, never widens across cell/turn.
Targets are solely used by diagnostic metrics/loss, never donor selection.

Record per-position logits-derived loss, top1/mass, top-two action margin,
best-correct versus best-incorrect margin, candidate correction range/range-to-
action-margin ratio and real-minus-shuffled effects. Stratify by frozen B0
wrong/right, not by a selected outcome. Save raw diagnostic rows under ignored
runs; commit compact aggregate summaries and hashes only.

At real Q8/R4, trace factual/null separation before/after each loop; node/slot
evidence distinguishability and hypothesis differences; H-attention mass on
owned evidence, other evidence, candidate context and root context; E-attention
mass on evidence, hypotheses and root context. Compare owned-evidence attention
to its uniform mass given actual token counts. Normalized RMS/diffuse attention
alone is not a collapse test.

Gradient pairs: both schedules, Q8/R4, both updates, correct-set loss over two
examples. Track returned-board input gradients and every named model gradient
(absent markers preserved), aggregating state encoder/pooling/evidence/hypothesis/
readout/root groups. At update800 uniform schedule also isolate factual-only and
null-only gradient contributions by stopping the opposite stream at the readout;
same saved weights and forward values, no optimization. Verify recomputed paired
readout agrees before interpreting contributions. Nonzero gradients demonstrate
connectivity, not useful learning. No root gradient is expected with base_frozen.

## Serialized-only retrospective DEV analysis

No new DEV model invocation. Recompute four existing contrasts on KQRvK M3 n750,
IDs sorted and two schedules averaged within position. Use original paired
bootstrap20,000 resamples, seeds0x7A50_0101..0104 and ranks499/19499. Label as
RETROSPECTIVE SUPPLEMENTARY ANALYSIS, not the original all4500 classifier, and
do not change NO_SIGNAL or any original report. Record the preregistration/code
population discrepancy explicitly. CONFIRM/V4_TUNE/HOLDOUT_C remain unopened
to models. No Q16/R8 rescue, seeds, training or V6 variant fitting.


## Recorded utility failure and prospective instrumentation amendment v2

The first update0 scratch forward invocation exited101 with CubeCL missing-resource
and allocation errors before serializing scientific diagnostic rows. The CUDA
forward child was not interrupted; its coordinator was stopped before any gradient
launch. Failure log and binary are retained. No forward800 or backward probe ran.
The utility built forward-only traces on Autodiff rather than the graph-free inner
backend used by production evaluation. Retained autodiff activations are a likely
resource cause, not a proven production/backend defect. Additionally, inspection
found that named parameter hooks require visit_float plus enter/exit callbacks;
the initial visitor only overrode visit_float_with_path, which the pinned Param
implementation does not call. This was found before gradient measurement.

Model diagnostics stopped on that failure. This explicit pre-execution amendment
permits ONE new utility version, no production change: forward-only tracing on the
existing model.valid() inner backend; correct named visitor dispatch for gradient
readback. Same panel, checkpoints, information, equations, controls, metrics and
20-minute invocation bounds; no tuning or outcome-selected sample. Original failed
update0 attempt remains invalid and is never merged. This instrumentation recovery
is not a repeat to obtain a preferred scientific outcome. Any v2 CUDA invocation
failure stops all remaining CUDA diagnostics; no further automatic repair/retry.
