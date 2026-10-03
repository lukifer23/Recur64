# V5 experiment ledger (append-only)

Labels: PRE-REGISTERED / DETECTED / TESTED / MEASURED / INFERRED / NOT RUN.
Corrections are appended; old entries are never rewritten.

## V5-E0 - lineage, preregistration and HP/data inventory

- **Date:** 2026-10-03
- **Status:** PRE-REGISTERED / DETECTED. No V5 model code, qualification or
  training had run when the contracts were frozen.
- **Lineage:** new worktree/branch from accepted source
  `17a782f8ebe68d1519ba3dc808c2473f0fd3a9f5`; donor commit read-only; no merge.
- **Frozen identity:** `docs/V5_ARCHITECTURE.md` and
  `docs/V5_RESEARCH_PLAN.md`.
- **HP:** hardware/toolchain/CUDA items in `docs/V5_HP_ENVIRONMENT.md` are
  DETECTED only. V5 CUDA is NOT TESTED.
- **Data:** no `proof-train.json` was found under Desktop, Documents, Downloads
  or the checkout. Dataset-dependent work is BLOCKED until custody passes.
- **NOT RUN:** V5 code, model construction, tests, CUDA graph, drill, Stage A,
  Stage B, DEV evaluation, R8, query controller, replication, sealed sets.

## V5-E1 - Milestone B implementation and CPU preflight

- **Date:** 2026-10-03
- **Status:** TESTED / MEASURED on CPU; CUDA NOT TESTED in this entry.
- **Identity:** counterfactual_relational_loop_v1, config digest
  0f1c31d5fb3873ecca356a83c413674442633bdd9e53e9f1744523058b4fb00c.
- **Measured size:** 7,160,080 unique parameters; 28,640,320 FP32 parameter
  bytes. No parameter padding was added.
- **Attempt E1a:** the first debug qualification process failed visibly with
  Windows STATUS_STACK_OVERFLOW before evidence output. Root cause was the
  default worker stack while constructing/recording the full Burn graph.
- **Correction E1b:** V5 train/qualification commands received an explicit
  64 MiB worker-stack boundary. The complete debug CPU qualification then
  passed at physical microbatch 2.
- **Executed graph:** Q2/Q4/Q8 x R1/R2/R4 each ran forward, backward and AdamW
  on the paired factual/null graph. Q8/R4 then ran 50 additional resident-model
  updates; R8 ran forward only.
- **Key invariants:** exact CPU all-null correction 0; returned-payload input
  gradient L2 0.0003146815; nonzero finite gradients in state encoder,
  evidence block, hypothesis block and correction readout; graph-free baseline
  fingerprint exact after reader updates; full model/optimizer restore exact.
- **Timing status:** debug-build diagnostic only, not a release performance
  claim. Worst observed warm update was 1.2428614 s.
- **Evidence:** docs/evidence/v5/model-info.json and
  docs/evidence/v5/cpu-qualification-debug.json.
- **Dataset limitation:** exact P25 TRAIN remains missing, so the engineering
  drill and both pilot stages remain NOT RUN.
