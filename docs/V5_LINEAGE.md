# Recur64 V5 lineage

Status: **PRE-REGISTERED 2026-10-03. No V5 model measurement or training had
occurred when this record was written.**

V5 is the `counterfactual_relational_loop_v1` research line on branch
`experiment/hp-v5-counterfactual-loop`. Its accepted source is exactly
`17a782f8ebe68d1519ba3dc808c2473f0fd3a9f5`, the final V4 P0/P1 commit on
`experiment/workstation-v4-evidence-belief`.

The HP donor commit `79ffd1429e55e00d2767ffa482f89ced63668720` is read-only
environment/history evidence. It is not an ancestor to merge: the source and donor
have diverged since `03e62f7312650722138e0b59db924d0197ec2baf`.

Historical conclusions remain closed:

- V3 demonstrated teacher-query-pattern leakage; it did not establish useful
  returned-content use by ACTIVE.
- V3.5 made returned content measurably relevant but ACTIVE did not beat B0 or
  FIXED.
- V4 protected B0 and qualified its mechanism, but its trained evidence path did
  not materially change decisions. Its trust/gradient explanation was not tested.
- V4 utility learning was not run.

V5 starts from random weights. No V3/V3.5/V4/HP checkpoint initializes it. V5 is
not a repair or continuation of those experiments.

The read-only HP historical dataset inventory is recorded in
`docs/evidence/v5/historical-dataset-inventory.json` for future disjointness
checks. Those x1/x2 files are not V5 inputs and were not evaluated. No exact P25
TRAIN, V4_TUNE_V1 or HOLDOUT_C artifact was found or opened here.

## Scope boundary

Authorized: implementation, CPU/CUDA qualification, one bounded engineering
drill, Stage A, Stage B, and one reader pilot with seed 5301.

Not authorized: seeds 5302/5303, learned query control, self-play, external-engine
supervision, sealed evaluation, multiple architecture variants, hyperparameter
sweeps, or automatic follow-up science.
