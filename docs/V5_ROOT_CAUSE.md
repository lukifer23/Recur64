# V5 acquisition invariant failure

Status: **ENGINEERING STOP, 2026-10-03. No dataset-dependent run has started.**

The owner authorization says uniform_frontier_v1 chooses uniformly from the
complete current unqueried legal frontier. Only base_ranked_depth_v1 has the
depth-five and five-edges-per-branch limits.

The implementation at scientific source
`97921bdd3cab701dff6478c7f0fdae525f0d6d00` incorrectly applies one global
`MAX_DEPTH = 5` to both schedules. The uniform selector discards every legal edge
whose parent is at depth five. Its graph verifier also refuses depths above five.
The initial V5 depth contract and representation copied the ranked schedule's
limit into a global bound; this defect was present in V5-B.

## Reproduction

Run:

    cargo test -p recur64-v5 --release uniform_frontier_includes_legal_edges_below_a_depth_five_acquired_node -- --nocapture

The real chess fixture starts from the standard starting position and follows
five first-legal-action StateQuery transitions. It uses no solver or labels.
The action-index path is `[400,400,5,5,320]`. The depth-five node has **19 legal
unqueried edges**, and the query budget has **three remaining units**. The
production uniform selector includes **zero** of those 19 edges.

The release test fails with expected 19 versus actual 0 and process exit 1.
It is retained as a failing regression. No assertion was relaxed or removed.

## Coupled cause

- `graph.rs`: the same MAX_DEPTH controls uniform filtering, ranked DFS and
  manifest verification.
- `config.rs`: max_depth is frozen to five.
- `lib.rs`: STRUCTURAL_FEATURES reserves five one-hot depth fields.
- `model.rs`: the turn-role fields begin at offset five. Simply removing the
  acquisition filter would allow deeper depth fields to collide with turn/slot/
  action fields, so it is not a sufficient fix.
- Existing acquisition tests checked reproducibility and nested prefixes within
  the truncated implementation. The network qualification fixtures did not
  establish complete-frontier coverage at the depth boundary.

The new qualification preflight refuses this known contract violation before
constructing or executing the model. Earlier CPU/CUDA fixture measurements remain
historical network evidence, but do not satisfy the full engineering gate.

## One proposed correction, not implemented

Separate ranked DFS depth limit five from graph/representation capacity sixteen,
which covers the maximum depth attainable under the authorized Q16 diagnostic.
Remove the uniform frontier depth filter. Use sixteen one-hot depth fields,
followed by the existing turn-role, four-slot and action-geometry fields; shift
their offsets explicitly. Validate graph depth against actual charged Q and the
declared representation range. Keep the ranked schedule's depth/branch limits.

This preserves D=256, heads=8, FFN=768, encoder/core block counts, FP32, residual
scale, readouts, sampling, Q/R exposures, loss, seed and total updates. It adds
eleven structural input fields to the evidence initializer, for a **projected**
2,816 additional parameters (projected total 7,162,896). Those are accounting
projections, not measured model counts. The corrected input contract/configuration
must receive a new digest and a pre-pilot preregistration amendment.

After owner review: implement that correction; retain and pass the regression;
check depth-six through depth-sixteen graph/encoding bounds; rerun relevant release
tests, gradient/null/resume qualification on CPU and CUDA; then re-freeze the
physical layout. Do not start a drill or either training stage before all gates
pass and exact P25 custody is available.

## Current scientific status

The CUDA build at 97921bd succeeded. Its new measured CUDA/CPU qualification was
stopped when this violation was found. P25 TRAIN is also still missing. Drill,
Stage A, Stage B, all DEV evaluations/diagnostics and pilot classification remain
NOT RUN. This is an unqualified implementation, not architectural falsification.

LEARNED QUERY CONTROLLER NOT TRAINED.
MULTI-SEED REPLICATION AND SEALED CONFIRMATION NOT RUN.
V4_TUNE_V1 AND HOLDOUT_C REMAIN UNEVALUATED.
