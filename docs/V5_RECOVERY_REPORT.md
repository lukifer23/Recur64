# V5 recovery report - mandatory data-capacity STOP

Start HEAD: `dca2d900e05729d1f9fda484912218ed9a94561d`, clean V5 worktree,
fetched remote parity 0/0. Final scientific code:
`df261ab328606ce235b9b145070b608046a257c5`. Evidence publication follows as a
documentation-only commit on experiment/hp-v5-counterfactual-loop. R15 untouched.

The requested data contract cannot be completed: exhaustive KRvK M1 capacity is
**189 canonical classes, not 2000**, even before ANY exclusions. Section 23's
mandatory STOP applies. No count, family, depth, exactness or canonical identity
was relaxed. This is an infeasible data prerequisite, not a learning result.

## Execution diagnosis

1. Architecture/model forward/backward/optimizer math unchanged. 7,162,896
   parameters; FP32 physical 2; config
   `849133a5cdf169f187778bace2f858aa4747d2e8defef3bb5ac1bffc839774ee`.
2. Measured diagnostic source ad80002. All original model/optimizer fingerprints
   unchanged after clone forward and four ordinary/profiled/reversed updates.
   No clone/harness defect proven. Fresh independent checkpoint replicas verified
   against one canonical full model/populated AdamW snapshot for every replay.
3. NORMAL/NORMAL exact 3/3; PROFILE/PROFILE exact 3/3; NORMAL/PROFILE and reverse
   exact 0/3 each. CASE C. This is a fresh diagnostic snapshot after one fixture
   update, not the deleted historical resident qualification checkpoint.
4. First differing output: logits flat index 34, -0.08269035816192627 versus
   -0.08347725868225098. 34/68 elements differ, max absolute
   0.000786900520324707, RMS 0.00027925268468156, max ULP 105616.
   Centered delta max 0.0007868991233408451, max ULP 13174912.
   Correct-set loss 3.5191752910614014 versus 3.5195679664611816;
   max absolute 0.00039267539978027344, 1647 ULP.
5. First named parameter gradient: correction_hidden.bias, max absolute
   0.000015350407920777798, RMS 0.000003216976790882095,
   max ULP 1793660161 (near-zero sign crossings inflate signed ULP distance).
   First post-AdamW parameter: same tensor, max absolute
   0.000070914626121521, RMS 0.000007956640117224964, max ULP 339856.
   First moment: correction_hidden.bias.V1.Rank1.momentum.moment_1,
   max absolute 0.000001535042429168243, RMS 0.00000032169778276294743,
   max ULP 1744092657. Every optimizer counter exact.
6. Forty fresh boundary tests: only frozen_root_encoder_and_candidate_path and
   frozen_base_lift fences reproduce divergence. Earlier upload/view and later
   reader/loss/backward/optimizer fences do not. This localizes scheduling
   sensitivity to the frozen baseline completion/lift boundary; a specific
   internal framework/kernel defect remains UNPROVEN. No permanent fence added.
7. Historical D9 FAIL preserved, with unchanged historical report bytes.
   Current exact execution gate remains uncleared. No new contract adopted.
   Proposed owner-review v5_execution_profile_qualification_v2 is documented in
   V5_PROFILING_ROOT_CAUSE.md; numerical-reference/root-cause evidence is needed
   before freezing new acceptance criteria. No tolerance relaxation proposed as
   a substitute for explaining the observed forward shift.
8. CUBLAS_WORKSPACE_CONFIG and CUDA_LAUNCH_BLOCKING unset; optional environment
   experiments NOT RUN. Burn/CubeCL, toolkit, driver, TF32 state and precision
   unchanged. Diagnostic exit 0 means report completion, not qualification PASS.

## Data generation

Frozen family v5_hp_exact_endgames_v1, selection v5_hp_dataset_select_v1,
audit v5_hp_independent_audit_v1. V5_DATA_PLAN.md was committed/pushed at
76ef38e before capacity measurement; final code/tests pushed at df261ab.

| Requested identity | Seed | Requested cells | Requested total | Accepted |
|---|---|---|---:|---:|
| V5_HP_TRAIN_V1 | 0x7A50_1001 | five families x M1/M2/M3 x 2000 | 30000 | 0 |
| V5_HP_DEV_V1 | 0x7A50_1002 | KQRvK/KRRvK x M1/M2/M3 x 750 | 4500 | 0 |
| V5_HP_CONFIRM_V1 | 0x7A50_1003 | same as DEV | 4500 | 0 |

Capacity command at current source exited 1, explicitly STOP. It enumerated
all 249,984 distinct-square KRvK placements: 175,168 legal/live placements,
21,959 canonical classes, exactly 189 M1 classes. M1 has no ambiguity/fraction
filter, and zero historical/split exclusions were applied. Supply cannot reach
2000 even if all unavailable workstation exclusions are ignored.

Re-enumeration with one thread versus two reproduced every candidate, FEN,
canonical identity and depth exactly. All 189 capacity labels independently
audited through GameState/apply/termination: checked 189, failures 0.
This is capacity evidence, **not an accepted TRAIN split**.

Capacity-only measured digests:

- sorted capacity-record content:
  `5af24246587e12e82f2e8472f42c01138c7f48287cdc7fc1e60bdbe5c51fd71b`
- canonical set:
  `75a4eefe7b6bb333c24d921845764aa0bd5d47634519f05005301912d8d1650f`
- exact-FEN set:
  `19b8c66718de59dfe8acc25266086e6f8354546a2462f6917736511737dd510a`

TRAIN/DEV/CONFIRM measured content/FEN/canonical digests: NOT AVAILABLE.
Their full audits, pairwise intersections and full-split regeneration: NOT RUN.
No raw dataset files or dataset manifests/seals were fabricated. Required-artifact
manifest explicitly records generated=false, custody=false and null digests.
The old P25 required-artifact document is preserved as a historical archive.

Local historical filename inventory covered both worktrees' runs directories;
older x1/x2 target/tune/confirmation filenames exist, but no historical identities
were consumed/excluded because selection never proceeded. Cross-disjointness
from unavailable workstation-only raw datasets was NOT verified. No historical
V4_TUNE_V1/HOLDOUT_C label file was opened or evaluated in this pass.

CONFIRM artifact: NOT GENERATED, NOT SEALED, UNEVALUATED. All ordinary access is
blocked. Do not claim a physical seal or custody PASS for a nonexistent file.

## Scientific integration

New v5_stage_recipe_v2 and v5_hp_data_v1 are preregistered, **not adopted/bound**:
DATA-B has no complete accepted artifacts, so scientific binding would fabricate
identity. Retired recipe-v1/P25 implementation is inactive historical code;
V5Data::load and verify_custody refuse every scientific split pending measured
native binding. Final tests demonstrate TRAIN/TUNE/CONFIRM-shaped files refuse.
Training/DEV acceptance of the requested native split roles cannot yet be proven;
there is no accepted native dataset. Ordinary CONFIRM access remains denied.

Planned DEV expectation is 4500 total, six cells of 750, primary KQRvK M3 n=750.
The existing 4403/507 implementation is not relabelled as new-data behavior and
is unreachable behind the production data lock. Full role-specific loaders,
manifest/seal/custody, recipe-v2 data binding, independent DEV integration,
per-cell whole-split tests and interrupted-shard/merge tests remain NOT COMPLETED
because the prerequisite exact-count contract has no feasible TRAIN split.

Standalone graph provenance PASS at final scientific source df261ab. Persistence
is v5_graph_artifact_v1 envelope; scientific AcquiredGraph semantics/config
unchanged. Actual Q8 generation/audit accepts the current envelope and refuses
stale source, tampered graph, copied metadata, naked graph and wrong config.
Compact source-bound receipt: docs/evidence/v5/graph-provenance-df261ab.json.
Stage/study graphs are reconstructed in memory; source-bearing qualification/
diagnostic bundles retain provenance. No separate sidecar can be copied.

## Qualification, tests and drill

- Full release workspace at 76ef38e: native exit 0, 586 passed, zero failed,
  two preserved ignores, 68 result blocks. Final source df261ab changes only
  audit-test strengthening and explicit diagnostic snapshot metadata.
- Final-source V5 release tests: native exit 0, 40 passed, zero failed,
  one preserved diagnostic ignore. All old V5 assertions/tests retained.
- Final-source V5 all-target Clippy -D warnings PASS. CLI all-target Clippy PASS
  with the two previously documented unrelated V4 lint allowances
  collapsible_if/manual_is_multiple_of. Affected Rust files rustfmt check PASS.
- Serial pinned CUDA diagnostic build at ad80002 PASS; current-source CPU CLI
  build at df261ab PASS. No later-source CUDA qualification/build claim.
- Fresh CPU/CUDA qualification NOT RUN: CASE D was not reached and no harness
  repair cleared exact D9. Historical CPU PASS/CUDA FAIL remain historical.
- Resolved historical physical layout: FP32 microbatch 2, A accumulation 32,
  B accumulation 18; fallback 1 not used. No qualified native-data layout claim.
- FIT/Q16 drill NOT RUN. No complete dataset/audit/custody/seal gates, and D9 FAIL.
- NOT RUN: Stage A, Stage A DEV baseline, Stage B, all DEV reader evaluation,
  ablations, composition, bootstrap/pilot classification, scientific Q8/R8,
  Q16 drill, learned query controller, seeds 5302/5303, multi-seed replication,
  self-play, V4_TUNE_V1, HOLDOUT_C and V5_HP_CONFIRM evaluation.

V5 STAGE A NOT RUN.
V5 READER PILOT NOT RUN.
LEARNED QUERY CONTROLLER NOT TRAINED.
MULTI-SEED REPLICATION NOT RUN.
V5_HP_CONFIRM_V1 NOT GENERATED OR SEALED; UNEVALUATED AND ACCESS BLOCKED.
V4_TUNE_V1 AND HOLDOUT_C REMAIN UNEVALUATED.

## Owner continuation: complete capacity census

Continuation began from `5c74230c4810ce2fd3dbe00ae14a03f34a24facb`.
Architecture, scientific math, config and historical D9 are unchanged.
The full exact generator census finds all six light-family TRAIN cells below
2000 eligible canonical classes: KQvK 306/576/1076; KRvK 189/532/438.
Heavy-family capacities are KQQvK 95649/174163/4409,
KQRvK 111273/306595/211215, KRRvK 41612/122082/108086.
The six light cells total 3117, a shortfall of 8883. Both exhaustive census
processes exited 0; these are capacity results, not audited accepted splits.
Evidence: `evidence/v5/data/capacity-df261ab-all-families-m3.json`.
No requested count, dataset identity, scientific digest or recipe was amended.
The required explicit count decision remains pending; requesting an end-to-end
run cannot make the original unique/canonical quotas possible.
## Continuation CUDA diagnostics and final stop status

At unchanged scientific source df261ab, the pinned serial CUDA release build
passed. Separate workspace-config and launch-blocking diagnostics both complete
and reproduce CASE_C, exact self-repeatability, passing clone purity and the
same cross-mode numerical failure. Maximum logit difference remains
0.000786900520324707 (105616 ULP); first flat index 34. The localized frozen
baseline completion/lift boundary remains the first implicated fence boundary;
individual kernel cause is not proven. Full source-bound numerical archives
and compact receipt are under `evidence/v5/profile-environment-df261ab.json`.

Exact D9 remains FAIL. No new execution contract is adopted, no qualification
is relabelled, and no FIT drill is authorized by these diagnostics. Complete
TRAIN/DEV/CONFIRM generation remains blocked by the original exact quota
contract; all six light-family cells fail capacity, not just KRvK M1.
No native content/set digest, pairwise overlap result, audit PASS, custody PASS
or confirmation seal is invented. The confirmation artifact is NOT GENERATED,
NOT SEALED, UNEVALUATED and access-blocked. Recipe-v2/data-v1 remain pending.

V5 STAGE A NOT RUN.
V5 READER PILOT NOT RUN.
LEARNED QUERY CONTROLLER NOT TRAINED.
MULTI-SEED REPLICATION NOT RUN.
V5_HP_CONFIRM_V1 NOT GENERATED, NOT SEALED AND UNEVALUATED.
V4_TUNE_V1 AND HOLDOUT_C REMAIN UNEVALUATED.