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

## V5-E2 - release CPU/CUDA qualification and resolved physical layout

- **Date:** 2026-10-03
- **Status:** TESTED / MEASURED on release CPU and the intended NVIDIA CUDA
  device. Source `028025da1c7486eb0aa9140a88509c2822275e8a`; config digest
  `0f1c31d5fb3873ecca356a83c413674442633bdd9e53e9f1744523058b4fb00c`.
- **Build attempt E2a:** `--no-default-features --features cuda` failed at
  compile time because the pinned model crate's CPU type alias is unconditional.
  No graph executed and no CPU substitution occurred.
- **Correction E2b:** the compatible pinned build is `--features cuda`. It
  compiled CUDA alongside the default feature; the V5 CLI explicitly selected
  `cuda` and refuses if CUDA support is absent.
- **Identity correction:** preliminary passing release reports omitted the
  source SHA. They were not committed. The report schema was corrected, the
  correction was pushed, and both qualifications were rebuilt and rerun from
  the exact source SHA above.
- **CUDA:** physical microbatch 2 passed Q2/Q4/Q8 x R1/R2/R4 paired factual/null
  forward, full backward and AdamW; 50 additional resident Q8/R4 updates; and
  forward-only R8. Null centered-logit error was 0 (required <=1e-6), payload
  input-gradient L2 was 0.00038374067, every required reader group received a
  finite nonzero gradient, baseline identity and checkpoint restoration were
  exact. Worst warm shape update was 0.4693646 s; first/last repeated updates
  were 0.1897180/0.2006881 s.
- **Memory:** NVIDIA reported 4,096 MiB total. Device-wide used memory rose from
  144 MiB to 338 MiB, a 194 MiB peak delta, while unrelated workloads remained
  alive. WDDM did not provide process-resident accounting, so this is a
  conservative device-wide delta rather than a process-only peak.
- **CPU:** physical microbatch 2 passed the same Q/R training matrix, 50-update
  retention check and R8 forward diagnostic. Exact null error was 0; payload
  input-gradient L2 was 0.0003146815; worst warm shape update was 0.7318495 s;
  baseline and restore checks were exact.
- **Decision:** freeze FP32 physical microbatch 2 for the pilot. The authorized
  microbatch-1 fallback was not needed.
- **Evidence:** `docs/evidence/v5/cuda-qualification.json` and
  `docs/evidence/v5/cpu-qualification-release.json`.
- **Dataset limitation:** the 24-position engineering drill, Stage A, Stage B,
  DEV matrix and pilot gates remain NOT RUN because exact P25 TRAIN is absent.

## V5-E3 - complete bounded pilot harness and final-source requalification

- **Date:** 2026-10-03
- **Status:** IMPLEMENTED / TESTED on fixture and engineering paths; dataset work
  NOT RUN.
- **Scientific source:** `df6e2aa650c12726ad7094dae04a7c73339139a8`.
  Source identity is the last commit touching crates, Cargo manifests/lock or
  configs, so later documentation-only evidence commits do not recursively
  invalidate a report. Uncommitted scientific paths are still refused.
- **Implemented commands:** bounded Q8 drill with guarded Q16 fallback; per-cell
  update-0/update-800 reader evaluation; six-cell merge; ablation/composition
  report; exact bootstrap pilot report; and gate-guarded forward-only R8.
- **Evaluation invariants:** Q8 is acquired once per position/schedule and Q2/Q4
  are prefixes; R shares that graph; training acquisition seeds no longer depend
  on R; payload shuffle records non-self donor paths matched within family/depth
  and observed node depth; structure-only graph identity is invariant to payload
  intervention; composition nulling is applied after state encoding; normal
  traced replay must match its source evaluation.
- **Telemetry:** every record includes all requested policy/correction metrics,
  query and reader accounting, factual/null per-loop state/update RMS and
  attention health. Gate code uses 20,000 SplitMix64 resamples and the frozen
  ranks/seeds without rounded comparisons.
- **Checkpoint correction:** a new Stage B run now saves immutable update 0
  before its first optimizer step. Evaluation loads only exact update 0 or 800
  generations and revalidates model content, recipe and baseline fingerprint.
- **Tests:** full default workspace suite passed: 562 passed, 0 failed, 1 ignored
  (the historical deep-perft ignored test). V5-specific total: 19 library/
  integration tests plus 6 CLI-boundary tests, all passed. V5 clippy passed with
  warnings denied; CLI clippy passed with only the two documented pre-existing
  V4 lint names allowed.
- **Final-source CUDA:** PASS, FP32 microbatch 2. Exact null error 0; payload
  gradient L2 0.00038374067; baseline/restore exact; worst warm shape update
  0.4107268 s; repeated first/last 0.1865621/0.2088036 s; R8 forward 0.0801630 s;
  device-wide peak 338 MiB from 144 MiB (194 MiB delta).
- **Final-source CPU:** PASS, FP32 microbatch 2. Exact null error 0; payload
  gradient L2 0.0003146815; baseline/restore exact; worst warm shape update
  0.7699340 s; repeated first/last 0.7603199/0.7372230 s; R8 forward 0.4794547 s.
- **Evidence:** `docs/evidence/v5/cuda-qualification.json`,
  `cpu-qualification-release.json`, and `implementation-summary.json`.
- **Hard stop:** the exact P25 artifact is still absent. Drill, Stage A, Stage B,
  update-0/update-800 DEV evaluation, ablations, pilot classification and R8 are
  all NOT RUN. No substitute data or training was used.

## V5-E4 - qualification and prerequisite audit

- **Date:** 2026-10-03
- **Status:** IMPLEMENTED / CPU CONTRACT TESTS PASS; final-source CPU/CUDA
  qualification must be regenerated after this source checkpoint.
- **Audit correction:** the earlier qualification's optimizer restoration check
  loaded moments but only compared output logits. The new check compares every
  FP32 parameter and every optimizer moment/counter, then continues both copies
  through one matched AdamW update and requires exact parameter/moment equality.
  Earlier reports remain under `qualification-df6e2aa-cpu.json` and
  `qualification-df6e2aa-cuda.json`; E3 is historical evidence.
- **Checkpoint integrity:** all V5 checkpoint loaders now verify the optimizer
  content hash as well as the model hash. The CPU resume test uses the production
  LR schedule and deterministic cell sampler on explicitly test-only real chess
  fixtures; it compares the resumed LR, episode graph, logits and every parameter,
  and refuses an otherwise valid replacement optimizer file.
- **Scientific identity:** staged and untracked scientific files are refused.
  The binary records its scientific source at build time and refuses execution
  after a source commit until rebuilt. A Git-interface test covers clean,
  documentation-only, unstaged, staged, untracked and stale-build cases.
- **Loop contracts:** direct immutable-anchor gradients reach both production
  blocks at every R1..R4 loop with mutable state/memory held fixed for each test
  read. R1 exactly matches the first R4 state; no-feedback agrees at the first
  loop and changes the next evidence update. The library's process-global Flex
  RNG is serialized between model-building unit tests.
- **Pilot prerequisites:** `evaluate-baseline` separately records final Stage A
  Q0 on all DEV positions. Training requires the fixed drill report; Stage B
  additionally verifies the baseline report against its actual Stage A checkpoint,
  authoritative DEV labels/actions and device/layout. A single matched Q16
  diagnostic can satisfy the engineering prerequisite after informative Q8
  failure, without changing the recorded Q8 result.
- **Validation:** release workspace suite: 568 passed, 0 failed, 1 historical
  ignored test. Focused V5 release tests: 23 passed. V5 clippy with warnings
  denied and CLI clippy with only the two existing V4 allowances passed.
- **Resolved build findings:** moved the recall test module to the file end to
  satisfy clippy; corrected baseline correct-set indices to the authoritative
  `u32` type. No scientific geometry, precision, LR, loss, data or acquisition
  recipe was changed.
- **Data recheck:** exact P25 TRAIN remains absent. Drill, Stage A, final Stage A
  DEV baseline, Stage B, reader evaluations and pilot diagnostics remain NOT RUN.

## V5-E5 - complete uniform frontier invariant failure

- **Date:** 2026-10-03
- **Status:** ENGINEERING STOP / CURRENT IMPLEMENTATION UNQUALIFIED.
- **Affected source:** `97921bdd3cab701dff6478c7f0fdae525f0d6d00` and earlier V5
  sources. The global depth-five cap incorrectly truncates uniform acquisition.
- **Measured regression:** five exact transitions from standard start position,
  path `[400,400,5,5,320]`; depth-five node has 19 unqueried legal edges and three
  remaining Q units; current uniform selector includes zero. Release regression
  fails, exit 1. The test remains failing and visible.
- **Qualification correction:** E2/E3 report real CPU/CUDA network fixture
  measurements, but do not establish the complete-frontier invariant. Their broad
  engineering PASS claim is superseded by this stop. The 568-pass release suite
  at E4 preceded this new failing boundary regression.
- **Latest CUDA build:** 97921bd release CUDA build succeeded; new measured CPU/
  CUDA qualification did NOT RUN because the audit found this violation first.
  Qualification now refuses this known contract violation before model execution.
- **Root cause and proposed correction:** `docs/V5_ROOT_CAUSE.md`. Removing the
  selector filter alone is unsafe because the frozen depth representation has
  only five one-hot fields. No correction to the frozen representation has been
  implemented pending the required invariant-failure review.
- **Other evidence retained:** exact resume/optimizer-content checks, loop prefix,
  feedback, direct recall and finite differences passed on test-only real chess
  fixtures; compact evidence is `gradient-recall-audit.json`.
- **NOT RUN:** drill, Stage A/final DEV B0, Stage B, reader matrix, ablations,
  composition, pilot bootstrap/gates and conditional R8. P25 remains missing.
- **Pushed stop checkpoint:** `0c52b6e32d44e90107bcf34f3688377a0bc4ce31`
  preserves the failing regression, visible qualification refusal and root-cause
  report on `experiment/hp-v5-counterfactual-loop`. No proposed correction was
  implemented.

## V5-E6 - release qualification refusal-path verification

- **Date:** 2026-10-03
- **Scientific source:** `0c52b6e32d44e90107bcf34f3688377a0bc4ce31`.
- **Build:** `cargo build --release -p recur64-cli` passed in a reported 1m 54s.
- **Executed boundary:** the rebuilt release CLI requested CPU qualification at
  microbatch 2 and exited 1 with the explicit uniform-frontier contract error.
  Its new output path existed neither before nor after the command. The source
  check passed; the invariant preflight failed before model construction.
- **Scope:** refusal-path validation, not model qualification. No forward,
  backward, optimizer update, drill, training or DEV evaluation was executed.
- **Artifact recheck:** P25 TRAIN remains absent at the documented relative path
  in both local worktrees. Remote fetch confirmed branch parity before the check.
- **Evidence:** `docs/evidence/v5/engineering-stop-preflight.json`.
- **Next action unchanged:** required invariant-failure review of the proposed
  depth-contract correction, then corrected qualification and exact P25 custody.

## V5-E7 - delegated depth-contract correction before pilot

- **Authority/date:** owner delegated next-step choice, 2026-10-03.
- **Implementation:** complete uniform frontier restored; ranked depth five and
  five edges per branch preserved; depth representation expanded to 16 disjoint
  fields. Graph manifest/subcontract v2 now binds configuration and verifies Q,
  path, depth, ownership, parent and ranked limits. Old graph/config identities
  refuse. D/heads/FFN/blocks/precision/loss/seeds/sampling/updates unchanged.
- **Actual model:** 7,162,896 parameters, evidence initializer 139,776; measured
  increase 2,816. Configuration digest
  `d74109e229e49dc9962c348202db3527a5ce4c63da20efcd04a2bbe2577ff937`.
- **Retained regression:** five queries, three Q units remaining, 19 legal edges,
  all 19 included (previously zero). Depths 6..16, invalid depth/Q/path/parent,
  old-manifest refusal, structural offsets/padding and deep paired-null tested.
- **Focused release outcome:** 28 passed, zero failed, one explicitly runnable
  non-qualifying numerical diagnostic ignored by default. Full workspace and
  new measured CPU/CUDA qualification are separate subsequent gates.
- **Data:** local matching-filename recheck still finds no P25 TRAIN. Accepted
  source PR-triggered workflow run lookup returned empty (limited scope, not a
  global GitHub artifact absence claim). CLI `gh` was unauthenticated; no token
  was retrieved or login performed. No regeneration/replacement attempted.
- **Artifact manifest:** `docs/evidence/v5/required-artifacts.json`.

## V5-E8 - finite-difference failure and same-weight numerical reference

- **Initial failure retained:** after E7 geometry, original FP32 epsilon 0.05
  test failed at direction 0xA502, relative error 0.1643647402524948 versus 0.12.
  Fixed all-direction ladder exposed 0xA504 error 0.3099137842655182 at that step.
- **Diagnosis:** no cherry-picked step/direction replaces the gate. Same actual
  model/weights/graph in test-only FP64 reference gives FP32/FP64 autodiff error
  <=0.00014744318395504692 and numerical errors below unchanged 0.12. Production
  FP32 parameters remain unchanged. See `V5_NUMERICAL_ROOT_CAUSE.md`.
- **Failed reference attempts:** explicit FP64 reference initially hit default
  FP32 mask/direction dtype mismatches; the backtrace localized them. These are
  retained failures, not model qualification results or a framework upgrade.
- **Correction:** numerical reference arm uses exact same weights in FP64.
  Original fixture, four directions, epsilon and 12% assertion are unchanged.
  Additional derivative parity <0.001 and exact old/new FP32 centering tests pass.
  The old FP32 ladder is retained as an explicit diagnostic, not a scientific gate.
- **Scope:** unit qualification only. No CUDA/CPU measured qualification, drill,
  training, DEV evaluation, recipe tuning or architecture redesign was launched
  to compensate for the failure.

## V5-E9 - release workspace boundary expectation updates

- **Initial full-suite attempt:** stopped at the V5 model-info boundary's old
  exact 7,160,080 expectation. Updated to the measured amended 7,162,896 count;
  no assertion was removed or widened. New graph identity/config digest checks
  were added.
- **Second attempt:** the new digest assertion used the wrong JSON key
  `config_digest`; actual model-info schema is `scientific_config_digest`.
  The test key was corrected. This was a test/schema mismatch, not a numerical
  or model invariant failure. These nonzero exits are retained here.
- **Current boundary:** the corrected full workspace run is still required.
  Do not infer a full-suite PASS from the earlier focused 28-test result.

## V5-E10 - depth-amended full suite and failed CPU qualification

- **Source:** `7737640572c158fbda9c4cbadfe1332eb9303abc`, config
  `d74109e229e49dc9962c348202db3527a5ce4c63da20efcd04a2bbe2577ff937`.
- **Full release workspace:** 573 passed, zero failed, two explicitly ignored
  diagnostics (historical expensive perft and the FP32 numerical ladder).
  Scoped rustfmt and V5 clippy with warnings denied passed. CLI clippy passed
  with only the recorded pre-existing V4 lint allowances.
- **CPU qualification:** microbatch 2; nonzero exit, aggregate FAIL solely
  because graph-free/autodiff baseline values were not exact. Null error 0,
  payload gradient L2 0.003922534, all four reader groups nonzero; every baseline
  parameter and baseline output remained exact after reader updates. Complete
  parameter/moment restoration and one resumed parameter/moment update exact.
- **Executed work:** Q2/Q4/Q8 × R1/R2/R4, both streams differentiated,
  50 resident Q8/R4 updates; worst warm 0.7456381 s. R8 was engineering-only
  forward, 0.4578827 s. These are test-only fixtures, NOT a drill or pilot.
- **Evidence:** `qualification-7737640-cpu.json`, numerical gradient evidence
  `cpu-gradient-depth-7737640.json`. FP64 reference retains pinned FP32 RMS
  statistics; no fully FP64 normalization is claimed.
- **Diagnosis/decision:** `V5_EXECUTION_PARITY.md` freezes V5-D8 before its
  qualification: use the pinned default softmax primitive equation on both
  backends and require unchanged exact gates and exact old/new autodiff parity.
- **NOT RUN:** CUDA at this source, data drill/training/DEV/pilot. TRAIN absent.

## V5-E11 - explicit softmax execution parity tests

- **Decision:** V5-D8 in `V5_EXECUTION_PARITY.md`, frozen before measured
  qualification. No assertions/tolerances, geometry, precision, training recipe
  or data scope were relaxed.
- **Initial compile error retained:** the new parity harness attempted to clone
  `V5Inputs`, which has no Clone implementation. It now reconstructs the same
  versioned raw fixture input for each arm; no production input API changed.
- **Focused release:** three parity tests passed. Built-in graph-free softmax
  differs from Autodiff by 7.450580596923828e-9 on fixed isolated inputs; the
  explicit formula is exact. Full graph-free/autodiff root context, hypotheses
  and z0 are now bit-exact. Old/new Autodiff reader logits, centered deltas and
  returned-payload gradients are bit-exact for all nine Q/R conditions.
- **Parameter integrity:** baseline parameter content digest unchanged by this
  test. This is same-weight execution parity, NOT trained improvement.
- **Remaining gate:** full release workspace, fmt/clippy, source checkpoint,
  CUDA build and fresh CPU/CUDA qualification. No data-dependent work ran.
- **Lint failure retained:** V5 all-target clippy rejected the diagnostic module
  placed before production items (`items_after_test_module`). The module was
  moved to the file end; no lint suppression or production equation change.
- **Full release outcome:** 577 passed, zero failed, two explicitly ignored,
  across 68 result blocks. After that compiled run the diagnostic module was
  relocated (lint-only) and the CLI digest assertion pinned to the actual new
  digest; focused V5/CLI rerun covers those final test-only edits. V5 all-target
  clippy passes with warnings denied; CLI clippy passes with only the previously
  recorded V4 allowances. Edited Rust files pass rustfmt --check.
