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
- **Executed work:** Q2/Q4/Q8 Ãƒâ€” R1/R2/R4, both streams differentiated,
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
- **Final focused rerun:** after lint-only module relocation and the CLI literal
  digest pin, V5 32 passed/one ignored and CLI V5 boundary seven passed.
- **Push:** `2af69fe63e1a9725b1728c7c6b7dd701c01837b1` pushed before fresh
  qualification. Its CUDA release build passed; no measured qualification at
  that source was launched before the precision-preflight correction below.

## V5-E12 - explicit FP32-only build refusal

- **Preflight finding:** the V5 command path did not apply the historical TF32
  build refusal, so a future `--features tf32` build could be mislabelled FP32.
  The current CUDA build used only `--features cuda`, NOT tf32 or autotune.
- **Correction:** V5 configuration validation refuses TF32-enabled builds before
  model construction, identity use or qualification. A pure interface test
  checks both build-flag cases using the actual production guard; no TF32
  execution, alternate precision or new backend was enabled.
- **Scientific effect:** valid FP32 outputs/config digest/geometry unchanged;
  incompatible execution now fails visibly. New scientific source must be
  committed/pushed and rebuilt before qualification. No dataset-dependent work.
- **Focused validation:** all five config tests passed, V5 all-target clippy
  with warnings denied passed. No TF32-enabled binary was compiled or run.
- **Serial-build correction:** the final full CPU workspace suite and CUDA build
  were started concurrently. CUDA failed replacing `target/release/recur64.exe`
  with Windows access-denied during the concurrent suite, whose CLI tests use
  that shared output path. This is consistent with an executable-sharing race;
  the later process inventory found no remaining `recur64.exe`, so the exact
  holder at the failure instant was not observed.
  This is an operational build failure, not a CUDA graph qualification result;
  no measured qualification launched. Do not delete/kill unrelated processes.
  Wait for the suite, inspect live target-executable processes, then retry the
  SAME pinned CUDA build serially. No source/recipe change is a remedy.
- **Final-source full release suite:** 578 passed, zero failed, two explicitly
  ignored across 68 result blocks. Serial CUDA build succeeded (reported 2m18s).

## V5-E13 - final-source qualification, runtime setup failure retained

- **Scientific source:** `64c4dd4008a9b5bc8d715515279ec67e6e1173a1`.
  Config `849133a5cdf169f187778bace2f858aa4747d2e8defef3bb5ac1bffc839774ee`.
- **CPU:** measured qualification PASS at microbatch 2, null error 0, exact
  graph-free baseline reference, exact frozen-base outputs/all parameters after
  reader updates, full parameter/moment restoration and resumed update exact.
  Worst warm qualifier update 0.7248367 s; input gradient L2 0.0039228043.
  Report: `qualification-64c4dd4-cpu.json`.
- **Fresh gradient directions:** all four pass unchanged 0.12; maximum error
  0.05313823775366243. Same-weight autodiff parity maximum
  0.00014744318395504692. `gradient-64c4dd4.json` retains all values and the
  pinned FP32 RMS statistics limitation in the test-only FP64 tensor reference.
- **First CUDA runtime attempt:** PATH-only setup failed visibly at NVRTC header
  compilation (`cuda_runtime.h` missing), exit 1, no qualification report or
  checkpoint created and no CPU substitution. The pinned header DOES exist.
  CubeCL 0.10 `install::cuda_path` requires process-local CUDA_PATH or chooses
  its default Windows toolkit directory. The read-only HP donor setup sets
  both variables. Retry the SAME source/binary with both set to the existing
  user-space 12.9.1 root. Compact failure: `cuda-runtime-header-failure-64c4dd4.json`.
- **Corrected CUDA attempt:** SAME committed source/binary, both process-local
  CUDA_PATH/PATH set; intended RTX 2050 functional qualification PASS, no CPU
  substitution. All nine shapes, 50 resident updates, both differentiated arms,
  baseline/reference/all-parameter equality, complete model/moment restoration
  and continued update exact. Null error 0 against unchanged 1e-6 limit. Gradient
  L2 0.00059487484 and all four reader groups nonzero/finite.
- **Measured envelope:** device-wide sampled baseline/peak 138/1,068 MiB,
  930 MiB increase. All 50 resident samples 364 MiB, first/last update intervals
  0.2029461/0.1985860 s, worst warm interval 0.4127371 s. Engineering-only R8
  forward 0.0796470 s. No continuously sampled/process-only peak is claimed.
- **Timing limits:** qualifier wall intervals include gradient-coverage host
  reads, and do not have an explicit post-AdamW timer barrier. Separate encoders,
  streams, loops, backward, transfer and full-decision timings remain to be
  instrumented/qualified; these values are not a full-data training projection.
- **Resolved physical layout:** FP32, 2 positions; A accumulation 32, B
  accumulation 18. Fallback 1 NOT USED. Source-tagged CPU/CUDA reports retained;
  canonical qualifier paths contain the same parsed reports. Old source
  reports remain archived. No qualification or preflight checkpoint initializes
  the drill or pilot.
- **Still NOT RUN:** exact P25 custody, FIT drill/Q16, Stage A/B, all DEV science,
  pilot gates/diagnostics, scientific conditional R8, replication and sealed
  confirmation. This is functional engineering evidence, not learned chess use.

## V5-E14 - synchronized execution accounting protocol

- **Registered before measurement:** `V5_TIMING_ACCOUNTING.md`. SAME architecture,
  geometry, FP32, recipes, seeds, losses and practical pilot gates. Add explicit
  post-AdamW completion fences and observers to the REAL shared implementation.
- **Accounting:** separate exact root CandidateFacts CPU cost from Q; expose
  existing duplicated host preparation/upload, valid/padded rows, graph/depth/
  branch identity and every factual/null E/H loop application. Do not conceal
  duplication as free computation or claim isolated DMA/online latency.
- **Parity prerequisite:** exact same-device outputs, all payload/parameter
  gradients and ALL AdamW parameters/moments; CPU unequal-Q padding at R1/R2/R4.
  Old functional qualifications alone are no longer a current training gate.
- **Development failures retained:** initial compile incorrectly iterated Burn's
  `Shape` directly, then attempted its private `dims` field. Both failed visibly
  (E0277/E0616); use the pinned public `dims()` accessor. No measured run or
  scientific result arose from either failure. A third compile (E0614) corrected
  the accessor's owned `usize` items, which must not be dereferenced. No assertion
  was weakened. The fourth compile (E0284) required its explicit const rank
  `dims::<D>()`; confirmed from pinned CubeCL 0.10 `Shape` documentation.
- **Still blocked:** no `*proof*train*.json*` found in Desktop/Documents/Downloads
  or D: during this continuation. Exact P25 custody and FIT drill/training remain NOT
  RUN. No substitute, regeneration or sealed input was used.
- **First full-suite logging attempt:** all 68 result blocks report 581 passed,
  zero failed, two preserved ignores, but the PowerShell stderr/Tee wrapper
  returned exit 1 without an explicit native exit record. Preserve its log in
  ignored `runs/v5/qualification-accounting/`; rerun the SAME suite with native
  redirection and explicit Cargo exit capture before accepting the process gate.
- **Native full-suite rerun:** SAME source, explicit `CARGO_NATIVE_EXIT=0`;
  68 result blocks, 581 passed, zero failed, two preserved ignores. Thus no
  full-suite test failure is supported by either log; the original wrapper
  discrepancy is retained, not silently relabelled as a passing process.
- **Scoped checks:** edited-file rustfmt check and release V5 all-targets Clippy
  `-D warnings` PASS. Release CLI all-targets Clippy PASS with only the previously
  documented V4 `collapsible_if` / `manual_is_multiple_of` allowances. Measured
  current-source CPU/CUDA qualification remains pending.

## V5-E15 - synchronized CPU pass / CUDA profiling parity failure, STOP

- **Scientific source pushed before measurement:**
  `d97004981f52eb077da6bb72584efaca341b8c5b`; branch remote parity 0/0.
  Unchanged config `849133a5cdf169f187778bace2f858aa4747d2e8defef3bb5ac1bffc839774ee`.
  D9 timing digest `af2a950d922443a73d448fbdc3ab87bfc50a011dea4ea350516ffffa14cad20f`.
- **Serial CUDA build:** native MSVC/pinned user-space CUDA 12.9.1, FP32,
  `--features cuda`, native exit 0, reported 3m14s. No version/driver/precision
  changes or unrelated processes stopped. Source was clean and rebuilt.
- **CPU:** actual paired nine-shape/50-resident-update qualification PASS,
  exit 0. Null 0; all current parity/accounting/baseline/restoration/continuation
  flags true. Whole qualifier 50.8978161 s, synchronized worst warm 0.7291423 s,
  dedicated profiled update 0.7159244 s. Report `qualification-d970049-cpu.json`.
- **CUDA:** actual intended RTX 2050 graph ran, but overall FAIL/exit 1.
  `profile_outputs_and_all_gradients_exact=false` and
  `profile_adamw_parameters_and_moments_exact=false`. Phase accounting, null 0
  under unchanged 1e-6, graph-free reference, complete baseline integrity,
  complete model/moment restoration and ordinary continuation all pass. Finite
  nonzero gradients reach all four groups. These passes DO NOT clear D9.
- **Diagnostic envelope only:** whole qualifier 32.0531012 s, synchronized worst
  warm 0.3128205 s, dedicated profiled update 0.1837027 s. Device-wide sampled
  baseline/peak 138/1,100 MiB, delta 962; every resident sample 364 MiB. Report
  `qualification-d970049-cuda.json` is retained, not relabelled PASS.
- **Stop and diagnosis:** `V5_PROFILING_ROOT_CAUSE.md` separates aggregate failing
  comparisons from still-unmeasured per-field differences and possible causes.
  No automatic retry, threshold relaxation, LR screen, drill or training after
  failure. Next bounded work is per-field/clone-purity/normal-normal/profile-profile
  diagnosis, plus standalone graph source-provenance validation. No underlying
  CUDA/framework cause is claimed yet. This is not architectural falsification.
- **Custody/NOT RUN:** exact TRAIN still missing. FIT drill/Q16, Stage A/B, all
  DEV/pilot science, conditional scientific R8, seeds 5302/5303, query controller
  and sealed confirmation NOT RUN. Engineering-only R8 forward was executed;
  it is not a scientific extra-loop diagnostic. Temporary round-trip artifacts
  that passed all checkpoint checks were removed; failed reports are preserved.
- **Report publication safety:** the first canonical-alias guard refused a
  mistranscribed historical SHA before any write. Corrected the guard by comparing
  the full parsed old aliases with their preserved 64c4dd4 archives. Current
  canonical aliases match the d970049 reports exactly as parsed; CUDA remains
  FAIL and cannot satisfy the qualification loader. No old report was deleted.

## V5-E16 - recovery handoff and profile diagnostic registration

- Verified clean V5 worktree at dca2d900e05729d1f9fda484912218ed9a94561d;
  fetched origin and measured branch parity 0/0. R15 worktree untouched.
- Historical reports, including qualification-d970049-cuda.json, unchanged.
- Added engineering-only diagnose-profile-parity; no training authority.
  Canonical full-precision model/AdamW snapshot, clone fingerprints, three
  repetitions of all four orderings, independent checkpoint-loaded replicas,
  named numerical tensor/gradient/parameter/moment differences and counters.
- Exact D9 remains FAIL until unchanged qualification legitimately passes.
  Architecture, math, FP32, model config and learning rates unchanged.
- Owner excludes workstation P25 transfer; native-data preregistration follows
  before generation. No data-dependent operation or training run in this entry.

## V5-E17 - native data preregistration

PRE-REGISTERED before generation: V5_DATA_PLAN.md. No data-dependent training
exists. Exact KRvK M1 capacity census must precede complete split generation;
underfilled canonical cell stops without reducing the requested quota.
TRAIN/DEV/CONFIRM counts and seeds frozen exactly as owner requested.

## V5-E18 - measured CUDA profile diagnostic CASE C

Source ad80002; serial pinned release CUDA build native exit 0. Diagnostic
native exit 0, training_authorized=false. All clone purity checks PASS;
three normal/normal and three profile/profile exact; all cross/reverse differ.
Single-fence localization reproduces divergence only at frozen root encoder
completion and frozen-base lift. Detailed numerical results appended in
V5_PROFILING_ROOT_CAUSE.md. D9 remains FAIL; no harness defect proven.
Focused numeric report test and V5 all-target clippy -D warnings PASS.
No qualification, drill, training or evaluation launched.

## V5-E19 - persistence-only graph provenance closure

Audited persisted paths: CLI graph generate/audit is the standalone JSON
producer/consumer; stage/study acquired graphs are reconstructed in memory,
while qualification bundles carry source SHA. Added v5_graph_artifact_v1
envelope with source/architecture/config/full/structure digests and role.
Scientific AcquiredGraph and config digest unchanged. Loaders refuse naked,
stale, mismatched or tampered envelopes. No sidecar copying path exists.
Initial compile failed E0599 because the existing method is named
compute_structure_digest; corrected the caller without changing graph math.

DATA-A implementation includes existing exact-pool/label reuse, stable seeded
selection with strict counts/dedup/exclusions, independent material/canonical
validation and audit, and exhaustive capacity/regeneration report. Production
V5Data load/verify are locked pending complete measured native DATA-B: retired
P25 cannot authorize work, and no fabricated native digest is bound. Remaining
full split/shard/manifest/custody/recipe-v2 integration is conditional on the
mandatory quota capacity gate; it is not claimed implemented or measured.

## V5-E20 - DATA-A release verification and strengthened audit tests

Full release workspace at source 76ef38e: native Cargo exit 0, 586 passed,
zero failed, two preserved ignores, 68 result blocks. Graph CLI envelope
generation/audit and stale/naked refusal passed. V5 all-target Clippy -D
warnings and affected-file rustfmt checks passed before final test strengthening.
Final test strengthening explicitly asserts a terminal GameState and distinct
mirrored-FEN canonical duplication. Diagnostic metadata now stores explicit
root FENs and legal-action/target-index lists; replay math is unchanged.
Focused final-source V5 tests and CLI rebuild required before capacity command.
No native production dataset, qualification, drill or learning run occurred.

## V5-E21 - exact native TRAIN capacity failure; mandatory STOP

Scientific source df261ab; rebuilt current CPU CLI exit 0. Data plan/code/tests
pushed before measured capacity command. Complete KRvK M1 census: 249984
placements, 175168 legal/live, 21959 canonical classes, only 189 M1 classes.
Required 2000, exclusions zero, no M1 fraction/ambiguity filter. Two full
enumerations (2 threads/1 thread) reproduce exactly; all 189 labels pass
independent audit. Capacity command exits 1 and accepts zero datasets.
Section 23 STOP applies; no quota/family/depth/exactness relaxation.
Report docs/evidence/v5/data/capacity-df261ab-krvk-m1.json is capacity-only,
not a TRAIN manifest. Dataset files, DATA-B digests, pairwise overlaps, custody,
CONFIRM seal and recipe-v2 integration NOT COMPLETE/NOT RUN.
Final V5 tests 40 passed/one preserved ignore/native exit 0. Final V5 Clippy
-D warnings, CLI Clippy with the two documented V4 allowances, affected fmt PASS.
Current-source standalone Q8 graph envelope/audit and five negative boundaries
PASS; source-bound receipt committed. Historical D9 report unchanged.
No fresh qualifier, drill, Stage A/B, DEV or sealed evaluation after either
engineering or data failure. Full report: V5_RECOVERY_REPORT.md.

## V5-E22 - complete exact pool capacity census after owner continuation

Owner requested continuation/end-to-end execution. Before changing any quota,
ran existing `proof pool` exhaustive enumeration at scientific source df261ab,
max depth 3, two threads. Both native processes exited 0. No accepted native
split, training, DEV or CONFIRM evaluation occurred.

Eligible M1/M2/M3 classes: KQQvK 95649/174163/4409;
KQRvK 111273/306595/211215; KRRvK 41612/122082/108086;
KQvK 306/576/1076; KRvK 189/532/438. All six light-family cells
fail the requested 2000 quota. The earlier single-cell reduction proposal would
still be infeasible. Complete census receipt:
`docs/evidence/v5/data/capacity-df261ab-all-families-m3.json`.
These are exact solver capacity measurements, not independently audited split
manifests. The original count contract and STOP remain in force pending an
explicit pre-generation amendment; no quota relaxation is inferred from a
request to continue. CUDA fusion is disabled in the pinned feature graph.
## V5-E23 - permitted CUDA environment diagnostics; exact D9 still blocked

Serial pinned CUDA release build at scientific source df261ab passed (native
exit 0, 15m10s). Independently tested CUBLAS_WORKSPACE_CONFIG=:4096:8 and
CUDA_LAUNCH_BLOCKING=1, one process-local setting at a time. Both diagnostic
commands exited 0 (report completion, not qualification PASS). Each uses one
canonical full model/AdamW snapshot and independently verified fresh replicas.

Both report CASE_C: five clone-purity checks PASS; NORMAL/NORMAL 3/3 exact;
PROFILE/PROFILE 3/3 exact; NORMAL/PROFILE and reverse 0/3 exact each. Counters
remain exact. Both reproduce the same 34 differing logits, maximum absolute
0.000786900520324707, RMS 0.00027925268468156, maximum ULP 105616.
Only frozen_root_encoder_and_candidate_path and frozen_base_lift single fences
reproduce the forward divergence. Neither setting clears or materially changes
the observed repeatability classification. Burn fusion is disabled in the
actual pinned feature graph; no fusion cause is asserted. Individual kernel or
storage/stream cause remains unproven. No toolkit, driver, precision, TF32,
backend, architecture, model math or production fence was changed.

Compact receipt: docs/evidence/v5/profile-environment-df261ab.json. Full
per-tensor numerical reports are committed as losslessly compressed JSON under
docs/evidence/v5/profile-diagnostic-df261ab-{cublas4096,launch-blocking}.json.gz.
Both decompressed byte streams were verified against recorded raw SHA-256.
Starting model digests are identical across variants; optimizer record hashes
include process-local ParamIds and differ. No shared cross-environment checkpoint
identity is claimed; all within-variant replica checks are exact.

Historical qualification-d970049-cuda.json hash unchanged. D9 remains FAIL.
CASE_D was not reached; no fresh qualifier, FIT drill, Stage A/B, DEV or sealed
set evaluation was run. The proposed versioned execution contract remains
NOT ADOPTED. All six light-family TRAIN quota failures remain a separate mandatory
STOP, with no native split/seal/custody or scientific recipe-v2 adoption.
## V5-E24 - frozen baseline lifetime diagnostic registration

Owner explicitly requested work until CUDA D9 is fixed. Source baseline starts
at 6ef50bd. Add engineering-only diagnostic schema v2; preserve prior reports.
Ordinary replay is unchanged. New fresh-snapshot normal/profile probes hold
extra references to context, pooled, hypotheses, z0 separately and together,
three repetitions each. Read these references only after backward/AdamW and
all numerical readback, so no new pre-reader fence is introduced.
This tests activation lifetime/aliasing; it is not a model-math change, fix,
qualification or training authorization. No result is asserted before running.
Cargo check and V5 all-target Clippy -D warnings pass. Diagnostic rustfmt passes.
## V5-E25 - baseline output retention refuted as CUDA parity remedy

Scientific source c98ae68, pinned serial CUDA build PASS (10m26s), default
CUDA environment (both optional knobs unset). Diagnostic native exit 0,
schema v2, original CASE_C reproduced. All 15 extra lifetime comparisons fail
cross-mode forward/gradient/AdamW/moment equality. Retaining context, pooled,
hypotheses or z0 alone, or all four, does not repair parity. All retained
baseline outputs themselves are EXACT between modes after the complete replay:
context [2,64,256], pooled [2,256], hypotheses [2,34,256], z0 [2,34].
This refutes the tested baseline-output lifetime hypothesis; the fence-sensitive
reader computation remains unexplained. No production fix or new qualification.
Full numerical evidence: profile-diagnostic-c98ae68-lifetime.json.gz.
Next inspect unused graph input uploads in frozen baseline construction: it
builds full V5Inputs although base() consumes only RootInputs fields. Test input
retention and an engineering-only root-input path before adopting any change.
## V5-E26 - unused frozen graph-input probes, schema v3

Add fresh-snapshot diagnostic arms retaining the frozen baseline's unused full
V5Inputs through the complete step (mask16), and computing the baseline from
RootInputs only (mask32), with all baseline outputs retained too (mask47).
Production frozen-baseline execution remains unchanged. RootInputs profiling
builder visibility is crate-only; its equations and uploads are unchanged.
These probes test the duplicate graph-packet upload/lifetime hypothesis.
No remedy, exact gate PASS or backend defect is claimed before measurement.

Add a focused CUDA-only example executing the SAME diagnostic function to avoid
recompiling unrelated CLI training/evaluation monomorphizations on every probe.
It rejects CUDA-less execution, dirty scientific code and build/source mismatch,
requires a fresh output path, uses the same 64MiB worker stack, and has no
qualification/training command. Build binds RECUR64_DIAGNOSTIC_SOURCE_SHA.
V5 all-target CUDA Clippy -D warnings and diagnostic rustfmt PASS.
## V5-E27 - temporary frozen input retention clears diagnostic parity

Scientific source 6f8a3d8, focused CUDA build PASS (12m31s including a fresh
feature-union dependency build). Same CUDA backend type as CLI, schema v3,
default CUDA environment, native diagnostic exit 0. Original CASE_C reproduced.
Retaining the full temporary frozen V5Inputs through the complete replay
(mask16) makes forward, every gradient, post-AdamW parameters, moments and
counters EXACT in all three normal/profile pairs. Its unused states/flags are
also exact. RootInputs-only construction (mask32) and RootInputs-only plus
baseline-output retention (mask47) retain the original cross-mode discrepancy
in all three repetitions. No production fix adopted. This isolates a positive
input-lifetime/allocation intervention, not the specific faulty field or kernel.
Full numerical evidence: profile-diagnostic-6f8a3d8-frozen-input.json.gz.
Next compare the returned-state encoder before E/H computation using the same
checkpoint and input preparation, then narrow which temporary input matters.
## V5-E28 - narrow temporary input ownership and payload encoder localization

Engineering schema v4. Retain only root fields (64), graph fields (128), states
(256), flags (512), structural features (1024), owner indices (2048), relation
planes (4096), or bool masks (8192), through full backward/AdamW; no early
readback, copies, altered equations or extra fences. Existing full retention
and unmodified mode controls remain. Also record final factual/null H tensors
only after AdamW completion, preserving their original lifetime.

Three additional fresh-checkpoint pairs stop after the SAME returned-state
encoder, before E/H. Use the same payload-gradient leaf/reference ownership,
input construction and baseline preparation; first readback after encoding.
Diagnostic-only accessor calls existing StateEncoder.forward, unchanged.
Measure encoded payload, original states/flags, baseline context/hypotheses.
This is forward-only localization, not a gradient/optimizer qualification.
V5 all-target CUDA Clippy -D warnings and rustfmt PASS. No fix claimed yet.
## V5-E29 - payload masking is the first measured divergence

Source f61bdb0, serial focused CUDA build PASS, diagnostic native exit 0.
Original CASE_C persists. All three encoder-only comparisons differ at flat
index 8192: normal -1.1473956108093262 versus profiled zero. Exactly 2048
encoded payload values differ; max absolute 2.9452357292175293, RMS
0.3535476223471146. Input states/flags and baseline context/hypotheses are
EXACT. Entire temporary frozen-input retention still clears every group in
three repetitions; individual field/group retention does not. This localizes
the error before E/H, with unexpectedly zeroed payload blocks suggesting
masking. It does not yet prove which primitive is defective. No production
fix or qualification PASS. Full report: profile-diagnostic-f61bdb0-fields.json.gz.

## V5-E30 - minimal CUDA boolean-mask reproduction

Engineering-only mask primitive command (--mask in focused example), no model
training or authorization. Three repeats, four allocation sizes, fenced and
unfenced upload, three mathematically equivalent mask/broadcast orders.
Expected all-true node mask leaves an all-one FP32 tensor unchanged. Record
output numerical differences, node/inverse mask counts and backend metadata.
No precision, backend, environment contract or model equations changed.
CUDA all-target Clippy -D warnings and affected rustfmt PASS.

## V5-E31 - mask readback type correction and stride-preserving control

Source f989478 primitive command exited 1: CUDA boolean readback stores U8,
while to_vec<bool> requires Native. Preserve operational failure log; no
numerical result was produced. Use TensorData.iter<bool> host conversion.
Add single reshape [2,8] -> [2,8,1,1] control and a mask-only example avoiding
full model diagnostic monomorphization. Model execution is still unchanged.
Pinned burn-std split_strides source predicts trailing singleton dimensions
lose pitched batch stride on the second unsqueeze; measure metadata/output.

## V5-E32 - measured pitched-mask defect and equivalent implementation repair

Source 9f3c3d0 mask-only CUDA build PASS (2m24s), native run exit 0.
Two successive trailing unsqueezes turn the expanded mask batch stride into
8 rather than allocated pitch 16. Single reshape preserves [16,1,0,0] and
is exact in all 24 controls. Expanding before negation masks all 8192 second
fixture values incorrectly in all 24 trials. Negate-first variants happen to
pass with this primitive allocation history despite malformed stride; this
is not evidence their padding reads are valid. Archive mask-primitive-9f3c3d0.json.gz.
Pinned burn-std 0.21.0 split_strides skips new singleton dimensions without
advancing past old trailing singleton dimensions, losing the pitched stride.
This explains allocation/fence sensitivity and zeroed second-example payload.

Replace ONLY returned-payload mask's two trailing unsqueezes with one reshape
[b,qn] -> [b,qn,1,1], then the SAME expansion, negation and mask_fill semantics.
Architecture, FP32 storage, parameters, config, losses, optimizer, backend,
precision settings and fence schedules are unchanged. No Clone defect found.
This is an execution tensor-layout implementation repair, not model redesign
or a relaxed qualification contract. Full exact parity remains to be measured.
Add an independent mixed-row host-mask forward/backward regression across
Q2/4/8/16; extend GPU primitive controls to mixed and invalid second rows.
Affected rustfmt and all-target CUDA Clippy -D warnings PASS.

## V5-E33 - unchanged current-source D9 and CPU/CUDA qualification PASS

Source 003d296e094c28fc488cd56ef0944b60299983f9. Fresh actual CLI qualifiers:
CPU native exit 0/PASS; CUDA native exit 0/PASS. Both original D9 fields are
true: exact outputs/all gradients and post-AdamW parameters/moments. Historical
d970049 D9 remains FAIL. All seven prior qualification reports are byte-identical.
No tolerance, precision setting, environment knob, architecture, parameter,
config or execution-contract change. The repair is tensor-layout implementation;
Clone contamination remains refuted, not retroactively claimed as the cause.

Both qualifiers cover all nine Q2/4/8 x R1/2/4 shapes, 50 resident Q8/R4
updates, zero null error, frozen baseline parameters, graph-free baseline,
complete checkpoint/moment restoration and exact continued AdamW update.
FP32, microbatch2, 7,162,896 parameters, unchanged configuration and
v5_synchronized_execution_profile_v1 contract/digest. No learning data used.
Qualifiers ran while the remaining focused-example compilation was active;
timings are engineering observations under that operational condition, not
online latency or a pilot practical-gate result. CUDA wall 37.6270178s.

Fresh standalone graph CLI checks PASS: two current-source valid artifacts;
six required stale/config/tamper/copied/naked/historical-source refusals exit1.
Full release workspace and repeated independent snapshot diagnostic are still
in progress. No FIT drill, Stage A/B, DEV or confirmation evaluation run.
Native data quota infeasibility remains independent of this cleared CUDA gate.

## V5-E34 - independent repeatability and mixed-mask root-cause confirmation

Source 003d296; serial pinned CUDA CLI/examples build native exit0 (16m25s).
Fresh independent diagnostic native exit0, ALL_EXACT. All three repetitions of
NORMAL/NORMAL, PROFILE/PROFILE, NORMAL/PROFILE and PROFILE/NORMAL are exact:
5 forward tensors, 176 gradient entries including absent markers/payload,
175 post-AdamW parameters, 182 moment entries and all counters. Zero differing
elements, absolute/RMS/relative differences and ULP distances; no first differing
tensor. Five original clone-purity fingerprints remain unchanged.

Strengthened primitive native exit0. Correct single-reshape control EXACT in
72/72 trials across all-valid, invalid-second-row and mixed masks. Old two-step
negate/expand and implicit-broadcast forms fail 24/24 invalid-second-row and
24/24 mixed cases. Their all-valid coincidental pass is explained by padding
contents. This confirms a physical stride defect, not ordinary CUDA numerical
nondeterminism. Primitive overall FAIL intentionally retains bad negative
controls; corrected control PASS is separately explicit. No gate relaxation.

Receipt cuda-mask-recovery-003d296.json binds lossless numerical archives and
verified compressed/decompressed hashes. Current-source qualification PASS
under unchanged timing contract; historical CASE_C and D9 FAIL preserved.
No new execution contract is needed to clear this defect. Full workspace tests
are still compiling; no training/drill/data acceptance claim follows.

## V5-E35 - completed source-bound release validation

Full release workspace native exit0: 587 passed, zero failed, two preserved
ignores, 68 test-result blocks. New mixed-row returned-payload forward/backward
regression PASS. All original representation/null/recall/feedback/permutation,
complete frontier depth1-16, loss/gradient/baseline/resume/refusal tests retained.
Source remains 003d296e094c28fc488cd56ef0944b60299983f9; no code changes after
CPU/CUDA qualification and repeated independent numerical reports.
V5 all-target CUDA Clippy -D warnings native0, affected-file rustfmt native0.
Workspace-wide fmt native1 identifies 16 untouched files; all independently
verified byte-identical to starting HEAD. This limitation is recorded explicitly,
not hidden or called PASS. Validation receipt and complete compressed suite log
are committed. No test assertions deleted or weakened.

D9 repair objective achieved under the existing contract. Dataset quotas remain
infeasible under the frozen canonical uniqueness requirement; no accepted native
TRAIN/DEV/CONFIRM, custody/seal or recipe-v2 binding. No 24-FIT drill, Stage A/B,
DEV evaluation, learned controller, seeds5302/5303, self-play, confirmation,
V4_TUNE or HOLDOUT_C evaluation. No chess-learning result is claimed.

## V5-E36 - native heavy-family V2 preregistration

Owner authorizes v5_hp_data_v2, heavy-only 27000/4500/4500, new seeds and split identities. V1 stays infeasible historical preregistration. See V5_DATA_V2_PLAN.md. DATA-V2-A implementation and tests precede accepted generation; no model/architecture changes or learning authorized.

## V5-E37 - measured V2 generation and independent regeneration

Producer 738db983084a998664ce962f87fa3c4a6153f526 generated TRAIN27000, DEV4500 and CONFIRM4500 with exact cell quotas. 36000 independent original audits and 36000 fresh regeneration audits; zero failures. All three regenerated artifacts are byte-identical after full re-selection/re-labeling/re-audit from verified exhaustive pools. All three FEN/canonical intersections are zero. CONFIRM sealed=true/evaluated=false. Compact manifests, exact-byte bindings, audits, disjointness and seal are under docs/evidence/v5/data/v2/. Raw files remain ignored under runs/v5/data/v2/. No production loader binding or fresh consumer qualification claimed yet.

## V5-E38 - measured data-v2/recipe-v3 integration

After pushed DATA-V2-B, production constants bind exact measured V2 bytes/manifests. TRAIN-only/DEV-only scientific entry points, sealed CONFIRM custody, nine-cell sampler exposure and independent DEV4500/primary750 expectations are implemented. Recipe-v3 and evaluation/drill schema amendments are explicit. Model/math/configuration/mask repair are unchanged. Current-source qualification and final custody/drill remain pending; no Stage A/B or DEV science run.

## V5-E39 - current-source final engineering gate

Source d11659eca0774e0064bed0ef64ead2b725886d93: full release workspace596 PASS/zero failures/two preserved ignores; current CPU and RTX2050 CUDA qualification PASS unchanged exact D9 at FP32/microbatch2, nine Q/R shapes, 50 resident Q8/R4 updates and complete checkpoint/moment/resume. Twelve repeated independent normal/profile comparisons ALL_EXACT; clone purity PASS. Fresh persisted graph provenance six cases PASS. All three actual local raw-file custody/seal/disjointness and role refusal checks PASS. V5 CUDA Clippy -D warnings/serial pinned CUDA build/changed-file rustfmt PASS. Global formatting drift remains in16 verified unchanged files. Model/config/profiler/qualification/graph blobs match003d296; historical failed reports preserved. Conditional 24-position drill is now eligible; Stage A/B and DEV science remain unrun.

## V5-E40 - conditional Q8/R4 engineering drill completed; STOP

Every prerequisite passed before the run. Source d11659e, CUDA FP32/microbatch2,24 frozen stable-hash TRAIN positions (four per six KQR/KRR depth cells),48 schedule graphs,200 reader-only updates,peak LR1e-3/warmup20. Mean correct-set loss2.987107088 ->0.026624600,99.1086828% reduction: PASS. Baseline fingerprint exact, training finite, disposable parameters not reused. Q16 NOT RUN. Post-drill three-split raw custody and CONFIRM seal/evaluated=false PASS. STOP before Stage A; no Stage A/B, DEV science/pilot/controller/replication/self-play/LR screen/historical or native confirmation evaluation. V5_DATA_V2_REPORT.md is the complete handoff.

## V5-E41 - Stage A-only owner authorization and current V2 handoff

Starting remote HEAD 066270cc8818c528c1eeaca0217e85664449ac78; clean branch
parity 0/0. Scientific source remains d11659eca0774e0064bed0ef64ead2b725886d93.
Documentation-only current-V2 clarification preserves historical preregistration.
Fresh actual local TRAIN/DEV/CONFIRM custody, exact recipe receipt and saved
current-source CPU/CUDA exact D9/drill prerequisites verified before model
initialization. Stage A seed5301, fixed1200 updates, <=45-minute resumable chunks,
fresh random model, followed by exactly one final DEV4500 B0 evaluation is
explicitly authorized. No Stage B/update0/graphs/optimizer, reader pilot, Q/R DEV,
controller, replication, self-play or sealed-set evaluation is authorized.
No measured training result is asserted by this entry.

## V5-E42 - fixed Stage A complete; final B0 validation failure; STOP

MEASURED at unchanged scientific source d11659eca0774e0064bed0ef64ead2b725886d93.
Stage A seed5301 completed1200 updates, one bounded invocation1043.721400s,
zero resumes/native0, final model2d1c770a43a6455148b774e9ddb552b6ca33efd9fdd5d37593cefe7c8ae0bb00. All losses and complete final
model/moments finite, exact frozen recipe and balanced76800-example exposure:
Stage A EXECUTION VALID. First/last50 mean losses2.726584702618/0.570068903945.
The exactly-one final DEV4500 B0 command exited1 after19.8672492s at
baseline sorted DEV identity digest mismatch. Read-only code/hash inspection
proves sorted-ID digest8e52094f...bc1b5e is compared to ProofTargets82d4578a...69d2a0.
All DEV forward records completed in memory before report validation; no report
serialized, no metrics recovered, no retry. Scientific/evaluation code unchanged.
Post-run actual three-split custody, zero FEN/canonical overlaps and CONFIRM
sealed/evaluated=false PASS. Owner failure stop/source-preservation rules applied:
no code repair or further model execution. Stage B NOT AUTHORIZED. See
V5_STAGE_A_RESULTS.md and compact summary/failure/custody evidence.
