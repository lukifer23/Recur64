# V5 HP resume and artifact transfer

## Current V5 closure / proposed V6 (2026-10-05)

V5 is **CLOSED ? NO_SIGNAL**, with the frozen seed5301 experiment unchanged.
The owner-authorized closure investigation used existing DEV records and a frozen
216-position TRAIN panel; no retraining, new DEV inference or optimizer steps.
The V2 primary750 versus implemented pooled4500 discrepancy is now explicit.
Producer `3db24a9`, evaluator `7fb7461`, reviewed publication `3efce66` are separate
identities; later documentation commits do not replace them.

Start with [V5 closure](V5_CLOSURE.md), [measured diagnosis](V6_DIAGNOSTICS.md),
[one V6 proposal](V6_EXPERIMENT_PROPOSAL.md) and
[future P0 implementation prompt](V6_IMPLEMENTATION_PROMPT.md).
Recommendation: branch-local adversarial evidence backup with direct TRAIN
branch supervision, competent frozen B0, and same-information nonrecurrent
controls. This is a proposal: **V6 NOT IMPLEMENTED / NOT TRAINED**.
Gradient probes stopped at scratch preflight failure; backward results NOT RUN.
TRAIN/DEV/CONFIRM bytes remain unchanged; CONFIRM sealed and unevaluated.
Next step is owner design review, not a V5 resume or automatic V6 execution.

All lower status accounts and command blocks are preserved HISTORICAL material.

## CURRENT STATE: frozen Stage-B experiment COMPLETE — NO_SIGNAL

Owner review required. Recommend CLOSE V5 AND DESIGN V6, without implementing
it in this pass. No training/evaluation process remains active after publication.
Stage B remains800/800 source3db24a926815159d592e93b60a8ae51852abad13,
run `runs/v5/v2/seed-5301/stage-b`; modelc7f6b10a0b60982199cd352c157a377115c6f00a46c699a6a9b3bb859e8bd273.
Evaluator source7fb7461e94fa4819404913d9a51e91d4d154c167, v3 controls.
Complete raw matrices: `runs/v5/v2/seed-5301/eval-v3-000/all-dev.json` and
`runs/v5/v2/seed-5301/eval-v3-800/all-dev.json`. Both4500 unique DEV positions.
Current compact receipts: `docs/evidence/v5/stage-b-v3-summary.json`,
`stage-b-v3-pilot.json`, `stage-b-v3-ablation.json`, `stage-b-v3-post-custody.json`.
[Full result](V5_STAGE_B_RESULTS.md) includes primary matrices and diagnostics.
B0 exact; TRAIN/DEV/CONFIRM custody PASS; CONFIRM sealed=true,evaluated=false.
R8 notrun(NO_SIGNAL); no retraining/extraupdates/replication/controller/selfplay.
Historical v2 partials and failure remain preserved and are not final evidence.
The following older handoffs,StageA/P25 commands and blocked status are HISTORY,
not current instructions. Do not resume old training through the evaluator bridge.


> LATEST CONTINUATION REVIEW: no scientific process active; update800 complete.
> Existing update0 shuffle donor failure still blocks DEV completion. Recommend
> HARNESS REPAIR, subject to owner approval of a versioned evaluation-control
> amendment. No new training/evaluation/code changes; no pilot classifier output.
> Historical drill trajectory variance recorded in V5_STAGE_B_RESULTS.md.


> CURRENT STAGE B HANDOFF (2026-10-05): seed5301 training COMPLETE and VALID,
> all800 updates, final scientific source3db24a926815159d592e93b60a8ae51852abad13.
> Stage A and published B0 unchanged. Three update0 KQRvK DEV cells completed;
> KRRvK M1 refused its shuffle control (no other-root donor at node depth6).
> Further evaluation STOPPED. No update800 DEV, merges or pilot classification.
> No source repair/retry authorized under the frozen experiment. Next review:
> owner authorization for a prospective evaluation-control amendment preserving
> completed update0/800 checkpoints. Do not retrain or launch more evaluation.
> Run: runs/v5/v2/seed-5301/stage-b; partial controls: eval-000/.
> Post custody PASS; CONFIRM sealed=true/evaluated=false.
> See [V5_STAGE_B_RESULTS.md](V5_STAGE_B_RESULTS.md).
> Lower handoffs remain historical records and are superseded by this status.


> CURRENT STAGE A HANDOFF (2026-10-05): seed5301 Stage A completed1200 updates
> in one bounded CUDA FP32 invocation, no resumes; final model2d1c770a...ae0bb00.
> Final checkpoint integrity and post-run three-split custody PASS. Scientific
> source remains d11659e; all scientific code unchanged. The exactly-one final
> DEV B0 invocation FAILED report validation: sorted-ID hash compared to V2
> ProofTargets digest. No report/metrics serialized; no retry or code repair.
> STOP for owner review; Stage B/update0/optimizer/graphs NOT authorized.
> CONFIRM sealed=true/evaluated=false. See V5_STAGE_A_RESULTS.md.
> Prior STOP-before-Stage-A and P25/V1 text below is preserved historical status.


## Current V2 operation and Stage A-only authorization (2026-10-05)

The owner authorizes the complete frozen seed-5301 Stage A (1200 updates),
then exactly one final all-DEV B0 evaluation and owner review. Stage B, its
update-0 initialization, reader training and Q/R DEV science are NOT authorized.
Scientific source remains `d11659eca0774e0064bed0ef64ead2b725886d93`;
this amendment changes documentation only.

All lower references to active P25, 44,332 inherited positions, DEV 4,403,
KQRvK M3 n=507 and `runs/v25/p25/` commands are preserved HISTORICAL material.
They are superseded operationally by the measured native V2 lineage:

- TRAIN `V5_HP_TRAIN_V2`: 27,000, nine KQQvK/KQRvK/KRRvK x M1/M2/M3
  cells of 3,000; all records feed nine-cell balanced training.
- DEV `V5_HP_DEV_V2`: 4,500, six KQRvK/KRRvK x M1/M2/M3 cells of 750;
  primary KQRvK M3 n=750. Independent from TRAIN; never used during training.
- CONFIRM `V5_HP_CONFIRM_V2`: 4,500, sealed=true, evaluated=false.
- Active data contract `v5_hp_data_v2`; recipe `v5_stage_recipe_v3`.
- Stage A recipe digest
  `6642579e1f2472bda955ca7ada5bb3b8a435634c023b665684da1e4676347e70`.

Current commands from the V5 worktree after the documented process-local CUDA
setup and fresh three-split custody/recipe verification:

```powershell
.\target\release\recur64.exe v5 train --stage a --device cuda --data runs/v5/data/v2/v5-hp-train-v2.json --run-dir runs/v5/v2/seed-5301/stage-a --drill docs/evidence/v5/drill-d11659e-q8.json --microbatch 2 --qualification docs/evidence/v5/qualification-d11659e-cuda.json --max-minutes 45
if ($LASTEXITCODE -ne 0) { throw 'Stage A failed; preserve evidence and STOP' }
# Only after a clean bounded stop before update 1200:
.\target\release\recur64.exe v5 train --stage a --device cuda --data runs/v5/data/v2/v5-hp-train-v2.json --run-dir runs/v5/v2/seed-5301/stage-a --drill docs/evidence/v5/drill-d11659e-q8.json --microbatch 2 --qualification docs/evidence/v5/qualification-d11659e-cuda.json --max-minutes 45 --resume
if ($LASTEXITCODE -ne 0) { throw 'Stage A resume failed; STOP' }
# Exactly once, only after validated update 1200:
.\target\release\recur64.exe v5 evaluate-baseline --device cuda --data runs/v5/data/v2/v5-hp-dev-v2.json --stage-a runs/v5/v2/seed-5301/stage-a --microbatch 2 --qualification docs/evidence/v5/qualification-d11659e-cuda.json --output runs/v5/v2/seed-5301/stage-a/final-baseline-dev.json
if ($LASTEXITCODE -ne 0) { throw 'Final B0 evaluation failed; STOP' }
```

Fresh random initialization, FP32, physical2 x accumulation32, effective64,
seed5301, warmup80, peakLR3e-4, existing AdamW/loss/schedule; Q0, reader not
executed. No intermediate DEV, early stopping, checkpoint selection, adaptation
or extra updates. Preserve all checkpoints and failures. After final B0 and
post-run custody, STOP for owner review before any Stage B work.


> CURRENT V2 HANDOFF: consumer source d11659e, fresh CPU/RTX2050 CUDA exact D9
> PASS,12 repeated normal/profile comparisons exact, graph provenance PASS.
> Native TRAIN27000/DEV4500/CONFIRM4500 generated/audited/byte-identically
> regenerated, zero intersections, all actual local custody PASS. CONFIRM sealed
> and UNEVALUATED. Data-v2/recipe-v3 bound.24-position Q8/R4 drill PASS:
> 2.987107 ->0.026625 loss,99.11% reduction,200 updates, baseline exact.
> STOP BEFORE STAGE A. Stage A/B, DEV model science/pilot, controller, replication,
> V4_TUNE/HOLDOUT_C/CONFIRM evaluation NOT RUN. V1/P25 commands and earlier
> statuses below are preserved history. See V5_DATA_V2_REPORT.md.


> CURRENT CUDA RECOVERY (003d296): fresh CPU/CUDA qualification PASS under the
> unchanged exact D9 and synchronized execution contract. Three repetitions of
> each independent normal/profile order are EXACT; the proven cause was a
> pitched boolean-mask reshape stride defect. See V5_CUDA_MASK_RECOVERY.md.
> Historical failed reports are unchanged. Release workspace: 587 passed,
> zero failed, two preserved ignores. Changed-file rustfmt/V5 Clippy PASS;
> workspace-wide formatting has pre-existing drift in 16 unchanged files.
> Native dataset quota STOP remains; no drill, Stage A/B, DEV or pilot authorized.
> Historical status/commands below do not override this recovery or data lock.


> MEASURED DATA STOP (df261ab): exhaustive KRvK M1 capacity is only 189
> canonical classes, required 2000. Two full enumerations reproduce exactly;
> 189 independent audits, zero failures. No TRAIN/DEV/CONFIRM dataset accepted,
> no custody/seal or recipe-v2 integration. See V5_RECOVERY_REPORT.md. D9 FAIL.


> CURRENT OWNER AMENDMENT: P25 transfer/dependency is retired. New requested
> lineage is V5_HP_TRAIN_V1/DEV_V1/CONFIRM_V1; see V5_DATA_PLAN.md, registered
> before generation. Production scientific data loaders are locked until complete
> measured native DATA-B and recipe-v2 integration. The legacy P25 counts, digests,
> partition and commands below are historical and cannot authorize current work.
> CUDA diagnostic ad80002 is CASE C: clone purity PASS, each mode self-exact,
> cross-mode FAIL localized to frozen-root completion/lift fences. D9 remains
> FAIL. No Stage A/B, DEV science or drill authorized by this recovery result.


**PRE-PILOT, ENGINEERING STOP + CUSTODY BLOCKED:** current source
`d97004981f52eb077da6bb72584efaca341b8c5b`, FP32, physical microbatch 2:
synchronized CPU qualification PASS; CUDA qualification FAIL at normal/profile
exact output-gradient and AdamW parity, exit 1. Other recorded CUDA checks pass
but cannot override this failure. Full release workspace: 581 passed, zero failed,
two preserved ignores, native exit 0. Read `V5_PROFILING_ROOT_CAUSE.md` before
further execution. No gate/tolerance was loosened; no drill/pilot was launched.
Exact P25 TRAIN is also still required. The bounded GPU replay diagnostic and
standalone graph-source provenance audit precede any further qualification;
transferring TRAIN alone does NOT clear the engineering stop.
Read `V5_ROOT_CAUSE.md`, `V5_NUMERICAL_ROOT_CAUSE.md` and
`V5_EXECUTION_PARITY.md` for preserved failures and pre-pilot corrections.

## Worktree

```text
C:\Users\Caitl\Desktop\Code Projects\Recur64-v5
```

Branch: `experiment/hp-v5-counterfactual-loop`.
Historical stop checkpoint: `0c52b6e32d44e90107bcf34f3688377a0bc4ce31`.
The amended source must be rebuilt after its commit. Scientific identity is the
last commit touching `crates`, Cargo files or `configs`; documentation-only
commits do not change it. Previous depth-only configuration digest:
`d74109e229e49dc9962c348202db3527a5ce4c63da20efcd04a2bbe2577ff937`.
Current execution-amended digest, measured by rebuilt model-info:
`849133a5cdf169f187778bace2f858aa4747d2e8defef3bb5ac1bffc839774ee`.
Old qualifications cannot be reused across that identity change.

## Required missing artifact

Transfer the exact P25_DATA_V1 TRAIN file to a local ignored path, preferably:

```text
runs\v25\p25\data\proof-train.json
```

Required identity:

Machine-readable manifest: `docs/evidence/v5/required-artifacts.json`.

```text
positions: 44332
content digest: 3b25dc8549dd2fc9d47c30e294c273b3306aecb3eba91b964715326ddf74f2e6
FIT: 39929
FIT sorted-ID digest: a01932d7db863fbd0d160bc04bd3589137449d2c1e33d408f8d5d3f3cb45f8f5
DEV: 4403
DEV sorted-ID digest: f877219bc87d916ad8478a745f1572821c1a051337c3a071f2b37b9a5f83b899
```

Copy as a new file; do not replace another dataset. Run `recur64 v5 custody`
before any drill or training. A similar dataset or V3 TUNE is refused.
Regeneration is allowed only from the documented recipe with every exclusion
artifact and must reproduce the content digest bit-for-bit.

## Run safety

- Never start a fresh model in an existing run directory.
- Interrupted attempts are moved to a quarantine name, not deleted.
- Resume requires matching code/config/data/base/model hashes and restores model,
  optimizer, schedule, sampler and acquisition ordinal state.
- Every stage uses <=45-minute deterministic chunks and stops on unexpected
  failure.

## Process-local HP CUDA setup

In PowerShell, from the V5 worktree, before CUDA build OR execution:

    $env:CUDA_PATH = "$env:LOCALAPPDATA\Recur64\cuda\12.9.1"
    if (-not (Test-Path -LiteralPath "$env:CUDA_PATH\include\cuda_runtime.h")) { throw "Pinned CUDA headers missing" }
    $env:PATH = "$env:CUDA_PATH\bin;" + $env:PATH
    cmd.exe /d /s /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -no_logo -arch=x64 -host_arch=x64 && cargo build --release -p recur64-cli --features cuda'
    if ($LASTEXITCODE -ne 0) { throw "V5 CUDA build failed" }

Both CUDA_PATH (NVRTC headers) and PATH (DLLs) are required. Setting only PATH
failed visibly in V5-E13; CubeCL otherwise selected the default Windows toolkit
directory. No permanent environment or toolkit installation is changed. Do not
use --features tf32: V5 now refuses that incompatible build. Run builds and CLI
tests serially, because Windows can prevent replacement of a running executable.

## Executed commands

From the V5 worktree:

    cargo run -p recur64-cli -- v5 doctor --output docs/evidence/v5/doctor.json
    cargo run -p recur64-cli -- v5 model-info --json docs/evidence/v5/model-info.json
    cargo run -p recur64-cli -- v5 custody --data runs/v25/p25/data/proof-train.json
    cargo run -p recur64-cli -- v5 graph generate --fen "6k1/8/8/8/8/8/4Q3/3RK3 w - - 0 1" --position-id cli-fixture --schedule uniform-frontier --q 4 --output runs/v5/fixtures/graph-q4.json
    cargo run -p recur64-cli -- v5 graph audit --graph runs/v5/fixtures/graph-q4.json
    cargo run -p recur64-cli -- v5 qualify --device cpu --microbatch 2 --output docs/evidence/v5/cpu-qualification-debug.json
    cargo build --release -p recur64-cli --features cuda
    .\target\release\recur64.exe v5 qualify --device cuda --microbatch 2 --output docs/evidence/v5/cuda-qualification.json
    .\target\release\recur64.exe v5 qualify --device cpu --microbatch 2 --output docs/evidence/v5/cpu-qualification-release.json

The model-info and graph generate/audit examples have executed successfully.
Custody is expected to exit nonzero until the exact artifact is transferred. The
first CPU qualification attempt failed with a default-thread stack overflow;
after the explicit V5 64 MiB worker-stack boundary, the shown qualification
command executed successfully. A release build with `--no-default-features`
failed at compile time because the pinned model crate exposes an unconditional
CPU type alias. `--features cuda` is the tested compatible build. Both release
historical qualification reports bind to source
`df6e2aa650c12726ad7094dae04a7c73339139a8` and remain archived under
`qualification-df6e2aa-{cpu,cuda}.json`. They are not current qualifications.

Historical executed commands, after the process-local setup and clean source build:

    .\target\release\recur64.exe v5 qualify --device cpu --microbatch 2 --output docs/evidence/v5/qualification-64c4dd4-cpu.json
    .\target\release\recur64.exe v5 qualify --device cuda --microbatch 2 --output docs/evidence/v5/qualification-64c4dd4-cuda.json

Both passed at 64c4dd4. Current executed commands at committed/rebuilt d970049:

    .\target\release\recur64.exe v5 qualify --device cpu --microbatch 2 --output docs/evidence/v5/qualification-d970049-cpu.json
    .\target\release\recur64.exe v5 qualify --device cuda --microbatch 2 --output docs/evidence/v5/qualification-d970049-cuda.json

CPU exits 0/PASS; CUDA exits 1/FAIL. Canonical qualification paths now contain
these current reports, so dataset-dependent CUDA commands below MUST refuse.
Do not rerun the failed command automatically, bypass the loader, reuse the old
passing report or launch a drill/LR screen. Physical
layout is resolved as 2 positions: Stage A accumulation 32, Stage B accumulation
18. No microbatch-1 fallback was used. Drill qualification is still NOT RUN;
fixture qualification alone does not authorize bypassing the required drill,
and the current profiling failure is an additional hard stop.

Scientific source identity is the last commit touching `crates`, Cargo
manifests/lock or `configs`; staged, unstaged and untracked scientific changes
are refused, as is a stale binary built against a different scientific commit.
Documentation/evidence-only commits do not invalidate the executable identity.

## Dataset-dependent command sequence (NOT RUN)

After transfer, custody and the disposable drill are:

    .\target\release\recur64.exe v5 custody --data runs/v25/p25/data/proof-train.json --output docs/evidence/v5/custody.json
    .\target\release\recur64.exe v5 drill --device cuda --data runs/v25/p25/data/proof-train.json --microbatch 2 --qualification docs/evidence/v5/cuda-qualification.json --output docs/evidence/v5/drill-q8.json

Do not run Q16 if Q8 passes or is uninformative. Only an informative Q8 failure
permits the fresh disposable diagnostic:

    .\target\release\recur64.exe v5 drill --device cuda --data runs/v25/p25/data/proof-train.json --q 16 --q8-report docs/evidence/v5/drill-q8.json --microbatch 2 --qualification docs/evidence/v5/cuda-qualification.json --output docs/evidence/v5/drill-q16.json

Only AFTER exact custody, the qualifying FIT drill AND complete detailed
performance accounting (still pending), the frozen pilot stage commands are:

    cargo run --release -p recur64-cli --features cuda -- v5 train --stage a --device cuda --data runs/v25/p25/data/proof-train.json --run-dir runs/v5/seed-5301/stage-a --microbatch 2 --qualification docs/evidence/v5/cuda-qualification.json --drill docs/evidence/v5/drill-q8.json

After the completed update-1200 Stage A checkpoint, record its separate Q0 DEV
baseline before initializing Stage B:

    .\target\release\recur64.exe v5 evaluate-baseline --device cuda --data runs/v25/p25/data/proof-train.json --stage-a runs/v5/seed-5301/stage-a --microbatch 2 --qualification docs/evidence/v5/cuda-qualification.json --output runs/v5/seed-5301/stage-a/final-baseline-dev.json
    cargo run --release -p recur64-cli --features cuda -- v5 train --stage b --device cuda --data runs/v25/p25/data/proof-train.json --run-dir runs/v5/seed-5301/stage-b --stage-a runs/v5/seed-5301/stage-a --stage-a-evaluation runs/v5/seed-5301/stage-a/final-baseline-dev.json --microbatch 2 --qualification docs/evidence/v5/cuda-qualification.json --drill docs/evidence/v5/drill-q8.json

If Q8 failed informatively and the one matched Q16 diagnostic passed, add
`--drill-q16 docs/evidence/v5/drill-q16.json` to both training stages and
`pilot-report`. The Q8 report remains required; Q16 is never relabelled as Q8.
An uninformative, non-finite or baseline-changing drill cannot unlock training.
Stage B verifies the baseline report against the actual Stage A checkpoint,
the authoritative DEV labels/actions and the frozen device/layout.

Each invocation projects remaining work before starting and stops after at most
45 minutes. Add --resume to the identical command to continue the latest
complete immutable checkpoint generation. These stage commands have NOT RUN
because custody is blocked by the missing exact dataset.

Stage B saves update 0 before its first optimizer step. Evaluate only update 0
and the fixed final update 800. Run one bounded cell per process for each update:

    $cells = @(@('KQRvK',1),@('KQRvK',2),@('KQRvK',3),@('KRRvK',1),@('KRRvK',2),@('KRRvK',3))
    foreach ($cell in $cells) {
      $family = $cell[0]; $depth = $cell[1]
      .\target\release\recur64.exe v5 evaluate --device cuda --data runs/v25/p25/data/proof-train.json --stage-b runs/v5/seed-5301/stage-b --update 800 --family $family --mate-depth $depth --microbatch 2 --qualification docs/evidence/v5/cuda-qualification.json --output "runs/v5/seed-5301/eval-800/$family-M$depth.json"
      if ($LASTEXITCODE -ne 0) { throw "V5 evaluation failed for $family M$depth" }
    }

Repeat with `--update 0` and a distinct `eval-000` directory. Never overwrite
one update with the other. Merge each six-shard set with all inputs following one
`--input` flag:

    .\target\release\recur64.exe v5 eval-merge --input runs/v5/seed-5301/eval-800/KQRvK-M1.json runs/v5/seed-5301/eval-800/KQRvK-M2.json runs/v5/seed-5301/eval-800/KQRvK-M3.json runs/v5/seed-5301/eval-800/KRRvK-M1.json runs/v5/seed-5301/eval-800/KRRvK-M2.json runs/v5/seed-5301/eval-800/KRRvK-M3.json --output runs/v5/seed-5301/eval-800/all-dev.json
    .\target\release\recur64.exe v5 ablation --evaluation runs/v5/seed-5301/eval-800/all-dev.json --output runs/v5/seed-5301/eval-800/ablations.json
    .\target\release\recur64.exe v5 pilot-report --evaluation runs/v5/seed-5301/eval-800/all-dev.json --cpu-qualification docs/evidence/v5/cpu-qualification-release.json --cuda-qualification docs/evidence/v5/cuda-qualification.json --drill docs/evidence/v5/drill-q8.json --output runs/v5/seed-5301/pilot-report.json

The merge refuses overlapping or incomplete cell shards and verifies 4,403 DEV
positions plus 507 KQRvK M3 positions. `pilot-report` recomputes all summaries
from per-position records. Only a `PILOT_CANDIDATE` report unlocks:

    .\target\release\recur64.exe v5 extra-loops --device cuda --data runs/v25/p25/data/proof-train.json --stage-b runs/v5/seed-5301/stage-b --evaluation runs/v5/seed-5301/eval-800/all-dev.json --pilot-report runs/v5/seed-5301/pilot-report.json --microbatch 2 --qualification docs/evidence/v5/cuda-qualification.json --output runs/v5/seed-5301/r8-primary.json

All commands in this section are implemented and their argument surfaces are
covered by CLI tests. They are explicitly NOT RUN on this machine because the
required P25 artifact is absent.

## Current Stage A artifacts and owner review stop

Final checkpoint: runs/v5/v2/seed-5301/stage-a/checkpoints/update-000000001200.
Ignored operational receipts/logs: runs/v5/v2/seed-5301/preflight/.
Final B0 output runs/v5/v2/seed-5301/stage-a/final-baseline-dev.json DOES NOT EXIST:
the single command failed validation before publication. Do not rerun the earlier
example or resume completed Stage A. Read V5_STAGE_A_RESULTS.md and the compact
baseline-failure receipt. Next step is owner review of the narrow evaluation
identity-contract defect; no source amendment or repeat evaluation is authorized
by this handoff. Stage B/update0/graphs/optimizer remain NOT RUN.

## Current handoff: baseline v3 publication complete; owner review STOP

Evaluator source b00569b33ced75a0169804a4a3d5b746a1e0e654; immutable Stage A source
d11659eca0774e0064bed0ef64ead2b725886d93. Completed Stage A checkpoint/model/optimizer
and every existing run file are unchanged. The first failed attempt is preserved.
Replacement overall attempt2 published exactly one v3 result outside the old run:
runs/v5/v2/seed-5301/baseline-recovery/final-baseline-dev-v3.json.
Compact result: docs/evidence/v5/baseline-v3-seed5301-summary.json.
Fresh current CPU/CUDA qualification and pre/post raw-file custody PASS.
CONFIRM sealed=true/evaluated=false. Do not repeat baseline evaluation or resume
Stage A. Stage B's original source equality guard is unchanged and refuses this
predecessor artifact. Owner review and separate authorization are required for
any next phase; no Stage B update0, optimizer or training graphs exist.
