# V5 HP resume and artifact transfer

**STOP:** V5-E5 found a complete-uniform-frontier contract violation. Read
`docs/V5_ROOT_CAUSE.md` before executing the sequence below. Dataset transfer does
not clear this engineering stop. The current qualification preflight refuses
the known violation. A reviewed depth-contract correction, passing regression
and fresh CPU/CUDA qualification are required before any drill or training.

## Worktree

```text
C:\Users\Caitl\Desktop\Code Projects\Recur64-v5
```

Branch: `experiment/hp-v5-counterfactual-loop`.
Current scientific checkpoint: `0c52b6e32d44e90107bcf34f3688377a0bc4ce31`.
It contains the retained failing regression and qualification preflight, not a
depth-representation fix. Subsequent documentation-only commits do not change
this scientific source identity.

## Required missing artifact

Transfer the exact P25_DATA_V1 TRAIN file to a local ignored path, preferably:

```text
runs\v25\p25\data\proof-train.json
```

Required identity:

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
qualification reports pass and bind to source
`df6e2aa650c12726ad7094dae04a7c73339139a8`.

Scientific source identity is the last commit touching `crates`, Cargo
manifests/lock or `configs`; tracked uncommitted changes in those paths are
refused. Documentation/evidence-only commits therefore do not invalidate the
qualified executable identity.

## Dataset-dependent command sequence (NOT RUN)

After transfer, custody and the disposable drill are:

    .\target\release\recur64.exe v5 custody --data runs/v25/p25/data/proof-train.json --output docs/evidence/v5/custody.json
    .\target\release\recur64.exe v5 drill --device cuda --data runs/v25/p25/data/proof-train.json --microbatch 2 --qualification docs/evidence/v5/cuda-qualification.json --output docs/evidence/v5/drill-q8.json

Do not run Q16 if Q8 passes or is uninformative. Only an informative Q8 failure
permits the fresh disposable diagnostic:

    .\target\release\recur64.exe v5 drill --device cuda --data runs/v25/p25/data/proof-train.json --q 16 --q8-report docs/evidence/v5/drill-q8.json --microbatch 2 --qualification docs/evidence/v5/cuda-qualification.json --output docs/evidence/v5/drill-q16.json

With release CPU and CUDA qualification passing, the frozen pilot stage commands
are:

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
