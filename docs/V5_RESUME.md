# V5 HP resume and artifact transfer

## Worktree

```text
C:\Users\Caitl\Desktop\Code Projects\Recur64-v5
```

Branch: `experiment/hp-v5-counterfactual-loop`.

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

Copy as a new file; do not replace another dataset. Run `recur64 v5 custody
verify` before any drill or training. A similar dataset or V3 TUNE is refused.
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
`028025da1c7486eb0aa9140a88509c2822275e8a`.

With release CPU and CUDA qualification passing, the frozen pilot stage commands
are:

    cargo run --release -p recur64-cli --features cuda -- v5 train --stage a --device cuda --data runs/v25/p25/data/proof-train.json --run-dir runs/v5/seed-5301/stage-a --microbatch 2 --qualification docs/evidence/v5/cuda-qualification.json
    cargo run --release -p recur64-cli --features cuda -- v5 train --stage b --device cuda --data runs/v25/p25/data/proof-train.json --run-dir runs/v5/seed-5301/stage-b --stage-a runs/v5/seed-5301/stage-a --microbatch 2 --qualification docs/evidence/v5/cuda-qualification.json

Each invocation projects remaining work before starting and stops after at most
45 minutes. Add --resume to the identical command to continue the latest
complete immutable checkpoint generation. These stage commands have NOT RUN
because custody is blocked by the missing exact dataset.
