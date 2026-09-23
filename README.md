# Recur64

A Rust-first chess-learning research laboratory. The eventual question: at
matched end-to-end compute, does a small **recurrent** square-token transformer
that spends compute on internal refinement beat spending the same compute on
external search?

The repository currently contains **Phase 0** (a systems probe proving the
model-shaped graph trains on this workstation), **Phase 1** (explicit, tested
chess contracts: observation V1, action V1, rules profile, perft), and **Phase 2**
(the first complete vertical slice: PUCT self-play → batched inference → replay →
audit → train → checkpoint → arena → report). It is **not** a chess engine and
contains no UCI engine loop or strength claims.

> **Experimental branch note.** This checkout is `experiment/hp-r15`, a separate
> experimental lineage forked from Phase 2 (`78be205`). It explores a ~15M
> matched-parameter F15/R15 recurrence-vs-search comparison on a home HP machine.
> It is not mainline and must not be merged without a separate decision. See
> `docs/HP_EXPERIMENT.md` and `docs/HP_CHANGES.md`; machine-specific detail stays
> in those files.

## Requirements

- Rust toolchain 1.97.1 (see `rust-toolchain.toml`).
- Windows x86_64 with the configured linker (this machine uses `rust-lld` + a
  bundled MSVC/SDK library set; see `docs/HARDWARE.md`).
- No CUDA runtime is required for the CPU path. The GPU path uses a user-space
  CUDA 12.9.1 runtime under `%LOCALAPPDATA%\Recur64\cuda\12.9.1` (see
  `docs/DECISIONS.md` D3); no admin rights or system changes are needed.

## Build and test

```sh
cargo build --release
cargo test
cargo fmt --all --check
cargo clippy --workspace --all-targets
```

## Commands

```sh
# Read-only environment report (DETECTED vs TESTED).
cargo run --release -- doctor

# Exact parameter counts and executed-block accounting.
cargo run --release -- model-info --config configs/micro.toml
cargo run --release -- model-info --config configs/f10.toml
cargo run --release -- model-info --config configs/r10-probe.toml

# Bounded benchmark matrix; writes runs/<name>/bench.json and bench.md.
cargo run --release -- bench --config configs/micro.toml --output runs/micro-cpu \
    --inference-batches 1,16,64,128 --recurrences 1,2,4 \
    --train-batches 32 --iters 5 --warmup 2 --train-steps 3
```

Use `--release` for any throughput measurement.

### Chess contracts (Phase 1, CPU-only)

```sh
# Count legal move tree nodes (validates move generation/conversion).
cargo run --release -p recur64-cli -- perft \
    --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" --depth 5

# Print termination/rules/legal-candidate facts for a position.
cargo run --release -p recur64-cli -- validate-position \
    --fen "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"

# Encode a position as Observation V1 and list canonical legal actions.
cargo run --release -p recur64-cli -- encode \
    --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1" --actions

# Phase 1 CPU baselines (movegen/apply/encode/perft throughput).
cargo run --release -p recur64-cli -- bench-core --output runs/bench-core
```

Optional independent differential oracle (GPL-3.0, dev/test only, off by default):

```sh
cargo test -p recur64-core --features oracle
```

### Phase 2 vertical slice (CPU)

```sh
# Bounded collect -> audit -> train -> evaluate -> report.
cargo run --release -p recur64-cli -- run --config configs/smoke.toml \
    --run-dir runs/smoke-1 --force

# Individual stages.
cargo run --release -p recur64-cli -- selfplay --config configs/smoke.toml --output runs/sp
cargo run --release -p recur64-cli -- replay-audit --input runs/sp
cargo run --release -p recur64-cli -- report --run-dir runs/smoke-1
```

The GPU variant is `configs/smoke-cuda.toml` (add `--features cuda` and the CUDA
environment below).

### GPU (CUDA)

```sh
$env:CUDA_PATH = "$env:LOCALAPPDATA\Recur64\cuda\12.9.1"
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"

# Proof that the real graph runs on the GPU (forward R=1/2/4, backward,
# AdamW update, checkpoint restore).
cargo run --release -p recur64-cli --features cuda -- cuda-smoke --config configs/r10-probe.toml

# Synchronized GPU benchmark.
cargo run --release -p recur64-cli --features cuda -- bench \
    --config configs/r10-probe.toml --device cuda --output runs/r10-cuda \
    --inference-batches 1,16,64,128 --recurrences 1,2,4 \
    --train-batches 32,64,128 --iters 20 --warmup 5 --train-steps 3
```

## Layout

```
crates/recur64-core    chess contracts: squares, actions, GameState, rules,
                       observation V1, UCI, perft (CPU-only, no Burn)
crates/recur64-search  PUCT, Evaluator trait, chess adapter, game play (no Burn)
crates/recur64-model   probe graph, heads, losses, recurrence, optimizer,
                       checkpoint, precision gate, fixtures
crates/recur64-runtime inference owner/batcher, replay, learner, coordinator
crates/recur64-eval    paired-color systems arena
crates/recur64-cli     `recur64` binary: doctor | model-info | bench | cuda-smoke
                       | perft | validate-position | encode | bench-core
                       | selfplay | replay-audit | train | arena | run | report
configs/               micro.toml, f10.toml, r10-probe.toml, smoke.toml,
                       smoke-cuda.toml
docs/                  HARDWARE, ARCHITECTURE, BENCHMARKS, DECISIONS, STATUS,
                       REPRESENTATIONS, RULES_PROFILE, SEARCH, REPLAY, RUNS,
                       plus the preserved specs
```

## What Phase 0 does and does not prove

Proves: the real model-shaped graph forwards, backpropagates, optimizes,
checkpoints, and resumes in Rust; recurrent shared weights receive gradients;
policy masking/promotions/terminal handling are correct; compute accounting is
exact. Does **not** prove any chess strength or that recurrence helps. See
`docs/STATUS.md` and `docs/ARCHITECTURE.md`.
