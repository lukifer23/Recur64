# Recur64

A Rust-first chess-learning research laboratory. The eventual question: at
matched end-to-end compute, does a small **recurrent** square-token transformer
that spends compute on internal refinement beat spending the same compute on
external search?

This repository currently contains **Phase 0 only**: a systems probe that proves
the model-shaped graph trains on the target workstation with honest device and
precision reporting. It is **not** a chess engine and contains no chess rules,
search, self-play, replay, or UCI.

## Requirements

- Rust toolchain 1.97.1 (see `rust-toolchain.toml`).
- Windows x86_64 with the configured linker (this machine uses `rust-lld` + a
  bundled MSVC/SDK library set; see `docs/HARDWARE.md`).
- No CUDA runtime is required for the CPU path. The GPU path is pending (see
  `docs/DECISIONS.md` D3).

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

## Layout

```
crates/recur64-model   probe graph, heads, losses, recurrence, optimizer,
                       checkpoint, precision gate, fixtures
crates/recur64-cli     `recur64` binary: doctor | model-info | bench
configs/               micro.toml, f10.toml, r10-probe.toml
docs/                  HARDWARE, ARCHITECTURE, BENCHMARKS, DECISIONS, STATUS,
                       plus the preserved master spec and Phase 0 kickoff
```

## What Phase 0 does and does not prove

Proves: the real model-shaped graph forwards, backpropagates, optimizes,
checkpoints, and resumes in Rust; recurrent shared weights receive gradients;
policy masking/promotions/terminal handling are correct; compute accounting is
exact. Does **not** prove any chess strength or that recurrence helps. See
`docs/STATUS.md` and `docs/ARCHITECTURE.md`.
