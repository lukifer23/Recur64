# Recur64

A Rust-first chess-learning research laboratory. The eventual question: at
matched end-to-end compute, does a small **recurrent** square-token transformer
that spends compute on internal refinement beat spending the same compute on
external search?

**Status (Phase 4, in progress).** The full learning loop exists and runs
on the RTX 2000 Ada workstation: PUCT self-play → batched GPU inference →
replay → audit → train → checkpoint → searched and raw arenas → report.
Phase 4 is producing the first trustworthy mainline F10 evidence on it. See
`docs/STATUS.md` for the gate record and `docs/PHASE4_RESULTS.md` for measured
Phase 4 results.

| phase | scope | gate |
|---|---|---|
| 0 | systems probe: model-shaped graph trains on this machine (CPU + CUDA FP32) | GO |
| 1 | chess contracts: observation V1, action V1, rules profile, perft, oracle | GO |
| 2 | first vertical slice (Micro model) | GO |
| 3 | F10 + PUCT control baseline, bounded pilots | CONDITIONAL GO (historical; predates the Phase 4 fixes) |
| 4 | mainline harness convergence + GPU requalification | P4.2–P4.5 done; post-smoke fixes; F10 smoke v2 GO for the learning mechanism (strength not yet shown) |

Phase 4 so far:

- The main-workstation hardware schedule is **measured**.
- A root-cause audit replaced the model's readout head (**head v2**,
  `docs/DECISIONS.md` D40) and added standard self-play exploration: root
  Dirichlet noise and argmax after ply 30 (D41). Together they turned
  repetition-dominated self-play into mostly decisive games.
- The F10 search budget is requalified at **64 simulations/move**.
- A GPU memory defect in the inference-owner lifecycle was found and fixed
  (D44).
- **First corrected F10 smoke: CONDITIONAL.** The loop was interpretable,
  but the arena was repetition-dominated and never promoted a candidate.
- **Post-smoke fixes:**
  - arena exploration (D45)
  - multi-leaf PUCT with virtual loss, +83% throughput (D47)
  - a continuous trainer (D48)
  - an evaluation deadline, crash-safe replay archival, and at most two
    resident models (D38 / D37 / D46)
- **F10 smoke v2: GO for the learning mechanism.**
  - Training compounds across cycles and candidates are promoted.
  - Once the promoted value head guides self-play, search moves the policy
    target on 37% of positions (11% before).
  - Playing strength over the untrained reference is not yet demonstrated.

This is a research laboratory, **not** a chess engine. It has no UCI engine
loop and makes no strength claims.

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

### Phase 3 F10 baseline (CUDA)

```sh
# Stage A: warmup + batching/active-game sweep.
cargo run --release -p recur64-cli --features cuda -- bench-runtime \
    --config configs/f10-sweep.toml --output runs/sweep --grid small --games-per-cell 128

# Generate the frozen evaluation opening suite.
cargo run --release -p recur64-cli -- gen-openings --output configs/openings-v1.toml

# Raw-policy evaluation (no search) of a checkpoint.
cargo run --release -p recur64-cli --features cuda -- eval-policy \
    --config configs/f10-stage-c.toml --checkpoint runs/<run>/checkpoints/candidate --output runs/<run>/eval

# Bounded multi-cycle pilot (collect -> train -> evaluate).
cargo run --release -p recur64-cli --features cuda -- pilot \
    --config configs/f10-stage-c.toml --run-dir runs/f10-stage-c-1 --force
```

See `docs/F10_BASELINE.md` for measured results and the long-run decision.

### Phase 4 (CUDA, main workstation)

```sh
# Freeze one seeded, untrained reference that every Phase 4 step reuses.
recur64 freeze-reference --config configs/phase4/f10-reference.toml \
    --output runs/phase4-f10-reference-v2

# Scheduling sweep (games per cell must realize each requested concurrency).
recur64 bench-runtime --config configs/phase4/f10-reference.toml \
    --checkpoint runs/phase4-f10-reference-v2 --grid workstation --games-per-cell 32
recur64 bench-runtime --config configs/phase4/f10-reference.toml \
    --checkpoint runs/phase4-f10-reference-v2 --active 32 --max-batch 32 \
    --timeout-us 500 --simulations 64 --games-per-cell 64 \
    --output runs/sweep-s64 --replay-output runs/sweep-s64/replay

# Learner throughput per physical x accumulation layout (effective batch fixed).
recur64 bench-train --config configs/phase4/f10-reference.toml \
    --checkpoint runs/phase4-f10-reference-v2 --replay runs/sweep-s64/replay \
    --layouts 64x4 --output runs/train-64x4

# Prior vs search-target divergence of a replay (generating network only).
recur64 search-gain --config configs/phase4/f10-reference.toml \
    --checkpoint runs/phase4-f10-reference-v2 --replay runs/sweep-s64/replay \
    --output runs/sweep-s64

# GPU inference-owner lifecycle probe (measured 32-way schedule).
recur64 bench-lifecycle --config configs/phase4/f10-reference.toml \
    --checkpoint runs/phase4-f10-reference-v2 --reps 8 --output runs/lifecycle \
    --arena-games 32 --concurrency 32 --max-batch 32 --timeout-us 500

# Corrected two-cycle F10 learning smoke (frozen, pre-registered config).
recur64 pilot --config configs/phase4/f10-smoke.toml --run-dir runs/phase4-f10-smoke
```

`recur64` is `target/release/recur64` built with `--features cuda` and run
with the CUDA environment below. The measured schedule is in
`configs/hardware/workstation-main.toml`.

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
crates/recur64-search  PUCT (+ optional root noise), Evaluator trait, chess
                       adapter, game play, deterministic RNG (no Burn)
crates/recur64-model   square-token transformer, readout head v2, losses,
                       recurrence, optimizer, checkpoint, precision gate
crates/recur64-runtime inference owner/batcher, replay, learner, coordinator,
                       pilot, sweep, GPU telemetry, search gain
crates/recur64-eval    paired-color arenas, opening suites
crates/recur64-cli     `recur64` binary: doctor | model-info | bench | cuda-smoke
                       | perft | validate-position | encode | bench-core
                       | selfplay | replay-audit | train | arena | run | report
                       | bench-runtime | bench-train | bench-lifecycle
                       | search-gain | gen-openings | eval-policy | pilot
                       | freeze-reference
configs/               historical Phase 0–3 configs; configs/phase4/ (current
                       mainline contracts); configs/hardware/ (measured
                       machine scheduling profiles)
docs/                  STATUS, PHASE4_RESULTS, PHASE4_CONVERGENCE, DECISIONS
                       (ADRs), HARDWARE, ARCHITECTURE, BENCHMARKS,
                       REPRESENTATIONS, RULES_PROFILE, SEARCH, REPLAY, RUNS,
                       F10_BASELINE, evidence/phase4/, plus the preserved specs
```

## What is and is not established

Established, with evidence in `docs/STATUS.md` and
`docs/PHASE4_RESULTS.md`:

- The model graph trains, checkpoints and resumes in Rust on CPU and CUDA
  FP32.
- Recurrent shared weights receive gradients, and F10 and R10 are matched in
  unique parameters (9,805,672 under head v2).
- Chess contracts are exact (perft, independent oracle).
- The self-play → learning loop closes truthfully: audited replay, provenance,
  identity hashes, and refusal of mismatched checkpoints.
- The main-workstation schedule is measured.
- A fresh network now starts from a near-uniform policy and a neutral value.

**Not** established:

- any chess strength
- that F10 gains playing strength at this scale. Smoke v2 shows the
  learning mechanism working (value learning, promotions, value-guided
  search), but 0.500 against the untrained reference.
- that recurrence helps (the R10 R1/R2/R4 experiments have not started)
