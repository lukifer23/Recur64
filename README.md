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

> **Experimental branch note.** The current checkout is
> `experiment/hp-r15-h3-integration`, a separate experimental lineage forked
> from Phase 2 (`78be205`), reconciled with Phase 3 (`5ac291c`, merge `fa66c32`)
> and then merged with mainline `03e62f7` (merge `a926fa4`). It explores a ~15M
> matched-parameter F15/R15 recurrence-vs-search comparison on a home HP machine
> (RTX 2050, 4 GB). It is not mainline. See `docs/HP_EXPERIMENT.md` and
> `docs/HP_CHANGES.md`; machine-specific detail stays in those files.

**HP H3 (head v2, 2026-09-26) requalifies F15** after transferring the mainline
mechanisms. Measured on the RTX 2050: head-v2 F15/R15 are matched at 15,154,632
unique params; the D44 owner-memory fix holds (496 MB plateau vs the historical
3,909 MiB); K = 2, concurrency 8, search budget 32 are adopted from measurement;
and self-play draws fell to 22–28% from the historical 89.2%. R15 training
remains **not run**.

**H3.5B pre-smoke red team (2026-09-26):**
- **Weights reproduce.** `model_id` hashes the `.mpk` artifact, which embeds
  random ParamIds. Weights reproduce exactly per backend, measured with
  `recur64 model-digest` (D50 amended).
- **Paired arena.** The arena now pairs colour-swapped games on a common
  random stream (`arena_rng_policy = "paired_common_v1"`). The reference vs
  itself scores exactly 0.500, where it scored 0.362 before.
- **Smoke config fixed.** The LR schedule has headroom (370/37), D49 health
  stops are restored, and an LR-exhaustion guard is added.
- **Hardening.** Misplaced config keys are now refused (D51), and CLI evidence
  records the git SHA.

Pre-flight: `recur64 config-info --config configs/hp/f15-smoke-v2.toml`.

**HP perf pass (2026-09-27, [`docs/PERF_LEDGER.md`](docs/PERF_LEDGER.md)).**
The accepted HP build is `cargo build --release -p recur64-cli --features
cuda,fusion,autotune` with `inference_candidate_buckets = true` (D55).
Compared with the plain build, measured on a trained network:

- **Self-play:** +30 %.
- **Training:** +11 %.
- **Forward pass:** +38–63 %.
- **Arenas:** +8 %. They are latency-bound, so D56 early adjudication is the
  next lever there.
- **VRAM lifecycle:** flat, after a fusion-specific leak was root-caused and
  fixed.

**R15 P1 (2026-09-27):** 3-cycle smokes at R1/R2/R4 on the D55 build are all
CONDITIONAL. The mechanism is GO, and the flags are all truncation.

- The key finding is that R4's value learning stalls at the shared LR, while
  R1 and R2 learn.
- No strength claim yet.
- P2 (2 seeds × 8 cycles) is proposed in
  [`docs/HP_R15_RESULTS.md`](docs/HP_R15_RESULTS.md).

Fast probes:

- `recur64 bench-forward`: the production batch path, with parity against a
  saved baseline.
- `recur64 bench-runtime --games-per-cell 8`.
- `recur64 eval-arena --arena-games 8`.

**H3.6 corrected F15-v2 smoke: CONDITIONAL (2026-09-26)**
- **GO:** learning mechanism, continuous trainer, lifecycle, reuse and
  lineage. Self-play draws fell 0.22 → 0.09.
- **Strength:** the cycle-2 candidate beats the frozen reference 0.733
  [0.621, 0.846].
- **CONDITIONAL:** arena truncation was root-caused to won-but-unconverted
  positions, which the score silently drops.
- **R15 entry:** CONDITIONAL GO for planning only. Truncation-aware scoring
  must be pre-registered first, and R15 has not been trained.

- Historical head-v1 HP evidence (frozen): [`docs/HP_H1_RESULTS.md`](docs/HP_H1_RESULTS.md).
- H3 pre-registration: [`docs/HP_H3_PREREG.md`](docs/HP_H3_PREREG.md).
- H3 measured results: [`docs/HP_H3_RESULTS.md`](docs/HP_H3_RESULTS.md).

## Requirements

- Rust toolchain 1.97.1 (see `rust-toolchain.toml`).
- Windows x86_64 with Visual Studio 2022 Build Tools on the HP machine.
  Workstation linker details are recorded in `docs/HARDWARE.md`.
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

---

## X15 / "Chimera" � the experimental novel-architecture line

Branch `experiment/hp-r15-h3-integration` is the deliberately experimental
Recur64 line; `main` is the conservative conventional control line. X15 combines
four independently gated pathways:

1. a symbolic geometry-aware square-token transformer prelude;
2. an explicit recurrent latent reasoning scratchpad (`K` persistent thought
   slots, position-conditioned at thought 0);
3. a deterministic chess coprocessor exposed through cross-attention, with a
   **real WebAssembly** implementation (`wasm_v1`) and a native one (`native_v1`)
   that is byte-identical to it;
4. a literal canonical visual-board pathway: a small residual CNN over a
   deterministic top-down render.

**Probe harness** (the owner's philosophy: long runs do not come first):

```bash
# parameter accounting by subsystem, and the T1/T2/T4 block accounting
recur64 x15 info    --config configs/x15.toml

# P0: finite forward at T1/T2/T4, policy normalizes, WDL neutral at init
recur64 x15 sanity  --config configs/x15.toml --thoughts 1,2,4

# P0 gate: native vs WebAssembly ComputeBankV1, byte for byte
recur64 x15 parity  --positions 200

# P0: every gated subsystem gets a non-zero gradient on the first step
recur64 x15 grads   --config configs/x15.toml --thoughts 4

# P1: per-thought metrics (entropy, WDL, latent norm/delta, pathway scales)
recur64 x15 thoughts --config configs/x15.toml --thoughts 4
recur64 x15 bench --mode infer --config configs/x15_cuda.toml   # or --mode train --batch 32 --accum 4
```

More commands (all fast; cold GPU kernel compilation costs minutes once per shape):

```bash
# fixed-data teacher targets (deterministic PUCT ladder), batched through the inference owner
recur64 x15 gen-targets --teacher-config <run>/config.toml --teacher-checkpoint <ckpt> \
    --replay <run>/replay --out targets.json --positions 400 --synthetic-tactics 80 --batched
recur64 x15 audit-targets --targets targets.json --report --disjoint-from other.json

# train on the targets (micro-batched, seeded order), evaluate the SAME weights at T=1..N
recur64 x15 train-probe   --targets targets.json --out ckpt --thoughts 4 --batch-positions 96
recur64 x15 eval-reasoning --targets confirm.json --split confirm --checkpoint ckpt --thoughts 4
recur64 x15 compare --a ckA1 --a ckA2 --t-a 4 --b ckB1 --b ckB2 --t-b 1 --fixtures f.json --targets t.json

# tactical suite: generate, evaluate networks, and score the teacher as a ceiling
recur64 x15 gen-tactics --out tactics.json
recur64 x15 eval-tactics --fixtures tactics.json --checkpoint ckpt
recur64 x15 teacher-tactics --teacher-config ... --teacher-checkpoint ... --fixtures tactics.json
recur64 x15 facts-probe --fixtures tactics.json --checkpoint ckpt
```

Current measured state (the full MEASURED / INFERRED / NOT RUN split, every
experiment, its pre-registered rule and its outcome are in
`docs/HP_X1_EXPERIMENTS.md`; decisions in `docs/DECISIONS.md` D57-D63):

- **16.0M parameters**, identical at T = 1/2/4/8. X15 **runs and trains on the RTX 2050**:
  a T=4 training update of 96 positions takes about 1.4 s and peaks at 2.4 GB; the
  normal forward costs 91 / 108 / 142 ms at T = 1 / 2 / 4 (batch 32).
- Native and WASM coprocessor outputs are byte-identical (parity is a standing gate).
- **Fixed-data reasoning screens** (exact-history teacher targets from Recur64's own
  search; fresh, source-game-disjoint confirmation sets): extra recurrent thoughts have
  **not** beaten a one-pass network trained the same way, on teacher-KL or on tactics,
  in any pre-registered comparison so far (E7, E8, E11, E11b). Result: latent reasoning =
  no signal at this scale. Not proof that it cannot help; the data is small.
- **Candidate facts** (exact one-ply facts per legal move, D62) take mate-in-1 from
  chance to about 95% with about 50 s of training, once the channel is given a usable
  scale (gain 128; at gain 1 it learned almost nothing, which was measured and diagnosed).
- **Not established:** any playing-strength or conversion gain, any benefit of the
  visual or compute pathways, anything about problems that need multi-ply lookahead
  (the natural next test: mate-in-2 and multi-step captures). X15 is not wired into the
  pilot or self-play. The historical F15/R15 model is untouched and old checkpoints are
  refused by the X15 loader (and vice versa; X15 head v1 checkpoints are refused too).
