# Recur64

## Current V6 diagnosis: root/structure correction dominates; owner review

All four frozen P0 endpoints reproduced exactly on RTX2050 CUDA FP32. Removing
returned boards and flags preserves 11/12 principal and 4/4 one-pass helpful
paired roots. Broad acquisition mainly adds negative corrections to candidates
outside B0's top two; it adds no minimum-mate-positive branch on this panel.
These are repeatedly exposed TRAIN diagnostics, not held-out improvement or
proof of recurrent superiority. Original P0 failure and V5 NO_SIGNAL stand.

[Diagnostic findings and limitations](docs/V6_P0_FAILURE_DIAGNOSIS.md) and
[one prospective objective/content-contrast probe](docs/V6_P0_NEXT_OBJECTIVE_EXPERIMENT.md).
The proposal is pending owner review; it has not been implemented or fitted.
The diagnostic pass took no optimizer steps and invoked no DEV/CONFIRM model.
CONFIRM remains sealed and unevaluated. Current diagnostic source `d34c948` is
separate from frozen P0 producer `42f47b8` and this documentation publication.

## Current V6 branch: P0 complete; both readers failed learnability

Actual CPU/RTX2050 qualification passed. Both fixed 200-update competent-base
TRAIN experiments completed, but neither passed the frozen content-sensitive
learnability gates. The scientific campaign is NOT justified or run.
[Measured P0 results and stop decision](docs/V6_P0_RESULTS.md).

This branch `experiment/hp-v6-branch-backup` implements the revised full-information
branch reader and one-pass comparator. Its authorized endpoint is qualification
and two disposable competent-base TRAIN panels, not held-out training or DEV
inference. [Corrected frozen contract](docs/V6_P0_REVISED_CONTRACT.md) supersedes
the historical implementation ticket. V5 remains closed as NO_SIGNAL.

## Historical closure publication: V5 closed, V6 design proposed

The frozen V5 pilot is **NO_SIGNAL**: all reader conditions left DEV actions
unchanged. Its model producer (`3db24a9`), evaluator (`7fb7461`) and reviewed result
publication (`3efce66`) are distinct. Stage A/B weights remain immutable.
The closure investigation reconciles primary750 versus pooled4500 statistics and
measures sparse error-branch information and ineffective corrections over trained B0.

Read [closure](docs/V5_CLOSURE.md), [diagnosis](docs/V6_DIAGNOSTICS.md),
[prospective V6 design](docs/V6_EXPERIMENT_PROPOSAL.md), and
[next implementation ticket](docs/V6_IMPLEMENTATION_PROMPT.md).
At that historical closure publication V6 was not implemented or trained.
CONFIRM remains sealed/unevaluated.
The older branch/Phase4/V5 handoffs below are historical context, not instructions
for continuing the closed V5 run. Current operational entry point:
[HP resume/status](docs/V5_RESUME.md).

A Rust-first chess-learning research laboratory. The eventual question: at
matched end-to-end compute, does a small **recurrent** square-token transformer
that spends compute on internal refinement beat spending the same compute on
external search?

**Status (Phase 4, in progress).** The full learning loop exists and runs
on the RTX 2000 Ada workstation: PUCT self-play â†’ batched GPU inference â†’
replay â†’ audit â†’ train â†’ checkpoint â†’ searched and raw arenas â†’ report.
Phase 4 is producing the first trustworthy mainline F10 evidence on it. See
`docs/STATUS.md` for the gate record and `docs/PHASE4_RESULTS.md` for measured
Phase 4 results.

| phase | scope | gate |
|---|---|---|
| 0 | systems probe: model-shaped graph trains on this machine (CPU + CUDA FP32) | GO |
| 1 | chess contracts: observation V1, action V1, rules profile, perft, oracle | GO |
| 2 | first vertical slice (Micro model) | GO |
| 3 | F10 + PUCT control baseline, bounded pilots | CONDITIONAL GO (historical; predates the Phase 4 fixes) |
| 4 | mainline harness convergence + GPU requalification | P4.2â€“P4.5 done; smoke v2 GO (learning mechanism); P4.6 stopped on draw drift; root cause found; D51 curriculum NO-GO; next fix pending owner decision |

**Branch `experiment/workstation-v25` (Workstation V2.5).** A separate research line
that asks whether a stronger ONE-PASS model can learn conversion technique from exact
proof data. Main (this README's Phase 4 status) is unchanged and remains the control line.
Result, in one paragraph: exact CandidateFacts solve mate-in-one and help deeper mates
(especially through the candidate-token architecture); candidate tokens alone do not beat
the matched legacy head; five times more unique exact data closed the train/held-out gap
without raising held-out accuracy; the pre-registered M2 gate (0.75) was not met (best
~0.69), so P3 (conversion) is not authorized and the lineage stops for V3 design. Start at
`docs/WORKSTATION_V25_SUMMARY.md`; every rule was pre-registered in
`docs/WORKSTATION_V25_EXPERIMENTS.md` and `docs/WORKSTATION_V25_P25_PLAN.md`, and the
machine-readable evidence is in `docs/evidence/v25/`.

**Branch `experiment/workstation-v4-evidence-belief` (Recur64 V4 P0/P1, current state).** A new architecture line, `evidence_belief_v4` (30.0M parameters): an immutable base belief over root-action hypotheses, explicit content-causal evidence messages (exactly zero for zero content) and a learned query-utility head. Built with 20 architectural invariants, a CUDA smoke and a sealed `v4_tune_v1` (6,000 positions, never evaluated). TRAIN-only Stage B result: **ARCHITECTURE-STOP** - zero content is bitwise exact (C passes) but the trained evidence path does not change decisions (G: top-1 change 0.0; A: +2e-6 nats; B fails). Stage C and V4 final science were NOT run; HOLDOUT_C remains sealed and unevaluated. See `docs/V4_P1_RESULTS.md`, `docs/V4_RESEARCH_PLAN.md`, `docs/evidence/v4/`.

**Branch `experiment/hp-v5-counterfactual-loop` (Recur64 V5, current line).**
V5 retains counterfactual_relational_loop_v1: **7,162,896 parameters**, FP32,
physical microbatch2, configuration
`849133a5cdf169f187778bace2f858aa4747d2e8defef3bb5ac1bffc839774ee`.
The returned-payload mask repair/model math at003d296 remains unchanged.
Current scientific consumer source **d11659e** passes fresh CPU/RTX2050 CUDA
qualification under unchanged exact D9, all nine Q/R shapes,50 resident Q8/R4
updates, complete checkpoint/moment/resume and12 repeated exact normal/profile
comparisons. Historical d970049 CUDA D9 failure remains preserved.
Full release workspace: **596 passed, zero failed, two preserved ignores**;
V5 CUDA Clippy/changed-file rustfmt PASS.16 untouched formatting failures remain.

The owner-amended heavy-only native **V2** lineage is complete: TRAIN27000
(nine cells3000), DEV4500 and sealed CONFIRM4500 (six cells750 each),72,000
original/regeneration independent audits with zero failures. All three complete
regenerations are byte-identical. FEN/canonical overlaps are zero. Actual local
custody and source-bound graph provenance PASS. The infeasible light-family V1
plan remains historical; P25 is retired as a V5 dependency. Active data contract
is v5_hp_data_v2; scientific recipe is v5_stage_recipe_v3. TRAIN/DEV roles are
separate, and ordinary commands refuse CONFIRM.

The conditionally authorized24-position reader drill **PASS** at Q8/R4,200
updates/LR1e-3: mean correct-set loss2.987107 ->0.026625 (**99.11% reduction**),
finite training, baseline exact, disposable weights. Q16 was not run. This is
engineering evidence; Stage A, Stage B, DEV model evaluation, reader pilot,
controller and multi-seed replication remain unrun. CONFIRM remains sealed and
unevaluated; V4_TUNE_V1/HOLDOUT_C remain unevaluated. **STOP before Stage A.**
See [complete V2 handoff](docs/V5_DATA_V2_REPORT.md),
[preregistration](docs/V5_DATA_V2_PLAN.md) and
[measured evidence](docs/evidence/v5/data/v2/).

**Branch `experiment/workstation-v35-onpolicy` (Recur64 V3.5, current state).** One on-policy rescue of the V3 architecture: the learner selects every query and the proof oracle only labels the learner-visited states (never chooses). Three seeds, 800 updates each, evaluated once on V3_TUNE_V1 KQRvK M3: the model now uses queried-state content (Content-Use PASS, +0.037 nats, every seed), but Gate II (ACTIVE B8 - B0 = -0.022) and Gate III (ACTIVE B8 - FIXED B8 = -0.023) FAIL. Outcome **PARTIAL - CONTENT**; HOLDOUT_C remains sealed and unevaluated; B16 not run; the lineage stops and a V4 design memo is in `docs/V35_RESULTS.md`. Plan: `docs/V35_RESEARCH_PLAN.md`; evidence: `docs/evidence/v35/`. The V3 branch text below is unchanged history.

**Branch `experiment/workstation-v3-active-search` (Recur64 V3, current state).** A new research line
from `experiment/workstation-v25`, asking one narrow question: does the SAME set of weights improve as
its exact state-query budget grows (B0, 2, 4, 8, 16), because it learns which unresolved future states
to inspect and integrates what comes back? V2.5 showed that one more pass over the same information
does not help; V3 therefore never re-reads the same state. The model runs the V2.5 root encoder once,
then spends a budget of exact single-edge state queries chosen by a learned selector and integrated by
a shared gated planner. Status: the engineering build (P0 to P3) is complete and qualified on CPU and
real FP32 CUDA. P4 has since built the exact proof-trace data layer (`proof_trace_v1`, `v3_tune_v1`) and taken the
frozen feasibility measurement: `C_8(KQRvK M3) = 0.4336` against the pre-registered threshold 0.25, so the primary
B8 experiment is **scientifically qualified** (this says nothing yet about learning). **P5 and P6 are complete.** The P5 learning-rate screen selected peak LR 3.0e-4 (TUNE recipe selection only; ACTIVE does not yet
beat B0 in policy CE at B2/B4/B8). **P6 Gate I passed**: the separately trained, parameter-matched ALL-INFO control (exhaustive raw
depth-2 tree, no answer information) improves KQRvK M3 top-1 over B0 by +0.2351 (paired 95% CI [0.2044, 0.2671], threshold +0.20;
0.692 vs 0.457), so raw future-state information is sufficient for this model family on V3_TUNE_V1 - a narrow claim that does not
establish learned selective search. A query-content ablation shows the P5 teacher-forced accuracy comes from which edges were
queried (query-pattern leakage), not from state contents (`docs/V3_P6_RESULTS.md`). HOLDOUT_C is sealed and unevaluated; the plans are `docs/V3_P5_PLAN.md` and `docs/V3_P6_PLAN.md`; DAgger, P7, Gate II and Gate III have not been run and await owner approval. Main and V2.5 are unchanged. Start at `docs/V3_RESEARCH_PLAN.md` (every
gate is pre-registered before any data exists), then `docs/V3_ARCHITECTURE.md`,
`docs/V3_BUILD_RESULTS.md` (measured engineering results and suggestions), `docs/V3_P4_PLAN.md` and
`docs/V3_P4_RESULTS.md` (P4), and `docs/V3_EXPERIMENTS.md` (append-only ledger). Machine-readable evidence is in `docs/evidence/v3/`.

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
- **P4.6 qualification, stopped after 6 cycles:**
  - 3 promotions; 0.578 and then 0.594 against the frozen reference.
  - Self-play drifted into draws.
- **Root cause (MEASURED): there is no conversion signal at this scale.**
  - Most draws are won endgames that were never converted.
  - The value head learns "big lead = draw".
  - Neither the trained nor the untrained network can convert K+Q or K+R
    vs K: 1-2 of 64.
  - Deeper search (128 sims) and an MCTS-solver (D50, not adopted) do not
    help.
- **Endgame curriculum (D51): NO-GO** in its stage 2 pilot.
  - Too few curriculum positions, and most of them drawn.
  - The trained network converted endgames *worse* than the untrained one.
  - Next steps (a reverse curriculum from near-mate positions,
    endgame-appropriate exploration, an LR A/B) await an owner decision.
- **Value-head diagnostics (partial):**
  - The value head is *calibrated* to its data and learns and unlearns
    within about 50 updates.
  - The real bottleneck is that self-play almost never converts a lead, so
    the next fixes target the data: a reverse curriculum of mostly won
    positions, and conversion technique.
  - See `docs/STATUS.md`, "CURRENT STATE AND NEXT STEPS". Work is paused
    pending an owner decision.
- **Arena fix (D54, from the HP branch):** players now search their own
  trees. Earlier arenas mixed both networks.
- **Throughput pass (D53):** 1.38x self-play throughput and 1.20x training,
  all bit-exact or execution-only:
  - two inference owner threads
  - a 48 / 96 schedule
  - flattened linear layers
- **Also not adopted:** Burn fusion, autotune and TF32, all measured and
  documented.

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

### Workstation V2.5 (branch `experiment/workstation-v25`)

```sh
# Exact proof datasets (ProofTargetsV1): pool sizes, generation, independent audit.
recur64 proof pool --output runs/v25/proof --families KQvK,KRvK,KQQvK,KQRvK,KRRvK
recur64 proof gen --output runs/v25/proof-v2 --seed-train 2048000001 --seed-tune 2048000002 --seed-confirm 2048000003
recur64 proof audit --dir runs/v25/proof-v2
# P2.5: heavy-family holdouts, the scaled training set, LF/CF/C0/L comparisons.
recur64 proof p25-holdouts --output runs/v25/p25/holdouts --exclude-dirs runs/v25/proof,runs/v25/proof-v2
recur64 proof p25-data --base runs/v25/proof-v2 --exclude-dirs runs/v25/proof,runs/v25/proof-v2,runs/v25/p25/holdouts --output runs/v25/p25/data
# Policy-only exact-target training and evaluation (configs/v25/*.toml, FP32).
recur64 proof train --config configs/v25/candidate-v25-cf-cuda.toml --data runs/v25/proof-v2 --output runs/v25/p2/CF-s1 --seed 1 --lr 3e-4 --updates 400
recur64 proof eval --config configs/v25/candidate-v25-cf-cuda.toml --checkpoint runs/v25/p2/CF-s1/checkpoint --data runs/v25/p25/holdouts --split holdout_a --output eval.json
recur64 proof compare --a a1.json,a2.json --b b1.json,b2.json --per-seed --output compare.json
recur64 proof interaction --c0 ... --cf ... --l ... --lf ... --output interaction.json
# Facts cost, forward latency, learner layouts and VRAM on the real graph.
recur64 v25-qual --config configs/v25/candidate-v25-cf-cuda.toml --output runs/v25/qual
recur64 model-info --config configs/v25/legacy-facts.toml
```

### Recur64 V3 (branch `experiment/workstation-v3-active-search`)

```sh
# Describe the active-search graph: geometry, 11 contracts, exact parameters, budgets.
recur64 model-info --config configs/v3/active-search-v3-cuda.toml
# Engineering qualification on the real graph with the live state-query tool (FP32).
# Budgets above 16 are refused unless --engineering-stress (then the report is engineering only).
recur64 v3-qual --config configs/v3/active-search-v3-cuda.toml --output runs/v3/cuda-qual \
    --budgets 0,2,4,8,16 --batches 1,8,16 --train-budgets 2,4,8
# P4: verify custody, build and audit v3_tune_v1, trace and audit P25_DATA_V1 TRAIN, take the frozen measurement.
recur64 v3-p4 custody --train runs/v25/p25/data/proof-train.json --holdout-c runs/v25/p25/holdouts/proof-holdout_c.json \
    --inventory-dirs runs/v25/proof,runs/v25/proof-v2,runs/v25/p25/data,runs/v25/p25/holdouts --output c.json --seal-output seal.json
recur64 v3-p4 trace-gen --data runs/v25/p25/data/proof-train.json --dir runs/v3/p4/trace-train
recur64 v3-p4 trace-audit --data runs/v25/p25/data/proof-train.json --dir runs/v3/p4/trace-train
recur64 v3-p4 feasibility --data runs/v25/p25/data/proof-train.json --dir runs/v3/p4/trace-train --role primary --output f.json
# Exact state-query tool tests (differential against GameState, packet whitelist, digests).
cargo test -p recur64-statequery --release
cargo test -p recur64-model --release --test active_v3
cargo test -p recur64-cli --release --test active_boundary
```

Every historical command (`bench`, `cuda-smoke`, `v25-qual`, `proof ...`, the self-play and
training runtime) refuses `active_search_v3` with a visible error before any work.

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
configs/               historical Phase 0â€“3 configs; configs/phase4/ (current
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
- The self-play â†’ learning loop closes truthfully: audited replay, provenance,
  identity hashes, and refusal of mismatched checkpoints.
- The main-workstation schedule is measured.
- A fresh network now starts from a near-uniform policy and a neutral value.

**Not** established:

- any chess strength
- that F10 gains playing strength at this scale. Smoke v2 shows the
  learning mechanism working (value learning, promotions, value-guided
  search), but 0.500 against the untrained reference.
- that recurrence helps (the R10 R1/R2/R4 experiments have not started)
