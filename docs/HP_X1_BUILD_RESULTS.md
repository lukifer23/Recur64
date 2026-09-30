# Recur64 HP — X15 / "Chimera" build and resource results

Branch: `experiment/hp-r15-h3-integration`. Machine: HP home box — **RTX 2050
4 GB**, Ryzen 5 7535HS (6C/12T), 32 GB RAM, user-space CUDA 12.9.1
(`%LOCALAPPDATA%\Recur64\cuda\12.9.1`), **no `nvcc` on PATH**.

Every statement below is labelled. Nothing here is presented as validated on a
device it did not run on.

---

## MEASURED

### X0.1 — fast dev/test profile

| | value |
|---|---|
| Before (baseline, `[profile.dev.package."*"]` absent) | the full workspace gate was started; **after 1452 s it was still inside one test** (`arena_crn::paired_common_rng_mirrors_identical_evaluators`) and was stopped |
| After (deps at `opt-level = 2`) | that same test: **64.42 s, passes** |
| After | full `recur64-runtime` suite (63 tests, 2 filtered): completes in ~2 minutes |

The "before" figure is a lower bound, not a measurement of a completed run: the
run never finished. The port is kept because it turns a gate that could not
complete into one that does, with no test-semantics change (workspace crates
keep `opt-level = 0`, debug assertions and overflow checks).

### X0.2 — CUDA build pin + behavioural device check

- `.cargo/config.toml` pins `CUDARC_CUDA_VERSION = 12090`, matching this
  machine's user-space CUDA 12.9.1 and absent `nvcc`.
- `model_io::verify_device` runs an elementwise/reduction/matmul known-answer
  check on a helper thread with a 120 s timeout before every build/load, and
  refuses on a panic, hang or wrong value.
- Unit test `model_io::tests::device_check_passes_on_a_working_backend`: passes
  on the CPU autodiff backend.
- The CUDA path itself is **NOT RUN** here (see below).

### X0.3 — multi-owner inference pool

- `InferenceOwner::spawn_pool` + `inference_owners` (default `1`, execution
  only, never in the scientific identity) + `pilot::spawn_selfplay_owner`.
- Test `owner_pool_answers_every_request_and_every_owner_serves`: 64 concurrent
  requests across a 4-model pool — every request answered exactly once, 0
  errors, `owners == 4`, per-batch cap respected, **≥ 2 distinct owners served
  a batch** (real multi-owner service, not assumed), and shutdown joins cleanly.
  A pool of 1 still behaves exactly like the single owner.
- **Throughput on this stack is NOT measured** (no GPU run). Per the brief it
  stays available and defaults to 1 until measured.

### X0.4 — flattened rank-3 linears

- `linear_rows` now backs the input projection, Q/K/V/out, both FFNs, the
  policy source/destination projections and the promotion MLP.
- Unit test `linear_rows_matches_rank3_linear`: max |difference| ≤ 1e-5 against
  Burn's rank-3 form.
- Every pre-existing probe/R15 test still passes unchanged.
- **The +9.4 % / +20 % mainline throughput claims are NOT re-measured on this
  stack** (no GPU run).

### X1 — X15 architecture

```
recur64 x15 info --config configs/x15.toml
```

| subsystem | parameters |
|---|---|
| symbolic (input_proj + square_emb + prelude) | 3,777,041 |
| recurrent square core (shared) | 7,364,640 |
| output blocks | 3,682,320 |
| reasoning latents + bus | 198,275 |
| compute embedding + xattn | 69,760 |
| visual CNN + xattn | 430,560 |
| square xattn | 164,992 |
| heads | 331,018 |
| **TOTAL** | **16,018,606** |

- 61.1 MiB of FP32 parameters.
- Trunk geometry is R15's (width 512, 8 heads, ffn 768, 2/4/2), so the
  comparisons against the 15,154,632-parameter F15/R15 family are direct; the
  new subsystems are additive and reported separately.
- Parameter count is **identical at T = 1, 2, 4, 8** (asserted).
- Executed blocks: T=1 → 8, T=2 → 12, T=4 → 20 (the same accounting as R15 at
  R=1/2/4).

X15 is inside the owner's 15–20 M target and needs no change to the trunk to
get there.

### X1 — forward sanity (CPU Flex, batch 4, 3.1 s wall)

```
T=1  readouts=1 thoughts=1 executed_blocks=8   mean_policy_entropy=3.3570 max|WDL logit|=0.00e0
T=2  readouts=1 thoughts=1 executed_blocks=12  mean_policy_entropy=3.3601 max|WDL logit|=0.00e0
T=4  readouts=1 thoughts=1 executed_blocks=20  mean_policy_entropy=3.3604 max|WDL logit|=0.00e0
PASS: forward sanity
```

- finite at every T; legal policy mass exactly 1 after masking (padded entries
  are exactly 0 log-probability by design);
- WDL is exactly neutral at initialization (both WDL terms are zero-initialized);
- parameter count unchanged across T.

### X1 — module gradient probe (CPU, T=4, batch 4, 3 s wall)

| subsystem | grad norm |
|---|---|
| symbolic | 1.942e-2 |
| recurrent_core | 1.561e-2 |
| output_blocks | 2.961e-3 |
| reasoning_latents | 1.021e-2 |
| **compute** | **8.516e-4** |
| **visual** | **1.784e-3** |
| heads | 3.717e+1 |
| **gates** | **1.075e-4** |

Every gated subsystem receives a non-zero, finite gradient on the first step —
including compute, visual and the three learned gates. `PASS: every gated
subsystem received gradient`.

### X1.3 — deterministic coprocessor, native vs WebAssembly

```
recur64 x15 parity --positions 200
PASS: native == wasm on 253 positions (1728 bytes each)
  native 68 ms (272 us/position), wasm 1099 ms (4346 us/position)
  wasm overhead vs native: 15.97x
  compute_bank_version    : compute_bank_v1
```

- **Byte-for-byte equality** on 253 positions: the fixed special-move/tactical
  set (castling, en passant, promotion and capture promotion, check, double
  check, pins, mate-in-1, mate-in-2, stalemate, checkmate, insufficient
  material, clock features) plus 200 random legal playouts at mate depth 1 and
  40 more at depth 2.
- The equality is by construction: `NativeV1` calls `recur64-coproc`'s
  `compute_bank` directly and `WasmV1` calls the *same source* compiled to
  `wasm32-unknown-unknown`, interpreted by `wasmi`.
- The WASM artifact is committed and digest-pinned:

| | value |
|---|---|
| path | `crates/recur64-compute/assets/compute_bank_v1.wasm` |
| bytes | 1,577,622 |
| sha256 | `a405d873386675cdfc90b8acc264b2a1a32cf84d3cdfbb28cd6958e6afab1063` |
| built by | `scripts/build-compute-wasm.ps1` (profile `wasm-release`) |

An embedded-artifact test recomputes the digest, so a rebuilt-but-unpinned
module fails a test rather than silently changing what the model sees.

**WASM is ~16× slower than native** (4.35 ms vs 0.27 ms per position). That is
the honest cost of the WebAssembly experiment: `native_v1` is the practical
training-time provider, `wasm_v1` is the real-WASM implementation, and the two
are provably identical in semantics. Provider labels are recorded in the
identity today; once this parity evidence is accepted, demoting native-vs-WASM
to execution-only (as D55 did for fusion/autotune) would be a legitimate ADR.

### X1 — per-stage phase times (CPU, batch 4, mate depth 1, 64x64 render)

| stage | total | per position |
|---|---|---|
| observation | 411 µs | 0.10 ms |
| compute (WASM `wasm_v1`) | 6,793 µs | 1.70 ms |
| visual render | 4,279 µs | 1.07 ms |
| upload | (now measured) | — |

On this CPU both the coprocessor and the renderer are **materially more
expensive than the observation encoding**, which is exactly why they are timed
separately instead of being hidden inside "forward". On the GPU the symbolic
forward dominates; the host-side costs stay as listed.

### Test evidence

| suite | result |
|---|---|
| `recur64-coproc` | 23 passed, 0 failed |
| `recur64-compute` (incl. native==wasm parity) | 6 passed, 0 failed |
| `recur64-coproc-guest` | 3 passed, 0 failed |
| `recur64-runtime` (incl. 9 new X15 integration tests) | all pass, 2 filtered |
| `recur64-model` / `recur64-core` | all pass |

The two filtered tests are the pre-existing arena tests that take ~60 s each;
both **pass** when run (`paired_common_rng_mirrors_identical_evaluators`, 64.4 s).

## INFERRED (not measured)

- **VRAM.** 61.1 MiB of parameters is exact. Activation memory is not measured
  for X15. R15's measured CPU-batch-32 learner peak was 1,953 MiB of 4,096 on
  this card, and X15 at T=4 executes 2.5× the square-core blocks of T=1 while
  adding a 64x64 RGB visual CNN. **The owner's `≤ 3.2 GB at effective batch
  128` ceiling is therefore UNVERIFIED.** A GPU training-step measurement is
  the first prerequisite of any X15 training run.
- The GPU will make the coprocessor and renderer comparatively cheap and the
  symbolic trunk expensive; that ordering is inferred from the R15 profile,
  not measured for X15.

## NOT RUN

- **No CUDA build or CUDA execution of X15.** `recur64 x15 ...` is
  feature-gated but has only run on `Flex` (CPU). No `--features cuda` X15
  forward/backward has happened, so no GPU claim is made.
- **No training run of any kind.** No optimizer step beyond the unit test's
  single-step smoke, no self-play, no pilot.
- **`ReasoningTargetsV1` is designed, not generated.** The schema and its
  storage decision (positions as `(start_fen, prefix moves)` so repetition
  history survives) are documented in `HP_X1_ARCHITECTURE.md`; the generator and
  the one-command reasoning-target probe are the experiment agent's first task.
- **Tactical fixture suite and conversion-pathology suite: not built.** The
  conversion diagnostics are feasible from committed evidence
  (`docs/evidence/train1/final-arena/eval-arena.json` carries `final_fen` for
  every arena game, including the 111 threefold draws), but the suite is not
  implemented. Note the limitation that a committed `final_fen` has no
  repetition history, so `ObservationV1`'s repetition feature is not
  recoverable from it.
- **`inference_owners = 2` throughput on this stack: not measured.**
- **X15 is not wired into the pilot/self-play loop** (an explicit owner
  decision): `RunConfig` carries no X15 run path, so there is no `x15` *run*
  config, only the probe config. Batched inference and the probe harness are
  the integration surface.
- Peak VRAM, GPU forward latency at T1/T2/T4, and GPU WASM overhead: all
  unmeasured.

## Ports kept / rejected

**Kept:** `[profile.dev.package."*"] opt-level = 2`; `.cargo/config.toml`
CUDA-12.9 pin; `model_io::verify_device`; `InferenceOwner::spawn_pool` +
`inference_owners` (default 1); `linear_rows` for every rank-3 linear.

**Rejected (deliberately, with reasons):** mainline's 48/96 scheduling (this
branch has its own measured c12/b24 schedule); the D50 solver decisions (the
solver did not solve conversion); the D51 curriculum as a training recipe (it
failed its gate); TF32 as active precision (never achieved); T6 candidate
buckets.

## What the experiment-running agent must know

1. **Measure VRAM before anything else.** `x15` has never touched the GPU.
2. `wasm_v1` is ~16× native; use `native_v1` for throughput and keep `wasm_v1`
   for the parity/demonstration runs.
3. `final_only_v1` reads out once but still executes all T thoughts; use
   `same_target_v1`/`progressive_search_v1` if intermediate thoughts should
   carry loss.
4. `mate_depth = 2` is an exhaustive exact search and is diagnostic-only.
5. The visual encoder needs `resolution / 8` to be a power of two (64 or 128);
   the renderer supports 96 but the encoder does not.
6. `ComputeBankV1` derives the halfmove clock and repetition count from the
   normalized observation features, so they are lossy beyond the observation's
   own clamps (150, 5).

## P0 corrections (2026-09-29, after c60103b)

MEASURED by tests on CPU unless stated. See D60 and D61.

| Item | Result |
|---|---|
| Symbolic-only control | output bit-identical after scrambling every auxiliary parameter |
| Diagnostic readouts | T=4 under `final_only_v1`: 1 normal readout, 4 diagnostic readouts and 4 metric rows; final output identical |
| ComputeBank hanging bug | confirmed and fixed; WASM rebuilt; native/WASM parity holds |
| CUDA device check | X15 builds/loads go through `model_io::build_chimera` (counter-tested) |
| Visual resolution | only 64 accepted; config->render->encode test |
| Flattened linears | cross-attention Q/K/V/out and feedback use `linear_rows` |
| Deep supervision | loss API with mapping unit tests |
| Parameters | 16,018,606, unchanged |

## First CUDA resource results (2026-09-29)
MEASURED on RTX 2050 4 GB, FP32, `native_v1`: see `docs/HP_X1_EXPERIMENTS.md`
E1/E2. Normal forward batch 32: 91.5 / 108 / 142.5 ms at T=1/2/4, 517 MiB.
Training 32x4 (eff 128) at T=4: 1.47 s/update, 87 ex/s, 2.38 GB peak. New
command `recur64 x15 bench --mode infer|train`. In-process GPU telemetry is
printed by every CUDA `x15` command.

## Update 2026-09-29 (scaled runs, MEASURED)
- Training on `targets-train960` (T=4, 96 positions per update as 3 x 32 accumulated):
  100 updates in 136-144 s, 2.44 GB peak, GPU 86-89% busy; T=1 in 86-103 s (1.51 GB);
  symbolic-only in 47-51 s (1.35 GB).
- Teacher labelling through the shared inference owner: 2.6 positions/s (960 positions
  in 6 min 8 s, 152,135 evaluations, 0 errors), about 5x the batch-1 path with
  identical labels; owner batches stayed at 16 (tuning knob).
- Head v2 (`candidate_facts`) adds 8x16+16 and 16x1+1 parameters; the probe head is
  unchanged. Old X15 head-v1 checkpoints are refused.
