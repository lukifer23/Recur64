# HP H3 measured results — mainline science transfer + F15-v2 requalification

Status legend: **HISTORICAL** (old contract, preserved), **MEASURED** (ran on
this machine), **INFERRED** (mechanism, not directly measured here), **NOT RUN**.

Branch: `experiment/hp-r15-h3-integration`. Integration merge: `a926fa4`
(mainline `03e62f7` merged into HP `3430e6ca`; merge base `5ac291c`).
Frozen reference: `runs/hp-h3-ref-f15-v2`,
`model_id d89b408fcc7a3cd9874a0ffbbcafd121c6c78cbb9818d4adaac0fe8d51ea234b`.
Rules were fixed in [`HP_H3_PREREG.md`](HP_H3_PREREG.md) before measurement.

> Purpose: decide whether the historical HP F15 blockers were intrinsic to
> F15 / the RTX 2050 / recurrence, or consequences of head-v1 initialization,
> exploration, learner continuity, the update cap, and the CubeCL owner
> lifecycle. This document is evidence, not assumption.

## H3.0 — Integration (MEASURED)

- `git merge origin/main` produced **17 conflicts**, all generic core; resolved
  to mainline (main is the newer descendant of the shared harness).
- HP-only assets survived untouched: `configs/f15.toml`, `configs/r15.toml`,
  `configs/hp/*`, `configs/hardware/hp-home.toml`, `f15_r15_parity.rs`,
  `docs/HP_*`, `Grok-plan.md`, `scripts/hp-*`.
- Zero-science gate: `cargo fmt --check`, `cargo clippy --workspace
  --all-targets`, `cargo test --workspace --release` and the CUDA feature
  build/`check` all pass. One flaky observation: `tests/inference.rs::
  errors_propagate_to_every_request` failed **once** under heavy parallel load
  (the merged file is byte-identical to mainline; the test then passed 5/5 in
  isolation and in full-suite reruns). Recorded, not hidden.

## H3.1 — F15/R15 head-v2 contract (MEASURED)

`model-info` on both configs:

| | F15 | R15 |
|---|---|---|
| geometry | 512/8/768 | 512/8/768 |
| blocks | 0 + 8 + 0 | 2 + 4 + 2 |
| unique params | **15,154,632** | **15,154,632** |
| executed blocks | R1 = 8 | R1 = 8, R2 = 12, R4 = 20 |

The final pre-head RMSNorm (`final_norm`, 512 params) is the +512 delta from the
head-v1 count of 15,154,120. F15 and R15 remain **exactly matched** in unique
parameters. Contract tests now pin 15,154,632, assert `HEAD_VERSION == 2`, and
refuse head-v1 / pre-field (legacy) checkpoints.

- **Old head-v1 reference `4271e19f…`: refuse under v2 — MEASURED** at the
  metadata contract level (head-version and legacy-field tests pass). The old
  checkpoint was not modified or deleted.

### T0 sanity (MEASURED, no strength claim)

- CPU Flex, fresh model, 3 seeds x (openings-v1 + startpos), 39 positions:
  - F15 R1: entropy/uniform **0.999**, mean |value| **0.000**.
  - R15 R1/R2/R4: entropy/uniform **1.000 / 1.000 / 1.000**, mean |value|
    **0.000**. Head-v2 sanity holds across the recurrence ladder.
- Frozen CUDA reference, `eval-policy`, 40 raw games:
  entropy/uniform **0.999**, mean top-1 prob **0.043**, raw-vs-random score
  **0.567** (5 W / 1 L / 24 D; small, draw-heavy sample — not a strength claim).

### Reference reproducibility (CORRECTED in H3.5B; see D50)

**H3.5B correction (MEASURED, `docs/evidence/hp-h3/init-repro/`):**
- The different `model_id`s below are **artifact** differences. The `.mpk`
  record stores a random, OS-entropy `ParamId` per tensor.
- The weight values are **reproducible**. Ten separate-process freezes of the
  reference config gave 10 distinct `model_id`s but only 2
  `semantic_weights_digest`s, one per backend:
  - CPU: `81b02bcd…`
  - CUDA: `f81938a2…`
- 0 of 15,154,632 elements differ within a backend.
- The frozen reference `d89b408f…` is semantically identical to fresh CUDA
  freezes.
- CPU and CUDA initializations differ (different backend RNGs); that is
  expected.
- `model_id` remains the pinned artifact identity.

Original H3.1 text, whose inference is superseded:

> Freezing the same config with the same seed and the same binary produced
> **different `model_id`s on every run** (CUDA `247e03…`, `b23597…`, `d89b408…`;
> CPU `ded3b7…`, `42a7d9…`). Byte comparison of two CPU `model.mpk` files: same
> size (60,628,164 B), **1,699 differing bytes scattered across the whole file**.
> So initialisation on this stack is *nearly* deterministic but a small subset of
> parameters varies between processes. **Consequence:** the frozen reference is an
> opaque, content-addressed, single-sample artifact. Runs must pin
> `reference_model_id`; it is not reproducible from config+seed. This is a
> systems fact about the stack, not an F15 property.

## H3.2 — RTX 2050 GPU lifecycle requalification (MEASURED — GO)

`bench-lifecycle`, `one`/`two`/`pilot`/`pilot-promoted`, reps 8, 16 sims,
concurrency 16, batch cap 16, timeout 1000 µs, frozen v2 reference.

| mode | VRAM before/peak/after (MB) | fwd latency | err | owners |
|---|---|---|---|---|
| one | 9 / **298** / 298 | ~12 ms | 0 | 1/1 |
| two | 298 / **490–492** / 490–492 | ~15 ms | 0 | 2/2 |
| pilot | 492 / **492** / 492 | ~15 ms | 0 | 2/2 |
| pilot-promoted | 492 / **492–496** / 492–496 | ~15 ms | 0 | 3 spawned / **2** resident |

- VRAM **plateaus**, no monotonic growth across 32 owner lifecycles; peak
  **496 MB** vs the historical **3,909 / 4,096 MiB**.
- Zero inference errors; forward latency flat after warmup; temperature 66–70 °C.
- `max_resident_owners = 2` in pilot/pilot-promoted → **D46 validated**.

**GO.** The D44 owner-memory fix holds on the HP machine. This directly answers
the third historical blocker: the late-run VRAM exhaustion and latency collapse
were the CubeCL owner lifecycle defect, not an intrinsic F15/RTX-2050 limit.

## H3.3 — HP performance requalification (MEASURED)

Frozen v2 reference, 64 sims, concurrency 16 unless stated, 32 games/cell.

### H3.3A multi-leaf K

| K | batch cap | trainable pos/s | vs K=1 | errors | decisive | 3fold+fifty | target entropy | VRAM |
|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 6.92 | — | 0 | 0.781 | 0.0625 | 2.996 | 298 MB |
| **2** | 32 | **7.83** | **+13.2%** | 0 | 0.781 | 0.0625 | 3.004 | 298 MB |
| 4 | 64 | 6.58 | −4.9% | 0 | 0.719 | ~0.03 | 2.930 | 586 MB |

**Adopted K = 2** — the smallest K > 1 meeting every pre-registered criterion
(≥10% throughput, 0 errors, decisive within 0.10, 3fold+fifty within 0.05,
entropy within 10%, VRAM margin). K = 4 fails the throughput bar. This confirms
mainline's D47 K = 2 choice **by independent HP measurement**, and confirms D47
was not a mainline-hardware artefact.

### H3.3B concurrency (K = 2)

| concurrency | batch cap | trainable pos/s |
|---|---|---|
| **8** | 16 | 7.5 |
| 12 | 24 | 7.5 |
| 16 | 32 | 7.83 |

All within **4.4%**. The pre-registered rule (<5% → prefer lower) selects
**concurrency 8, batch cap 16**. No queue pathology at any value; 12 logical CPU
threads were not oversubscribed.

### H3.3C training physical batch (effective 128)

| layout | ex/s | step | VRAM peak |
|---|---|---|---|
| **32x4** | 144.3 | 887 ms | 1290 MB |
| 64x2 | 151.8 | 843 ms | 1962 MB |
| 16x8 | 138.3 | 925 ms | 1964 MB |

64x2 is only **+5.2%** for **+672 MB**. Not "meaningful" against the VRAM cost on
a 4 GB card → **retain 32x4**. All layouts finite and identical loss trajectory.

### H3.3D timeout

Batching metrics show no material timeout effect (batch p50 15–16, wait p95
~10 ms at 1000 µs). **Retain 1000 µs.** No further timeout sweep (per pre-reg).

## H3.4 — F15-v2 search-budget curve (MEASURED)

Frozen v2 reference, K = 2, concurrency 8, batch 16, timeout 1000 µs, 32
games/cell. The historical S2 8-sim decision is **HISTORICAL** and not
transferred.

| sims | train_pos/s | truncation | decisive | train entropy | train top-1 | 3fold+fifty |
|---|---|---|---|---|---|---|
| 8 | 35.0 | 0.219 | 0.750 | 1.73 | 0.262 | 0.094 |
| 16 | 20.8 | 0.156 | 0.625 | 2.30 | 0.198 | 0.094 |
| **32** | **13.4** | **0.063** | **0.781** | **2.86** | **0.146** | **0.063** |
| 64 | 7.2 | 0.000 | 0.781 | 3.00 | 0.135 | 0.063 |

No budget degenerates (top-1 ≤ 0.262, entropy ≥ 1.73, truncation ≤ 0.219).
32 sims has **essentially identical** health to 64 (identical decisive 0.781 and
3fold+fifty 0.063; entropy within 4.9%) at **+86%** trainable pos/s → the
pre-registered rule selects **32**.

Root-search diagnostics (trainable): network→noise KL ~0.067 at every budget
(noise is a fixed root perturbation); noise→target KL falls 0.949 (8) → 0.434
(16) → 0.137 (32) → 0.025 (64); network→target KL falls 1.396 → 0.119. At T0 the
network value is ~0 and the root search value ~0.003, i.e. **search barely
improves on the prior until the value head learns** — expected and consistent
with mainline.

**Observation that resolves the first historical blocker:** every H3 self-play
cell is decisive 0.75–0.78 with 21.9–28% draws, versus the historical **89.2%
draws / 69.2% threefold+fifty**. Head v2 + D41 exploration removed the
repetition attractor at the source.

## H3.5 — arena qualification (MEASURED)

Reference-vs-reference, frozen v2, 32 sims, K = 2, c8, paired colors, 32 games.

| variant | decisive | threefold | fifty | truncation | terminations |
|---|---|---|---|---|---|
| V0 (argmax, no noise) | 2/32 = **0.063** | 30/32 = **0.938** | 0 | 0 | 2 checkmate, 30 threefold |
| **V2** (sample 30 plies + root noise 0.25) | 22/32 = **0.688** | 1/32 = **0.031** | 2/32 = 0.063 | 3/32 = 0.094 | 22 checkmate, 2 fifty, 3 insuff, 1 stalemate, 1 threefold, 3 truncated |

0 inference errors; peak VRAM 426 MB; 585 s; both sides the frozen reference
(so 0.362 is opening/color asymmetry, **not** a strength result).

**Adopted V2** — qualifies on every pre-registered criterion (decisive
0.688 ≥ 0.50; threefold 0.031 ≤ 0.30; truncation 0.094 ≤ 0.10) while V0 is
repetition-dominated (0.938 threefold). This confirms mainline's D45 result by
independent HP measurement, and resolves the second historical blocker (arena
uninformativeness). Truncation 0.094 is close to the 0.10 limit → **watch item**
for the smoke. V2 is a **new** HP scientific identity; historical HP arenas are
not relabelled.

## H3.6 — corrected F15-v2 smoke

**NOT RUN at the time of writing**; frozen as `configs/hp/f15-smoke-v2.toml`.
K = 2, concurrency 8, 32 sims, 32 games/cycle, 3 cycles, arena V2, continuous
trainer, reuse target 2.0, `planned_updates = 246`, safety cap 768.

## Transfer table

| mainline lesson | HP applicability | HP measurement | decision |
|---|---|---|---|
| D40 head v2 | contract + new reference | 15,154,632; T0 sane; v1 refused | ADOPTED |
| D41 self-play exploration | kills repetition | 22–28% draws vs 89.2% | ADOPTED |
| D42 search diagnostics | truthful root metrics | root_search present in every cell | ADOPTED |
| D43 inference uses inner backend | generic | inherited via merge | INHERITED |
| D44 owner memory cleanup | **fixes late-run failure** | 496 MB plateau, 0 err | ADOPTED (GO) |
| D45 searched arena | V0 repetition-heavy | V0 2/32 decisive | V2 pending |
| D46 ≤2 resident models | 4 GB card | max_resident = 2 | VALIDATED |
| D47 multi-leaf PUCT | throughput on HP | K=2 +13.2%, health same | ADOPTED K=2 |
| D48 continuous trainer | learner continuity | frozen in smoke | ADOPTED |
| D49 health stops | execution bounds | frozen in smoke | ADOPTED |
| D37/D38/telemetry/tooling | generic | inherited via merge | INHERITED |

## R15 entry decision

_To be written after the corrected F15-v2 smoke. No R15 training is performed._
