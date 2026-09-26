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

Reference-vs-reference, frozen v2, 32 sims, c8, paired colors, 32 games.

> **H3.5B correction:**
> - **K:** these arena cells ran at **K = 1**, not K = 2. In `runs/hp-h3-arena.toml`,
>   `search_leaves_in_flight = 2` sat below the `[model]` header, where TOML
>   scopes it to the model table and serde ignored it. The eval-arena JSON did
>   not record K (it now records `leaves_in_flight`).
> - **"Paired colors":** this meant a shared opening only. Each game had its own
>   RNG stream (`seed = base + i`), so under V2's sampling and noise the two
>   games of a pair were not paired in randomness. See H3.5B.

| variant | decisive | threefold | fifty | truncation | terminations |
|---|---|---|---|---|---|
| V0 (argmax, no noise) | 2/32 = **0.063** | 30/32 = **0.938** | 0 | 0 | 2 checkmate, 30 threefold |
| **V2** (sample 30 plies + root noise 0.25) | 22/32 = **0.688** | 1/32 = **0.031** | 2/32 = 0.063 | 3/32 = 0.094 | 22 checkmate, 2 fifty, 3 insuff, 1 stalemate, 1 threefold, 3 truncated |

0 inference errors; peak VRAM 426 MB; 585 s; both sides the frozen reference.

Identical evaluators scoring 0.362 is **not** a strength result. The original
"opening/color asymmetry" explanation is insufficient, since the evaluators
are identical. Corrected (H3.5B): it is a high-variance stochastic comparison
under **unpaired RNG streams**. The 95% CI [0.209, 0.515] includes 0.5.

**Adopted V2** — qualifies on every pre-registered criterion (decisive
0.688 ≥ 0.50; threefold 0.031 ≤ 0.30; truncation 0.094 ≤ 0.10) while V0 is
repetition-dominated (0.938 threefold). This confirms mainline's D45 result by
independent HP measurement, and resolves the second historical blocker (arena
uninformativeness). Truncation 0.094 is close to the 0.10 limit → **watch item**
for the smoke. V2 is a **new** HP scientific identity; historical HP arenas are
not relabelled.

## H3.5B — pre-smoke red team (MEASURED)

Rules were fixed in `HP_H3_PREREG.md` §H3.5B (commit `78e0743`) before the run.
Binary `78e0743` (clean, CUDA). Both arms: frozen `d89b408f…` vs **itself**,
`configs/hp/f15-smoke-v2.toml` @ `e798889` (32 sims, **K = 2**, c8, batch 16, 32 games,
openings-v1, D45 V2). Run sequentially on the RTX 2050 with nothing else on the GPU.
Evidence: `docs/evidence/hp-h3/arena-paired/{v2-old,v2-paired}/eval-arena.json`.

| | V2-old (`per_game_v1`) | **V2-paired (`paired_common_v1`)** | pre-registered rule |
|---|---|---|---|
| inference errors | 0 | **0** | 0 |
| W / D / L (candidate) | 7 / 7 / 15 | **14 / 4 / 14** | — |
| decisive fraction | 0.688 | **0.875** | ≥ 0.50 |
| threefold | 0.031 | **0.000** | ≤ 0.30 |
| truncation | 0.094 | **0.000** | ≤ 0.10 |
| candidate score (self vs self) | 0.362 | **0.500 exactly** | \|s − 0.5\| ≤ 0.05 |
| per-game 95% CI | [0.209, 0.515] | [0.335, 0.665] | — |
| complete / mirrored / identical-move pairs | 13 / 5 / 0 of 16 | **16 / 16 / 16 of 16** | recorded |
| pair-score histogram | 0.00: 2, 0.25: 6, 0.50: 5 | 0.50: 16 | — |
| peak VRAM / wall | 426 MB / 458 s | 426 MB / 328 s | — |

**Decision: ADOPTED `paired_common_v1`** for H3.6. It passed every criterion,
and the new scientific identity is frozen in the smoke config.

MEASURED observations:
- **CRN is exact here.** With identical evaluators, every pair replayed
  move-for-move on CUDA (16/16 identical move digests), batching included.
  The null comparison scores exactly 0.5. This makes the self-vs-self check
  tautological, as intended: arena variance now comes from differences
  between the models, not from random streams.
  - Per-game marginals are unchanged by CRN. The paired arm's higher decisive
    fraction (0.875 vs 0.688) is a different draw of the same per-game
    distribution, not an effect of pairing.
- **V2-old at K = 2 reproduced H3.5 (K = 1) game for game:** the same
  7/7/15/3 and the same terminations.
  - At T0 the head-v2 value is exactly 0 and priors are near-uniform, so
    multi-leaf selection with virtual loss visits leaves in the same order as
    sequential search.
  - So the H3.5 K misconfiguration (`arena-paired/H35-K-AUDIT.md`) did not
    change its result.
  - The CUDA arena is also run-to-run deterministic for this setup. K is
    expected to matter once the value head learns.
- **The pair-level Wald CI is anti-conservative at n = 13 to 16.** V2-old's
  pair CI [0.209, 0.406] excludes 0.5 for a model against itself. Pair
  intervals remain diagnostic only and must not drive promotion. A
  statistically stronger promotion rule is deferred to R15 entry, as
  pre-registered.
- The frozen reference won 11 of 14 decisive pairs as Black (paired arm).
  This is a T0 property of this network and openings, not a claim.

### D50 correction (MEASURED; see the reproducibility section above)

The artifact `model_id` differs per process because of generated ParamIds.
The weights reproduce exactly per backend, and `d89b408f…` equals fresh CUDA
freezes semantically (`semantic_weights_digest f81938a2…`).

## H3.6 — corrected F15-v2 smoke (MEASURED — CONDITIONAL)

**Run:**
- `runs/hp-h3-f15-smoke-v2-h36`, config `configs/hp/f15-smoke-v2.toml`.
- Scientific identity `866c6afd…`, binary `bd42495` (clean).
- Reference `d89b408f…` (semantic `f81938a2…`), RTX 2050 CUDA FP32.
- 3 cycles in 3,904 s of the 10,800 s budget. Status `completed`; no health
  stop.

**Evidence:**
- `docs/evidence/hp-h3/smoke-h36/`: the cycle reports, `pilot.json`, lineage,
  the pre-flight record and `metrics-extract.json`.
- Root cause: `docs/evidence/hp-h3/smoke-h36/rootcause/`.

| | cycle 0 | cycle 1 | cycle 2 |
|---|---|---|---|
| **actor → candidate** | `d89b408f` → `990e5e54` | `990e5e54` → `dda1283f` | `990e5e54` → `d0ee3ced` |
| **SYSTEM** audit / inference errors / failed games | ok / 0 / 0 | ok / 0 / 0 | ok / 0 / 0 |
| VRAM start / peak (MB) | 298 / 1,452 | 1,452 / 1,516 | 1,516 / 1,516 |
| max temp (°C) | 72 | 72 | 70 |
| resident owners (max) | 2 | 2 | 2 |
| collect / train / eval / wall (s) | 385 / 75 / 499 / 960 | 385 / 68 / 1,063 / 1,518 | 293 / 55 / 1,058 / 1,407 |
| **DATA** W / D / B / T | 9 / 7 / 14 / 2 | 14 / 5 / 11 / 2 | 16 / 3 / 12 / 1 |
| draw share | 0.219 | 0.156 | **0.094** |
| threefold + fifty | 0.062 | 0.031 | 0.031 |
| mean plies | 189.0 | 180.4 | **138.3** |
| trainable positions | 5,247 | 4,973 | 4,027 |
| target entropy / top-1 | 2.857 / 0.146 | 2.834 / 0.155 | 2.862 / 0.160 |
| **SEARCH** KL net→noise / noise→target | 0.067 / 0.137 | 0.069 / 0.158 | 0.070 / 0.176 |
| KL target vs network | 0.258 | 0.293 | 0.321 |
| mean \|network value\| / \|root value\| | 0.000 / 0.003 | 0.024 / 0.020 | 0.027 / 0.024 |
| **TRAINER** step (of 370) | 0 → 82 | 82 → 160 | 160 → 223 (0.60) |
| updates at zero LR | 0 | 0 | 0 |
| LR first → last | 8.1e-6 → 2.87e-4 | 2.87e-4 → 2.11e-4 | 2.10e-4 → 1.24e-4 |
| WDL loss first → last | 1.099 → 0.914 | 1.011 → 0.814 | 0.874 → 0.843 |
| policy loss first → last | 3.163 → 3.120 | 3.107 → 3.071 | 3.226 → 3.161 |
| grad-norm mean / max | 1.85 / 3.83 | 2.31 / 4.70 | 2.26 / 5.27 |
| all metrics finite | yes | yes | yes |
| **REPLAY** reuse (target 2.0) / cap bound | 2.000 / no | 2.008 / no | 2.002 / no |
| updates requested = scheduled = done | 82 | 78 | 63 |
| fresh fraction / mean age (cycles) | 1.00 / 0.00 | 0.67 / 0.33 | 0.49 / 0.68 |
| **EVAL** vs parent: W / D / L / T | 12 / 7 / 11 / 2 | 10 / 6 / 12 / 4 | 12 / 12 / 4 / 4 |
| score, CI | 0.517 [0.357, 0.676] | 0.464 [0.298, 0.631] | 0.643 [0.511, 0.775] |
| pair mean, pair CI | 0.554 [0.407, 0.700] | 0.396 [0.201, 0.591] | 0.667 [0.541, 0.792] |
| decisive / truncation | 0.719 / 0.062 | 0.688 / **0.125** | 0.500 / **0.125** |
| vs frozen reference: score, CI | = parent arena | 0.574 [0.429, 0.719] | **0.733 [0.621, 0.846]** |
| reference truncation | — | **0.156** | 0.062 |
| raw vs random / raw vs parent | 0.54 / 0.547 | 0.52 / 0.484 | 0.52 / 0.484 |
| **decision** | promote | hold | promote |

- The T0 baseline is raw vs random 0.5625.
- Lineage is exact. The replay generator is always the actor.
  - The trainer was continuous (0 → 82 → 160 → 223).
  - The accepted step was 82 after cycle 1's hold and 223 after cycle 2.
  - Checkpoints `snapshot-000` = `990e5e54` @ 82 and `snapshot-002` =
    `d0ee3ced` @ 223.

### Gates

| gate | result |
|---|---|
| zero illegal moves, audit clean | **GO** (audit ok every cycle) |
| zero inference failures, no NaN/Inf | **GO** |
| stable GPU resources | **GO**. 298 → 1,452 → 1,516 → 1,516 MB: a post-training allocator plateau, like mainline smoke v2 (757 → 3,207 → 3,239). ≤ 2 resident owners. |
| trainer advances continuously; lineage correct | **GO** |
| LR schedule does not exhaust | **GO**. It ended at 0.60 of plan with 0 zero-LR updates. |
| `max_updates` never controls reuse; reuse near target | **GO** (2.00 / 2.01 / 2.00; never cap-bound) |
| policy does not collapse | **GO**. Target entropy is flat at about 2.85 and top-1 ≤ 0.16. |
| WDL / value learning accumulates | **GO**. First-update WDL loss on each cycle's fresh data falls 1.099 → 1.011 → 0.874, and \|v\| goes 0 → 0.027. |
| self-play does not trigger D49; no draw attractor | **GO**. Draws fall 0.22 → 0.16 → 0.09 and games shorten 189 → 138 plies. This is the opposite of mainline smoke v2. |
| arena remains informative | **GO**. There are 16–23 decisive games per arena (≥ 4 required). |
| arena truncation acceptable | **CONDITIONAL**. It is 0.125, 0.156 and 0.125 in 3 of 5 arenas, above the 0.10 D45 qualification bound. It is root-caused as a scoring bias (below). |
| strength evidence | Moderate. Candidate vs the frozen reference at cycle 2 is 0.733 [0.621, 0.846], ≥ 0.688 under any truncation scoring. Raw policy is unchanged. |

**H3.6 verdict: CONDITIONAL.**
- The learning mechanism, continuous trainer, RTX 2050 lifecycle,
  replay/reuse and lineage are **GO**.
- The arena has a measured truncation bias that must be handled before
  arenas compare architectures.

### Root cause and findings

1. **Arena truncation = won-but-unconverted positions (MEASURED).**
   - A deterministic replay of the cycle-2 parent arena reproduced it game for
     game. Of its 4 truncated games, 3 were parent wins that couldn't be
     converted (K vs K+Q+N+N; K+B+N vs K twice) and 1 was a dead draw.
   - Truncated games are excluded from `candidate_score`, so unconverted wins
     vanish from the score.
   - Cycle 0's promotion **is not robust** to this: 0.517 as played, 0.484 if
     both truncations were parent wins.
   - Cycle 2's promotion (≥ 0.562) and the cycle-2 reference result (≥ 0.688)
     are robust.
   - Self-play has the same failure: 1–2 truncated games per cycle, whose
     400–800 positions are discarded from value training.
2. **Improvement is coming from the value head, not the raw policy
   (MEASURED).**
   - Raw vs random stays flat (0.5625 → 0.54 → 0.52 → 0.52), and raw vs
     parent is 31/32 draws.
   - Policy loss on fresh data barely moves.
   - Targets stay near-uniform, with entropy about 2.85 and top-1 about 0.15,
     while KL(target ‖ network) grows 0.258 → 0.293 → 0.321.
   - **INFERRED:** with an almost-zero value head, 32 simulations over about
     30 legal moves cannot sharpen visit targets, so the policy has little to
     learn yet. Policy learning is gated on value learning, which the 3-cycle
     smoke only begins.
3. **Learned play shortens games (MEASURED).** Games went 189 → 180 → 138
   plies and trainable volume went 5,247 → 4,973 → 4,027. Updates went
   82 → 78 → 63, so the 370-step schedule was 60 % used. Headroom sized for
   lengthening was not needed here. Future schedules should be sized from this
   measured trajectory.
4. **Determinism (MEASURED).** The CUDA arena replays exactly given its seed
   and models (four independent checks today). CRN pairing and replays are
   trustworthy on this stack.

**Disclosure (H3.5B):** an undocumented partial attempt exists at
`runs/hp-h3-smoke-v2/`.
- It was built at `9889c17` (H3.1), before the H3.2 lifecycle requalification.
- It completed cycle 0 only, and its status is still `running`.
- It is not H3.6 evidence and is left untouched.

## Transfer table

| mainline lesson | HP applicability | HP measurement | decision |
|---|---|---|---|
| D40 head v2 | contract + new reference | 15,154,632; T0 sane; v1 refused | ADOPTED |
| D41 self-play exploration | kills repetition | 22–28% draws vs 89.2% | ADOPTED |
| D42 search diagnostics | truthful root metrics | root_search present in every cell | ADOPTED |
| D43 inference uses inner backend | generic | inherited via merge | INHERITED |
| D44 owner memory cleanup | **fixes late-run failure** | 496 MB plateau, 0 err | ADOPTED (GO) |
| D45 searched arena | V0 repetition-heavy | V0 2/32 decisive, V2 22/32 (both K = 1) | ADOPTED V2 (H3.5); RNG pairing in H3.5B |
| D46 ≤2 resident models | 4 GB card | max_resident = 2 | VALIDATED |
| D47 multi-leaf PUCT | throughput on HP | K=2 +13.2%, health same | ADOPTED K=2 |
| D48 continuous trainer | learner continuity | H3.6: steps 0 → 82 → 160 → 223 carried across a hold | VALIDATED |
| D49 health stops | execution bounds | missing from the smoke config until H3.5B (silently off); H3.6 never near a stop | ADOPTED; restored in H3.5B |
| D37/D38/telemetry/tooling | generic | inherited via merge | INHERITED |

## R15 ENTRY DECISION (2026-09-26)

| # | criterion | status |
|---|---|---|
| 1 | corrected F15 learning mechanism works | **MET** (MEASURED). Value loss falls on fresh data, \|v\| grows, and the cycle-2 candidate beats the frozen reference at 0.733 [0.621, 0.846], robust to truncation. The raw policy has not moved yet. |
| 2 | continuous trainer works | **MET** (MEASURED; steps carried across the hold) |
| 3 | RTX 2050 lifecycle stable | **MET** (MEASURED; 1,516 MB plateau, 0 errors, ≤ 2 resident) |
| 4 | reuse / replay healthy | **MET** (MEASURED; 2.00 / 2.01 / 2.00, fresh fraction 1.0 / 0.67 / 0.49) |
| 5 | no immediate learned draw / repetition collapse | **MET** (MEASURED; draws 0.22 → 0.09) |
| 6 | arena scientifically usable | **PARTIAL**. CRN pairing is exact and informative. The truncation-exclusion bias (MEASURED) makes the per-game score unfit for cross-architecture comparison until truncation-aware scoring is pre-registered. |
| 7 | checkpoint / provenance semantics understood | **MET** (MEASURED; D50 corrected, semantic digest, non-null SHA) |
| 8 | F15 / R15 parameter parity exact | **MET** (15,154,632 both; pinned by tests and pre-flight `model-info`) |

**Decision: CONDITIONAL GO for R15 planning.**
- Do not start R15 training until prerequisite P0 below is pre-registered and
  tested.
- No R15 training was performed.

## R15 EXPERIMENT PLAN (plan only — not executed)

**Question:** at a fixed unique-parameter budget (15,154,632), does executing the
shared R15 core more times per evaluation improve learning under the same
self-play and search contract?

**Primary comparison:** R15 at **R1 vs R2 vs R4** (8 / 12 / 20 executed blocks).
- It uses the same weights layout, the same config except `recurrence`, and
  the same seeds.
- F15 stays the matched-parameter **feed-forward control**. It is not
  functionally identical to R15-R1 (0+8+0 vs 2+4+2 wiring), and is not treated
  as such.

**P0 prerequisites** (engineering, zero-science gate):
1. **Truncation-aware arena scoring.**
   - Report `score_truncation_as_draw` and a pre-registered material
     adjudication at the cap, alongside the historical score. The historical
     score stays unchanged.
   - Promotion for R15 reads a score that cannot improve by failing to
     convert. This is the minimal promotion-v3, pre-registered with a margin
     rule.
2. **An ADR on truncated self-play games:** adjudicate them into WDL targets
   or keep discarding them. The chosen policy is identical across all R arms.
3. **R15 frozen references:** freeze `r15` head v2 at seed 1 on CUDA. Record
   `model_id` and `semantic_weights_digest`, and confirm a second freeze
   reproduces the digest (D50 method).
4. **RTX 2050 requalification at R2 and R4:**
   - lifecycle, train-step VRAM at 32×4 and self-play throughput;
   - R4 runs 2.5× the executed blocks, so derive per-arm wall budgets from
     measurement;
   - no change of K, batch or concurrency between arms unless pre-registered.

**P1 — per-arm smoke:**
- Run the H3.6 contract per arm (3 cycles, 32 games, 32 sims, K = 2, paired
  arena) against each arm's own frozen reference.
- Gates: same as H3.6, plus the P0 truncation scoring.

**P2 — comparison run** (only after P1 GO for all arms):
- **Budgeting.** The primary budget is equal self-play games and equal search
  budget per arm (learning per game). A secondary is equal wall-clock,
  reported separately and never mixed.
- **Metrics per cycle:**
  - first-update WDL loss on fresh data and on a **fixed held-out position
    set**, identical for all arms;
  - candidate vs its frozen reference (truncation-aware);
  - raw policy vs random;
  - the full H3.6 metric set.
- **Cross-arm arenas** at matched cycles: R1 vs R2, R2 vs R4 and R1 vs R4,
  CRN-paired and truncation-aware.
- **Sample size.** Pre-register it. At 32 games, per-game SE ≈ 0.08, so
  detecting a 0.10 score difference needs roughly 96+ games per comparison.
  Pair-level Wald CIs are anti-conservative at n ≤ 16 (H3.5B) and are not
  decision rules.
- **Replication:** at least 2 seeds per arm before any claim.
- **Stop conditions:** D49 (with the schedule guard), lifecycle regression,
  and any cap binding.
- **Claim discipline:** "recurrence helps" requires R2 or R4 to beat R1 on
  the pre-registered primary metric in both seeds. Anything less is reported
  as "not shown".
