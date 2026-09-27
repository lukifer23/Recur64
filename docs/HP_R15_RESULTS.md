# HP R15 — prerequisites (P0) and core review results

Status legend: **MEASURED** (ran on this machine), **INFERRED**, **NOT RUN**.
Pre-registration: [`HP_H3_PREREG.md`](HP_H3_PREREG.md) §R15-P0 (commit `e2a9a44`) and
its P0.4 amendment (`938220f`). Prior phase: [`HP_H3_RESULTS.md`](HP_H3_RESULTS.md).
**No R15 training has been run.**

## P0.1 — truncation-aware arena scoring (MEASURED — ADOPTED, D53)

- **`material_v1` adjudication:** a game truncated at the cap is scored from
  its final material (P1 N3 B3 R5 Q9). A balance of ≥ +5 is a win, ≤ −5 a
  loss, and anything else a draw.
- **promotion-v3** (`promotion_score = "adjudicated_material_v1"`) makes
  promotion read that score. It is opt-in and identity-neutral by default.
- **Pre-registered validation:** the H3.6 cycle-2 parent-arena replay re-scores
  to exactly **12 / 13 / 7 = 0.578**, versus 0.643 as played
  (`h36_cycle2_replay_rescores_to_the_preregistered_value`).
- **Adoption:** all R15 arms use promotion-v3.

## P0.2 — truncated self-play games (DECIDED, D52)

Truncated self-play games stay untrainable, identically across arms. The
truncated share is reported per arm, and the comparison is CONDITIONAL if two
arms differ by more than 0.10.

## P0.3 — R15 frozen reference (MEASURED — GO)

`configs/hp/r15-reference-v2.toml` was frozen twice, in separate processes, on
CUDA at seed 1.

| | freeze a (pinned) | freeze b |
|---|---|---|
| `model_id` (artifact) | `385f4f27b5e9…` | `7ef81c0f8157…` |
| `semantic_weights_digest` | `17b0386992d7…` | `17b0386992d7…` |

- 0 of 15,154,632 elements differ between the two freezes.
- 136 tensors, head v2, 15,154,632 unique params: **identical to F15**.
- Pinned: `runs/hp-r15-p0-ref-r15-v2`. One reference serves R1, R2 and R4.
- Evidence: `docs/evidence/hp-r15-p0/reference/`.

## P0.4 — RTX 2050 requalification at R1 / R2 / R4 (MEASURED)

**Setup:**
- Binary `bac2bba`. The H3 schedule: K = 2, concurrency 8, batch 16,
  1000 µs timeout, 32 sims, training 32 × 4 on the H3 replay.
- The first attempt straddled Windows sleep, was discarded, and is disclosed
  in the pre-registration. The re-run held a keep-awake request.

| arm | executed blocks | train step (32×4) | train ex/s | train peak VRAM | self-play trainable pos/s | self-play fwd (ms, batch ≈ 12–14) | 32-game collect (s) | errors |
|---|---|---|---|---|---|---|---|---|
| R1 | 8 | 873 ms | 146.6 | 1,281 MB | 12.1 | 23.9 | 418 | 0 |
| R2 | 12 | 1,259 ms | 101.7 | 1,537 MB | 8.8 | 31.5 | 551 | 0 |
| R4 | 20 | 2,027 ms | 63.1 | **2,113 MB** | 6.3 | 48.4 | 718 | 0 |

- Every arm is below the 3.0 GB gate; the worst case is R4 training at
  2,113 MB. All training metrics are finite.
- R1 matches F15, both in train VRAM (1,290 MB) and in self-play
  (13.4 trainable pos/s).
- Evidence: `docs/evidence/hp-r15-p0/{train,runtime}/`.
- **R4 lifecycle (amended to 2 reps per mode, all 4 modes): GO.**
  - Post-shutdown VRAM: 289 (one), 417 → 449 (two), 449 (pilot), 449
    (pilot-promoted, 3 spawned / 2 resident).
  - 0 errors across 8 lifecycles, no growth. F15's H3.2 plateau was 492 MB.
  - Rep 0 of `one` ran under concurrent CPU load from a test build, so its
    337 s and 68 ms are inflated. Timing is not a gate.
  - Evidence: `docs/evidence/hp-r15-p0/lifecycle/r4/`.

**P0.4 verdict: GO for all three arms.** Measured per-arm cost for P1
budgeting: self-play collection for 32 games is 418 / 551 / 718 s, and train
steps are 0.87 / 1.26 / 2.03 s.

## Core code review (2026-09-27)

- **Method:** three independent read-only reviews covering model and
  training, search and self-play, and inference and runtime. Each finding
  was verified in code before any action.

### Fixed

| finding | severity | fix |
|---|---|---|
| **Arena search trees mixed both networks.** The per-node side router sent every tree node to the network of that node's side to move. Found independently by two reviewers. | critical for evaluation validity | **D54** `arena_tree_policy = "root_player_v1"`. The historical default is kept so earlier identities reproduce. |
| NaN policy silently became uniform, NaN WDL became value 0 (`softmax3` fallback), on both inference paths | medium (no-silent-fallback rule) | returns `EvalError` (`c1a5f68`) |
| A non-finite gradient was skipped in `grad_norm`, so the NaN guard could not fire | low–medium | propagated to the guard |
| A reply was sent before the metrics counter was incremented, the root cause of the recurring `errors_propagate_to_every_request` flake | low (metrics truthfulness) | count before reply (`bac2bba`) |

### Deferred, with reason

| finding | why deferred |
|---|---|
| Replay is sampled per shard, not per position, and an empty shard falls back to the oldest one. This couples the training distribution to the truncation rate. | Sampler semantics are part of the identity. To be pre-registered as sampler v2 before the R15 comparison. |
| The knight-promotion delta is never applied (`is_promo` tests the column, not the code). | Changes the head contract (HEAD_VERSION 3 and new references). Low impact. |
| `achieved_reuse` is total throughput over new positions, not fresh-position reuse. | Add a truthful fresh-reuse metric next to it. |
| A queue-full error is not backpressure, and nothing checks `concurrency × K ≤ 512`. | Not reachable at the current schedules. |
| `PHASE4_RESULTS.md` blames CubeCL autotune for forward variance, but autotune was never compiled in. | Mainline doc; annotated here, not rewritten. |

### Throughput (evidence-based; replaces the "launch-bound" premise)

- **Self-play forward cost grows with batch (MEASURED):** about 4.8 ms fixed
  plus ~1.4 ms per position at F15/R1.
- **The flat 17–23 ms seen in arenas is two owners sharing the GPU**
  (INFERRED). Each owner's timer includes the other's work.
- **Evaluation is 52–75 % of pilot wall time (MEASURED, H3.6).**
- **Ranked levers:**
  1. **Evaluation scheduling.** Implemented as execution-only settings:
     `eval_concurrency`, `eval_max_inference_batch`, and concurrent
     Phase-A matches. D54 also fills batches better. Estimated 1.5–2.5× on
     evaluation; NOT yet measured.
  2. **Burn fusion, then autotune.** Both are currently off
     (`default-features = false`). Estimated 1.3–1.6× per forward. Needs an
     ADR, a parity test and a D44 VRAM re-check.
  3. **Host-side cleanups:** cache `rel_idx` or the per-block bias, use one
     readback, drop observation copies. Estimated 2–5 %.
  4. **Pipeline the owner** (up to ~1.15× in self-play) and **fuse QKV** at
     load (3–5 %).

## P0.5 — D54 impact and evaluation scheduling (MEASURED; pre-registered `1e196cf`)

**Setup:**
- Same checkpoints and seeds as the H3.6 cycle-2 arenas (offset 2, paired RNG,
  K = 2, 32 sims, 32 games), now with `root_player_v1` (each side searches its
  own tree).
- Evidence: `docs/evidence/hp-r15-p0/d54-impact/`.

| cell | pair | H3.6 mixed trees (as played) | root-player as played | root-player T = draw | root-player **adjudicated** (W / D / L) | truncated |
|---|---|---|---|---|---|---|
| M1 | cycle-2 candidate vs frozen reference | 0.733 | 0.643 | 0.594 | **0.750 [0.618, 0.882]** (21 / 6 / 5) | 11 |
| M2 | cycle-2 candidate vs parent (step 223 vs 82) | 0.643 | 0.500 | 0.500 | **0.547 [0.420, 0.674]** (10 / 15 / 7) | 5 |

**Findings (MEASURED):**
- **Material effect in both cells.** The shift is ≥ 0.05 under the
  pre-registered rule.
- The mixed-tree bias is **not** a simple pull toward 0.5, as predicted. On M2
  it *overstated* the difference (0.643 vs 0.547 adjudicated). Mixed-tree
  scores are confounded in no fixed direction.
- **The learning is real** (0.750 vs the untrained reference) and mostly early.
  From step 82 to step 223 the gain is small and not significant at n = 32.
  Cycle 2's mixed-tree promotion would not clearly hold under correct trees.
- **Conversion is the dominant failure.** In M1, 10 of 11 truncations are the
  candidate holding crushing material (K+Q vs K, K+Q+Q+N vs K, K+R vs K)
  without mating.
  - Those self-play games are discarded from training (D52), so the value head
    never learns them as wins.
  - This is the top learning-side fix candidate, to take up after the
    throughput pass (needs an ADR).

**Scheduling (M3):** 1.20× (587 → 489 s) with 32/32 identical games. This is
below the pre-registered 1.3× adoption bar, so it is not adopted for R15. The
GPU reached ~91 % busy. See `docs/PERF_LEDGER.md` #1.


## P1 — per-arm R15 smokes (MEASURED; pre-registered `fa21ef1`, amended)

**Build and schedule:**
- D55 build (`--features cuda,fusion,autotune`), binary `fa21ef1`.
- Self-play at c12 / batch 24; evaluation at c8 / batch 16.
- root_player_v1 arenas, promotion-v3, 3 cycles.
- Evidence: `docs/evidence/hp-r15-p1/r{1,2,4}/`, with pre-flight records and
  metrics extracts.

### R1 (8 executed blocks) — **CONDITIONAL**

| | cycle 0 | cycle 1 | cycle 2 |
|---|---|---|---|
| actor → candidate | `385f4f27` → `7d067f4e` | `385f4f27` → `767db27c` | `385f4f27` → `e2819ee1` |
| wall (collect / train / eval) | 766 s (342 / 74 / 349) | 703 s (297 / 61 / 344) | 713 s (265 / 74 / 373) |
| audit / errors / VRAM peak | ok / 0 / 1,445 MB | ok / 0 / 1,477 MB | ok / 0 / 1,477 MB |
| draw / threefold+fifty / **truncated self-play** | 0.25 / 0.094 / **0.156** | 0.219 / 0.062 / 0.062 | 0.188 / 0.0 / 0.094 |
| mean plies / trainable positions | 211.7 / 4,773 | 179.7 / 4,950 | 167.8 / 4,169 |
| trainer step (of 370) / zero-LR updates | 0 → 75 / 0 | 75 → 153 / 0 | 153 → 219 / 0 |
| WDL loss first → last | 1.099 → 0.988 | 0.842 → 0.901 | 0.902 → 0.860 |
| reuse / cap bound / fresh fraction | 2.011 / no / 1.00 | 2.017 / no / 0.67 | 2.026 / no / 0.49 |
| **vs frozen reference, adjudicated** (= parent arena) | **0.359 [0.226, 0.493]** | 0.453 [0.311, 0.595] | 0.531 [0.386, 0.677] |
| arena truncation | 0.0 | 0.031 | **0.125** |
| raw vs random | 0.54 | 0.54 | 0.52 |
| decision | hold | hold | promote (step 219) |

**Gates:**
- All GO criteria are met: audit clean, 0 errors, VRAM plateau under 3 GB,
  continuous trainer with exact lineage, 0 zero-LR updates, the cap never
  binds, reuse ≥ 0.8 × target, and no D49 stop.
- Two **CONDITIONAL** flags: truncated self-play share 0.156 (> 0.10, cycle 0)
  and arena truncation 0.125 (> 0.10, cycle 2).

**Findings (MEASURED):**
- The 75-update candidate is **worse** than the untrained reference: its CI
  excludes 0.5 on the low side.
- It then recovers as training continues (0.453, then 0.531).
- INFERRED: an early, barely-trained value head adds misleading signal to
  search compared with a neutral zero value.
- The raw policy stays flat, as in H3.6.

### R2 / R4

These are pending (re-launched after the reference-artifact amendment).
