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

### R2 (12 executed blocks) — **CONDITIONAL**

The reference artifact is `0cd0036c…`, with the same weights as R1 (semantic
`17b03869…`).

| | cycle 0 | cycle 1 | cycle 2 |
|---|---|---|---|
| actor → candidate | `0cd0036c` → `4e60b0c5` | `0cd0036c` → `6a59f387` | `0cd0036c` → `755e8a2b` |
| wall (collect / train / eval) | 907 s (397 / 99 / 411) | 1,007 s (432 / 92 / 483) | 1,073 s (407 / 83 / 583) |
| audit / errors / VRAM peak | ok / 0 / 1,701 MB | ok / 0 / 1,733 MB | ok / 0 / 1,733 MB |
| draw / threefold+fifty / truncated self-play | 0.156 / 0.031 / 0.062 | 0.25 / 0.062 / 0.031 | 0.188 / 0.094 / 0.062 |
| mean plies / trainable positions | 176.0 / 4,833 | 187.5 / 5,599 | 184.4 / 5,102 |
| trainer step (of 370) / zero-LR updates | 0 → 76 / 0 | 76 → 164 / 0 | 164 → 244 / 0 |
| WDL loss first → last | 1.099 → 0.944 | 1.000 → 0.887 | 0.906 → **0.781** |
| reuse / cap bound / fresh fraction | 2.013 / no / 1.00 | 2.012 / no / 0.67 | 2.007 / no / 0.50 |
| **vs frozen reference, adjudicated** | 0.484 [0.342, 0.627] | 0.438 [0.280, 0.595] | **0.594 [0.474, 0.714]** |
| arena truncation | 0.0 | 0.062 | **0.25** |
| raw vs random | 0.43 | 0.46 | 0.48 |
| decision | hold | hold | promote (step 244) |

**Gates:**
- All GO criteria are met.
- One **CONDITIONAL** flag: arena truncation 0.25 in cycle 2 (> 0.10). The
  stronger candidate reaches more won-but-unconverted positions.

### R4 (20 executed blocks) — **CONDITIONAL**

The reference artifact is `6c41aec0…`, with the same weights as R1 (semantic
`17b03869…`).

| | cycle 0 | cycle 1 | cycle 2 |
|---|---|---|---|
| wall (collect / train / eval) | 1,436 s (562 / 124 / 749) | 1,374 s (600 / 125 / 649) | 1,466 s (580 / 95 / 791) |
| audit / errors / VRAM peak | ok / 0 / 2,275 MB | ok / 0 / 2,309 MB | ok / 0 / 2,309 MB |
| draw / truncated self-play | 0.156 / 0.062 | 0.25 / 0.062 | 0.094 / **0.125** |
| trainer step / zero-LR updates | 0 → 70 / 0 | 70 → 146 / 0 | 146 → 206 / 0 |
| reuse / cap bound | 2.022 / no | 2.003 / no | 2.005 / no |
| **vs frozen reference, adjudicated** | 0.406 [0.251, 0.561] | 0.359 [0.219, 0.500] | 0.578 [0.438, 0.718] |
| arena truncation | **0.156** | 0.094 | 0.031 |
| decision | hold | hold | promote (step 206) |

**Gates:**
- All GO criteria are met.
- **CONDITIONAL** flags: arena truncation 0.156 (c0) and truncated self-play
  0.125 (c2).

## P1 verdict and cross-arm observations (MEASURED; one seed, n = 32 per arena)

- **All three arms: CONDITIONAL.**
  - The mechanism is GO in every arm: 0 errors, audit clean, VRAM plateaus
    (1,477 / 1,733 / 2,309 MB), continuous trainers, 0 zero-LR updates,
    reuse 2.0 never cap-bound, exact lineage.
  - Every flag is a truncation flag. That is the conversion weakness already
    documented in D53, D56 and H3.6.
- **Value learning diverges by recurrence** (the main finding). The table
  gives the mean WDL loss over the last quarter of each cycle's updates.

| arm | c0 | c1 | c2 | fresh-data first-update WDL at c2 | max grad-norm |
|---|---|---|---|---|---|
| R1 | 0.881 | 0.852 | **0.832** | 0.902 | 7.19 |
| R2 | 1.015 | 0.923 | **0.859** | 0.906 | 7.88 |
| **R4** | 0.988 | 1.033 | **1.097** | **1.151** (> uniform 1.099) | **15.62** |

- **R4 in detail:**
  - At the shared LR (3e-4) and equal updates, R4's value head does not
    learn, and on fresh data it predicts worse than uniform.
  - Its gradient-norm spikes are twice R1/R2's.
  - INFERRED: the 20-block unrolled recurrence is harder to optimize at this
    LR and update count.
  - This is not instability in the pre-registered sense: metrics are finite,
    the policy is stable, and it promoted at c2.
- **All arms dip early, then recover.** Scores vs the reference go
  0.36 → 0.45 → 0.53 (R1), 0.48 → 0.44 → 0.59 (R2) and 0.41 → 0.36 → 0.58
  (R4). Every CI but R1 c0 and R4 c1 includes 0.5.
  - **No R1-vs-R2-vs-R4 strength claim is made** (pre-registered).
- **Raw policy is flat everywhere** (vs random 0.43–0.54). As in H3.6,
  learning shows up through the value head and search only.
- **Measured cost per cycle,** the input to P2 budgeting: about 725 s (R1),
  1,000 s (R2) and 1,425 s (R4).

## P1b — R4 value-learning LR probe (MEASURED; pre-registered `5de7d28`)

**Setup:**
- `bench-train`, 32 × 4, 120 updates from step 0, on the P1 R4 replay
  (96 reference-generated games).
- All cells start from the same reference weights (semantic `17b03869…`).
- Evidence: `docs/evidence/hp-r15-p1b/`, with per-update curves.

| cell | WDL loss, first 30 → last 30 updates | policy loss, last 30 | max / mean grad |
|---|---|---|---|
| A: R4 @ 3e-4 (P1) | 1.036 → **1.101** | 3.131 | 5.10 / 1.97 |
| B: R4 @ 1.5e-4 | 1.057 → **0.999** | 3.130 | 5.32 / 2.13 |
| C: R1 @ 3e-4 (control) | 1.051 → **0.954** | 3.130 | 4.52 / 2.05 |

**Verdict: LR-DRIVEN.**
- B − A = −0.101 (the bar was ≤ −0.05), and the control separates
  (C − A = −0.147).
- At the shared LR, R4's value head gets worse. At half the LR it learns,
  though at 120 updates it still trails R1 (0.999 vs 0.954).
- Gradient spikes here stay at ≤ 5.3 in every cell, so P1's 15.6 spikes were
  likely data-specific. The loss trend is the robust signal.
- No LR is adopted from this probe, as pre-registered. It only sets the P2
  design.

## P2 proposal (NOT RUN; needs owner approval, since it is a large GPU commitment)

**Amended by P1b.** P2 adds an **R4 @ 1.5e-4** arm next to R1, R2 and R4 at 3e-4.
Without it, "R4 is worse" cannot be separated from "the shared LR handicaps
deeper recurrence." There are 4 arms × 2 seeds. At 8 cycles that is about
8 × (725 + 1,000 + 1,425 + 1,425) s ≈ 10 h per seed, so **~20 h of GPU time**,
plus about 1.5 h of cross-arm arenas. A 6-cycle variant takes about 15 h.


The first recurrence comparison that could support a claim.

**Design:**
- Base design: 3 arms (R1/R2/R4) × **2 seeds** × **8 cycles**. P1b adds the
  R4 @ 1.5e-4 arm.
- Identical contract to P1, plus **sampler v2** (per-position recency
  weighting, pre-registered). Sampler v2 removes the shard-level coupling to
  the truncation rate.

**Metrics:**
- Pre-registered primary: **fresh-data first-update WDL loss at the final
  cycle**, the value-learning signal P1 showed separating by R.
- Secondary: each arm's adjudicated score vs the shared reference weights,
  plus **cross-arm arenas** at the final cycle (96 games per pair,
  CRN-paired, root-player, truncation-aware).
- Always report grad-norm distributions per arm.

**Claim rule:**
- "Recurrence helps" requires R2 or R4 to beat R1 on the primary metric in
  **both** seeds, and the cross-arm arena CI to exclude 0.5.
- "Recurrence hurts at a fixed LR" is reported symmetrically.

**Cost:**
- About 8 × (725 + 1,000 + 1,425) s ≈ 7 h per seed, so **~14 h of GPU time**
  for 2 seeds.
- Plus cross-arm arenas (~1.5 h).

**Before P2 (cheap):**
- A 20-minute pre-registered check of whether R4's value stall is LR-driven:
  60 updates at LR 1.5e-4 vs 3e-4 on the same replay, reading the WDL-loss
  trajectory.
- This decides whether P2 should also carry an R-scaled-LR arm. Changing LR
  per arm is a separate scientific choice, not a silent fix.

## V1 — fast value-learning sweep (MEASURED; pre-registered `e48f1b9`, amended `6da2132`)

**Setup:**
- 16 cells of 2–6 minutes each: `bench-train` on one fixed replay (P1 R1,
  96 games, about 13.9k positions), scored with `eval-value` on the fixed
  held-out set (10,249 positions).
- Metric: held-out WDL cross-entropy (uniform = 1.0986).
- Evidence: `docs/evidence/hp-r15-v1/` (one `result.json` per cell).
- The long P2 run was stopped by the owner in favor of this loop.

| cell (100 updates) | seed 1 | seed 2 |
|---|---|---|
| R1 @ 3e-4 | 0.9597 | 0.9543 |
| R1 @ 1.5e-4 | 0.9643 | — |
| R2 @ 3e-4 | 1.0059 | 0.9720 |
| R2 @ 1.5e-4 | 0.9633 | 0.9689 |
| R2 @ 7.5e-5 | 0.9594 | — |
| R4 @ 3e-4 | 0.9949 | **1.1487** (worse than uniform) |
| R4 @ 1.5e-4 | 0.9844 | 0.9790 |
| **R4 @ 7.5e-5** | **0.9535** | **0.9656** |

| cell (200 updates, seed 1) | held-out | train WDL (last 20) |
|---|---|---|
| R1 @ 3e-4 | 0.9743 (worse than at 100) | 0.833 |
| R4 @ 7.5e-5 | 0.9601 (worse than at 100) | 0.890 |

**Findings:**
1. **Seed-stability depends on recurrence at a fixed LR** (EXPLORATORY). At
   3e-4 the seed spread is R1 0.005, R2 0.034, R4 0.154.
2. **R4's best LR is lower** (CONFIRMED, ≥ 0.01 in both seeds). 7.5e-5 beats
   1.5e-4 (−0.031 / −0.013), which beats 3e-4 (−0.011 / −0.170).
   - The P1 R4 value stall was an LR artifact.
3. **At each arm's best LR, value learning per update ties** (not shown by
   the rule). R4 @ 7.5e-5 vs R1 @ 3e-4 is +0.006 / −0.011.
   - Per unit of compute, R1 is ahead, since R4 costs 2.5× per update
     (INFERRED from the measured step times).
4. **The fixed-replay loop is exhausted.** By 200 updates (about 1.8 epochs)
   both arms overfit: train loss falls while held-out loss rises.
   - Longer offline training on these 96 games measures memorization. A
     larger fixed dataset (or self-play) is needed to go further.

## Learning-loop probes after V1 (2026-09-28, MEASURED; exploratory)

### LR head-to-head
- R1 trained at 7.5e-5 (held-out WDL 0.894) vs R1 at 3e-4 (0.963), same data
  and 400 updates.
- 32 games: adjudicated **0.516 [0.404, 0.628]**, a tie.
- A better held-out value head **did not play stronger**. Evidence:
  `hp-r15-v1/h2h-*`.

### Policy diagnostic
- The policy head learns clean targets perfectly: 1.04 → 0.0000 on a fixture
  (`policy_learning.rs`, now a regression test).
- **No code bug.**

### T1 — **RETRACTED conclusion**
- Training the V1 network further on trained-actor data moved its policy loss
  (−0.023), while training on "reference-generated" data left it flat
  (+0.003).
- I first read this as "an untrained actor gives no policy signal."
- **That reading was confounded.** The reference-generated replay (the P1 R1
  replay) was already part of the V1 network's training set, so the flat
  policy reflects data it had already fit.
- The pilot data contradicts the claim. Policy loss falls at the same slow
  rate under an untrained actor (P1 R1 c0: −0.019) and a trained one (T2 c0:
  −0.015).

### T2 — the latest-actor loop (`snapshot_policy = latest`, LR 7.5e-5, warm start from V1 R1)
- 16 games × 2 cycles, 14.5 min. Evidence: `hp-r15-t2/`.
- **Negative:**
  - held-out WDL 0.889 / 0.902 → **1.041 / 1.138** (worse);
  - held-out policy 3.161 / 3.206 → 3.231 / 3.343 (worse);
  - self-play target entropy 2.16 → 1.77 (narrowing);
  - raw vs random 0.50 → 0.50.
- The network fits its own narrowing play. The `latest` option stays
  implemented, but it is **not adopted**.

### Where this leaves the project (MEASURED unless noted)
1. **Recurrence** gives no value-learning benefit at 15M params at matched
   LR, at 2.5× compute (V1, both seeds).
2. **LR 7.5e-5** gives better held-out value but no strength gain.
3. **[RETRACTED, see §"Measurement correction" below: held-out policy CE cannot show policy learning.]** ~~**The policy barely learns** under 32-simulation self-play targets in
   every setup tried. The cause is **not established**. Candidates: too few
   simulations for informative targets, target noise, or too little data.
   Each needs a clean test.~~

## P-1 — are 32 simulations the policy bottleneck? (MEASURED; pre-registered `8fb55f3`)
- **Setup:**
  - M = V1 R1 (`runs/v1/R1-lr7.5e-5-s1-u400-d288`).
  - Oracle: 8 games at 256 sims with noise 0 (seed 901). This gives 1,076
    positions, target entropy 2.58 and top-1 visit share 0.16.
  - Training sets: 8 games from M at 32 sims (seed 902) and at 128 sims
    (seed 903).
  - Each set: 40 updates at LR 7.5e-5 (32×4), then `eval-value` on the
    oracle.
  - Evidence: `docs/evidence/p-1/` (`result.json`, `oracle-*/`, `train-M*/`,
    `oracle256/`, `train32/`, `train128/`).
  - Wall time: 10.5 + 2.3 + 5.7 min generation, plus about 3 min training
    and scoring.

| Network | Oracle policy CE | Oracle WDL CE |
|---|---|---|
| M | 2.7387 | 0.7230 |
| M + 40 upd on 32-sim | 2.7242 (−0.0144) | 1.0015 |
| M + 40 upd on 128-sim | 2.7020 (−0.0367) | 0.5440 |

- **Pre-registered rule:** gain128 − gain32 = **0.0223 ≥ 0.02 → "sims are the
  policy bottleneck"**.
- **Caveats (MEASURED, disclosed and not explained away):**
  - The margin is thin: 0.0223 against a threshold of 0.02.
  - Single seed; 8 games per set.
  - **Data size is confounded.** The 32-sim set had 899 trainable positions
    because 3 of its 8 games hit the 400-ply cap. The 128-sim set had 1,531.
  - With 40 × 128 samples each, the 32-sim data was reused about 5.7×
    against about 3.3×.
- **Also observed (not a pre-registered claim):**
  - 32-sim targets are *sharper* (entropy 2.02,
    top-1 share 0.28) than deep-search targets (2.58 / 0.16). The 128-sim
    targets are close to deep search (2.59 / 0.19).
  - So 32 sims gives confident targets that are the wrong shape, not merely
    noisy ones.
  - Training on 32-sim data also made oracle WDL worse (0.72 → 1.00).
- **Status:** it passes the rule, but it is not yet acted on, because of the
  data-size confound. A size-controlled replication (P-1r) is pre-registered
  in `HP_H3_PREREG.md`.

## P-1r — size-controlled replication (MEASURED; pre-registered `7e6d566`) — **INCONCLUSIVE**
- **Setup:**
  - 32-sim arm: 16 games (seed 904), **3,726** trainable positions.
  - 128-sim arm: 8 games (seed 905), **1,323** trainable positions.
  - So the size control holds, against the hypothesis.
  - Same M, oracle, training and metric as P-1. Evidence:
    `docs/evidence/p-1r/result.json`.
- **Results:**
  - Oracle policy CE: M 2.7387, M+32 2.7196 (gain 0.0191), M+128 2.7016
    (gain 0.0370).
  - Oracle WDL CE: 0.723 / 0.588 / 0.543.
- **Rule:** gain128 − gain32 = **0.0179**, which falls between 0.01 and 0.02,
  so the result is **inconclusive**. The pipeline is not changed.
- **Consistent across P-1 and P-1r (MEASURED):**
  - The 128-sim gain was 0.037 both times.
  - The 32-sim gain was 0.014 and 0.019, the latter with 2.8× the 128-sim arm's
    data.
  - Per wall-clock, 32 sims produced about 5× more trainable positions per
    second (14.4 against 2.9).
  - The cost-matched question is untested.

## Measurement correction — held-out policy CE cannot show policy learning (MEASURED)
- The P2 held-out replays were generated by the **untrained** reference at
  32 sims, so their targets are near-uniform-prior search. Mean target entropy
  is 2.893 (s1) and 2.867 (s2): `holdout/gen-s*/sweep.json`.
- **The untrained reference itself** scores held-out policy CE 3.163 / 3.175
  (`holdout/ref-s*/eval-value.json`). The V1 R1 network scores 3.161 / 3.206.
  A policy that learns real preferences *should* move away from these
  targets.
- **Retracted:** the statement "the policy barely learns under 32-simulation
  targets", in §"Where this leaves the project" item 3, rested on this metric.
  It is **not established either way** by held-out CE.
- **Not affected:** held-out **WDL** CE, whose targets are game outcomes
  independent of the generator, and every value and recurrence conclusion
  (V1).
- On the deep-search oracle (P-1), M's excess loss over the target entropy
  (2.58) is only about 0.16. Both training sets reduced it.
  - Caveat: the oracle was searched from M's own prior, which favours M.

## P-2 — raw-policy arena, M vs untrained reference (MEASURED; pre-registered) — **not shown; test degenerate**
- **Setup:** 2 sims (prior argmax), noise 0, 32 paired games, 24 s.
- **Result:** 0 W / 32 D / 0 L, score 0.500, CI [0.500, 0.500].
  - Terminations: threefold 23, insufficient material 6, stalemate 3.
  - Evidence: `docs/evidence/p-2/`.
- **Design flaw (disclosed):**
  - Deterministic argmax play by both sides falls into repetition loops.
  - This arena cannot discriminate policies, so no policy claim is made from
    it.
  - A raw-policy comparison needs sampled play, which the arena does not
    support at 2 sims (`sample_action` weights by visits).

## P-3 — V1-trained network vs untrained reference, with search (MEASURED; pre-registered `c64dcc5`) — **not shown**
- **Setup:** 64 games under the standard R1 contract (32 sims, D45), paired,
  root_player_v1. Wall time **12.9 min**. Evidence:
  `docs/evidence/p-3/eval-arena.json`.
- **Outcomes:**
  - M W/D/L 19/24/14, 7 truncated.
  - Terminations: 33 checkmates, 15 insufficient material, 8 fifty-move, 1
    stalemate.
- **Scores:**
  - Adjudicated (material_v1): **0.570**, game CI [0.478, 0.663].
  - Complete-pair mean 0.600, pair CI [0.494, 0.706].
- **Rule:** both CIs include 0.5, so the result is **"not shown"**. The
  direction favours the trained network, but no strength claim is made.
- **What this means (MEASURED):** 400 updates on 288 self-play games give at
  most a modest strength gain, one that 64 games cannot resolve.
  - At about 12 s per game, resolving a ~5-point edge needs a few hundred
    games, which is roughly 40–60 min on this GPU.
