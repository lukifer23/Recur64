# Recur64 HP H3 — pre-registration

Thresholds here are fixed **before** the corresponding measurement, per
`AGENTS.md` evidence discipline. If a measured cell changes an outcome, the
rule is not moved; the deviation is reported instead.

Reference under test: `runs/hp-h3-ref-f15-v2`
(`model_id d89b408fcc7a3cd9874a0ffbbcafd121c6c78cbb9818d4adaac0fe8d51ea234b`),
F15 head v2, CUDA FP32, seed 1, R=1. Historical head-v1 evidence is untouched.

## H3.2 lifecycle GO rule (already exercised)

GO if, across `one`, `two`, `pilot`, `pilot-promoted` at reps 8:

- post-shutdown VRAM plateaus (no monotonic growth across reps);
- zero inference errors;
- forward latency stable after warmup;
- peak VRAM comfortably below 4 GB.

Otherwise STOP science. **Result: GO** (peak 496 MB; 0 errors; latency flat).

## H3.3A multi-leaf K selection rule

Adopt the **smallest K > 1** that satisfies *all* of:

1. trainable positions/sec improves by **≥ 10 %** vs K=1;
2. **0** inference errors;
3. decisive fraction within **0.10 absolute** of K=1;
4. threefold + fifty not worse by more than **0.05 absolute**;
5. target entropy within **10 %** of K=1;
6. VRAM has comfortable margin (peak **< 3.0 GB** preferred for sustained runs).

If K=2 qualifies, prefer **K=2** over K=4 unless K=4 gives a substantial extra
gain that clearly justifies the stronger search-execution perturbation and
stays resource-safe. Do **not** import mainline's K=2 choice without measuring.

## H3.3B concurrency rule

After K is fixed, measure a ring {8, 12, 16} at `batch cap = concurrency × K`.
Primary metric: trainable positions/sec. Constraints: 0 errors, stable VRAM and
thermals, no pathological queue. Differences **< 5 %** → prefer **lower**
concurrency. 24 is CONDITIONAL only if 16 clearly gains without pathological
queue behaviour. The CPU has 12 logical threads; do not oversubscribe blindly.

## H3.3C training physical batch

Effective batch stays **128**. Compare `32 × 4` (HP prior) vs `64 × 2`, optional
`16 × 8` control, on real new replay. Adopt `64 × 2` only if the effective batch
is identical, the speed gain is meaningful, and peak VRAM leaves safe margin on
the 4 GB card. Otherwise keep `32 × 4`.

## H3.3D timeout

Only revisit `500` vs `1000 µs` if batching metrics show timeout is material.
Default: retain **1000 µs**.

## H3.4 search-budget selection rule

Eligible primary budgets: **32 and 64**. 16/8 are curve/diagnostic unless 32+
prove impractical. 128 only if it materially improves data health/search while
retaining practical throughput; no 256 on the RTX 2050.

Definitions fixed in advance:

- "essentially the same data health as 64" = decisive fraction within **0.10
  absolute**, threefold+fifty within **0.05 absolute**, target entropy within
  **10 %**, and no degeneration flag set.
- "large practical throughput gain" for preferring 32 = **≥ 25 %** higher
  trainable positions/sec than 64.
- "materially improves" for preferring 128 = target entropy higher by **≥ 10 %**
  *or* decisive fraction higher by **≥ 0.10 absolute**, at **≤ 25 %** lower
  trainable positions/sec than 64.

Degenerate (health-fail) if **any**: trainable target top-1 > 0.90;
threefold + fifty > 0.80; trainable target entropy < 0.10; truncation > 0.50.

64 remains the transfer default if it is healthy and 32 does not meet the
"large" gain bar.

## H3.5 arena rule (D45 eligibility)

V2 = sample first 30 plies + root Dirichlet ε 0.25. Eligible for adoption iff:

- decisive fraction **≥ 0.50**;
- threefold **≤ 0.30**;
- truncation **≤ 0.10**.

If V2 qualifies and V0 (deterministic argmax, no noise) stays draw/repetition
heavy, adopt V2 as a **new** HP scientific identity. Historical HP arenas are
not retroactively relabelled.

## H3.6 smoke health stops (execution safety, not reward)

Stop a run if any of:

- draw share **≥ 0.85** for two consecutive cycles;
- threefold + fifty **≥ 0.60** in one cycle;
- truncation **≥ 0.25** in one cycle.

Always record: trainer step, cap_bound, reuse, fresh replay fraction, mean
sample age, lineage. `max_updates` is a safety cap only; if it binds and
controls reuse, the smoke is CONDITIONAL/NO-GO and the cap is **not** raised
mid-run.

## H3.5B — pre-smoke red team (fixed before any H3.5B GPU measurement)

### Disclosures found during the red team

- **Abandoned smoke attempt.** `runs/hp-h3-smoke-v2/` (gitignored, never reported)
  is a partial pilot built at `9889c17`. That is H3.1, *before* the H3.2
  lifecycle requalification and before this pre-registration was committed.
  - It completed cycle 0 only, and `metadata.json` still says `running`.
  - It was run with an empty `[health_stops]`.
  - Cycle 0: 5,247 trainable positions, 82/82 updates, reuse 2.00, hold at
    arena score 0.468, 978 s wall. Scientific hash `510d9bb4…`.
  - It is **not** H3.6 evidence and is left untouched. H3.6 uses a fresh run
    directory.
- **H3.5 arena ran at K = 1, not K = 2.** In `runs/hp-h3-arena.toml` (the config behind
  `docs/evidence/hp-h3/arena/v{0,2}`), `search_leaves_in_flight = 2` and
  `run_budget_minutes` sit **below** the `[model]` header. TOML assigns them to
  the model table, and serde ignores unknown keys, so the arena used the default
  K = 1 and the default budget.
  - The H3.5 text "32 sims, K = 2" is therefore wrong for the arena cells.
  - The eval-arena JSON never recorded K. It now records `leaves_in_flight`.
  - The H3.5B arms below use `configs/hp/f15-smoke-v2.toml`, where K = 2 is a
    top-level key.
- `configs/hp/f15-smoke-v2.toml` had **no `[health_stops]`**, so D49 was off
  despite being documented as frozen.
- CLI JSON (`eval-arena`, `bench-train`, `bench-lifecycle`) recorded
  `git_revision: null`, because build-script env is crate-scoped. Fixed
  centrally (`recur64_runtime::provenance`). Historical JSON is unchanged.

### H3.5B arena RNG gate

Fixed setup:
- The frozen reference `d89b408f…` vs **itself**.
- `configs/hp/f15-smoke-v2.toml`: 32 sims, K = 2, concurrency 8, batch 16, 32
  games, openings-v1, seed offset 0.
- D45 V2 (sample 30 plies, root ε 0.25).

Two arms, run sequentially on the RTX 2050 with nothing else on the GPU:
- **V2-old:** `--rng-policy per_game_v1` (historical `seed = base + i`). This
  re-run gives pair diagnostics that the H3.5 JSON lacks. The score is recorded
  as measured and not expected to equal 0.362, because H3.5 ran at K = 1.
- **V2-paired:** `--rng-policy paired_common_v1` (`seed = base + i/2`).

Adopt `paired_common_v1` for H3.6 **iff all** of these hold for V2-paired:

1. 0 inference errors;
2. decisive fraction ≥ 0.50;
3. threefold ≤ 0.30;
4. truncation ≤ 0.10;
5. |candidate_score − 0.5| ≤ **0.05** ("materially closer to 0.5" than
   H3.5's |0.362 − 0.5| = 0.138).

Record exact 0.5, mirrored and identical-move pairs, and any non-mirrored pair.
CUDA batching composition differs between the two games of a pair, so exact
mirroring is not guaranteed and is not a criterion.

Per-game marginal distributions are unchanged by common random numbers. If
criteria 2–4 fail for V2-paired, the arena is not informative at this budget:
**STOP before the smoke** and analyse, without silently falling back. Promotion
stays conservative-v2 on the per-game point estimate. Pair diagnostics are
reported, not used for promotion (no promotion-v3 in this pass).

### H3.6 LR schedule amendment

- **T0 volume (HP measured):** 5,247 trainable positions per cycle at 32 sims, K = 2
  (H3.4 s32). The abandoned attempt's cycle 0 reproduced it exactly. That gives
  ceil(5,247 × 2.0 / 128) = **82** updates.
- **Lengthening priors:**
  - Mainline F10 smoke v2 (external prior, not HP evidence): mean plies 184 →
    261 (×1.42), updates 93 → 127, and 38 updates at LR 0.
  - Historical HP F15 head-v1: mean plies 173 → 240 (×1.39).
- **Stress factor 1.75** for the learned cycles 1–2 (≈ ×1.4 observed plus 25 %
  margin): 5,247 × 1.75 = 9,183 positions → **144** updates per cycle.
- **`planned_updates = 82 + 144 + 144 = 370`, `warmup_updates = 37`** (the 10 %
  convention of every earlier schedule).
  - At the expected 246 steps the cosine LR is still about 0.3 × base.
  - It reaches 0 only at step ≥ 370.
- **No extension after the run starts.**

### H3.6 safety cap

- Truncated games are untrainable (cycle 0: 6,047 − 5,247 = 2 × 400). So
  trainable positions per cycle stay below 32 × 400 = 12,800, and the
  theoretical maximum request is ceil(12,800 × 2 / 128) = **200** updates per
  cycle.
- `max_updates` is **per cycle**. It is set to **256**, which cannot bind unless
  accounting is wrong.
  - If it binds, the smoke is CONDITIONAL/NO-GO.
  - It is never raised mid-run.

### H3.6 schedule-exhaustion guard

- Every cycle report carries an `lr_schedule` block with `step_start`,
  `step_end`, `planned_updates`, `fraction_end`, `lr_first`, `lr_last` and
  `updates_at_zero_lr`.
- `[health_stops] lr_schedule_end = true` stops the pilot at the cycle boundary
  once `step_end ≥ planned_updates`.
- `updates_at_zero_lr > 0` in any cycle makes the smoke **CONDITIONAL**, not
  normal learning.

### H3.6 budgets (execution bounds, unchanged)

- `position_budget = 45,000`, above the theoretical 3 × 12,800 = 38,400.
- `run_budget_minutes = 180`: the abandoned attempt's cycle 0 took 978 s
  with parent == reference, and the stress estimate is ~35 min per cycle.

D49 `[health_stops]` is restored with the pre-registered values
`draw_share_two_cycles = 0.85`, `threefold_fifty = 0.60` and `truncation = 0.25`.

## R15-P0 — prerequisites before any R15 training (fixed before measurement)

Owner-approved (2026-09-26) after the H3.6 CONDITIONAL result. **No R15 training
in P0.** P1 (per-arm R15 smokes) needs a separate go-ahead.

### P0.1 Truncation-aware arena scoring (`material_v1`)

H3.6 measured that truncated arena games are mostly won-but-unconverted
positions, and they are dropped from `candidate_score`.

**Material adjudication.** Every arena game is scored; none is dropped.
- A game that ended normally keeps its result.
- A game truncated at the ply cap is adjudicated from its final position:
  - Material uses standard values P = 1, N = 3, B = 3, R = 5, Q = 9 (kings
    excluded), counted from the candidate's side.
  - A balance of **≥ +5** is a candidate win, **≤ −5** is a candidate loss,
    and anything else is a draw.
  - A final position that cannot be reconstructed counts as a draw.
- Why +5: one rook's worth is the smallest standard balance that is a forced
  win with bare pieces (K+R vs K, K+B+N vs K).
  - K+B vs K+N (0) and K+R vs K+B (+2) are draws.
  - Q vs R (+4) scores a draw. That is conservative: it can only understate a
    real advantage, never invent one.

**Reported fields.** The historical fields (`candidate_score`, its CI and
`decisive_games`) are unchanged. New fields:
- `score_truncation_as_draw = (W + 0.5·(D + T)) / games`;
- `adjudicated {rule, wins, draws, losses, adjudicated_games, decisive,
  score, ci}`, where `score = (W' + 0.5·D') / games` over **all** games.

**Promotion-v3** (`promotion_score = "adjudicated_material_v1"`):
- It keeps the exact conservative-v2 health gates.
- The score test reads `adjudicated.score` instead of `candidate_score`, and
  the decisive-games floor reads `adjudicated.decisive`.
- A candidate can no longer gain by failing to convert, or by its opponent
  failing to convert against it.
- The default stays conservative-v2 on `candidate_score`, so every earlier
  identity is unchanged. The setting enters the scientific identity only when
  set.
- **No additional margin rule.** At 32 games the per-game SE is about 0.08, so
  a margin small enough to allow any promotion gives little protection. With
  the continuous trainer, learning does not depend on promotion. Strength
  claims come only from the pre-registered, larger cross-arm arenas.

**Validation (MEASURED before adoption):** re-score the cycle-2 H3.6 parent-arena
replay, which has final FENs. Expected (computed by hand from the four FENs):
- g0 → candidate loss;
- g13 and g15 → candidate losses (the parent had K+B+N);
- g19 → draw.

That gives adjudicated W / D / L = 12 / 13 / 7 and a score of
(12 + 6.5) / 32 = **0.578**, versus 0.643 as played. The code must reproduce
this exactly.

### P0.2 Truncated self-play games (ADR D52, decided now)

**Keep discarding** truncated self-play games from value and policy training,
identically for every R arm, and **report** the truncated share per arm per
cycle.
- Reason: adjudicating them into WDL targets would change the training
  targets. That needs its own measurement, and it would confound an R1/R2/R4
  comparison if introduced mid-programme.
- The bias is known and measured: 1–2 of 32 games per cycle in H3.6. It is
  equal in kind across arms.
- If one arm's truncated share exceeds another's by more than 0.10 absolute,
  the comparison is CONDITIONAL on that difference.

### P0.3 R15 frozen reference

- Config `configs/hp/r15-reference-v2.toml`: identical to
  `f15-reference-v2.toml` except the 2 + 4 + 2 block layout, `run_id` and
  `model_profile`. Seed 1, CUDA FP32, head v2.
- Freeze **twice in separate processes**. GO iff both freezes have the same
  `semantic_weights_digest`, 15,154,632 unique params and `HEAD_VERSION` 2.
- The first freeze becomes the pinned reference.
- One reference serves R1, R2 and R4, because recurrence is not a model
  parameter.

### P0.4 RTX 2050 requalification at R1 / R2 / R4

Tools: the frozen R15 reference and the H3 schedule (K = 2, concurrency 8, batch
16, timeout 1000 µs, 32 sims, training 32 × 4). Nothing else runs on the GPU.
- `bench-lifecycle` (modes one, two, pilot and pilot-promoted; reps 8) at
  R4, the worst case.
- `bench-train` at 32 × 4 on the H3 replay, at R1, R2 and R4.
- `bench-runtime` self-play, 32 games at 32 sims, at R1, R2 and R4.

GO per arm iff:
- 0 inference errors;
- the post-shutdown VRAM plateau does not grow monotonically across reps;
- peak VRAM **< 3.0 GB** in every tool;
- all training metrics finite.

Throughput (trainable pos/s, s per update) is **recorded, not gated**. It sets
each arm's P1 wall budget, at 2× the measured cycle estimate. No K, batch or
concurrency change between arms unless an arm fails the VRAM gate; in that
case, STOP and report.

### R15-P0.4 amendment (2026-09-27, before the R4 lifecycle step ran)

- **Sleep contamination.** The first P0.4 attempt (started 2026-09-26 18:00)
  straddled Windows sleep/resume (system time-change events at 19:05 and 23:45)
  and was stopped unfinished. Its outputs were deleted, not used. The re-run
  holds a user-space keep-awake request (`SetThreadExecutionState`) for its
  duration; no power setting is changed.
- **R4 lifecycle: reps 8 → 2 per mode**, all four modes kept.
  - Owner create/shutdown/reload is independent of recurrence: recurrence is
    a forward argument, not owner state.
  - D44/D46 were already validated over 32 lifecycles on F15 (H3.2).
  - At R4 the full 8 reps cost about 2.5 h of GPU time.
- **Gates unchanged:** 0 errors, no VRAM growth across reps, peak < 3.0 GB.
  Training VRAM (the R-dependent risk) was measured first: R4 peaks at
  2,113 MB.

## R15-P0.5 — D54 impact and evaluation-scheduling measurement (fixed before running)

**Diagnostic** re-runs of the H3.6 cycle-2 arenas: same checkpoints, same seeds
(offset 2), `paired_common_v1`, K = 2, 32 sims, 32 games, the run's own config.
They change no historical result.

| cell | models | tree policy | schedule | compared with |
|---|---|---|---|---|
| M1 | `d0ee3ced` vs reference `d89b408f` | `root_player_v1` | c8 / batch 16 | H3.6 mixed-tree 0.733 |
| M2 | `d0ee3ced` vs parent `990e5e54` | `root_player_v1` | c8 / batch 16 | H3.6 mixed-tree 0.643 |
| M3 | as M1 | `root_player_v1` | **c32 / batch 32** | M1 (wall time, game identity) |

**Reported, not gated:**
- the as-played score, `score_truncation_as_draw` and the `material_v1`
  adjudicated score, with the change against the mixed-tree result;
- wall time and batch statistics.

**Interpretation, fixed in advance:**
- D54 predicts that per-player trees *increase* separation from 0.5 whenever
  the networks differ. A shift of ≥ 0.05 in either cell counts as a material
  effect. A smaller shift is reported as "no material effect measured at
  n = 32".
- Scheduling (M3 vs M1) must not change what the arena measures. Move-digest
  identity of M3 vs M1 is recorded.
  - CUDA batch composition can legitimately change float results, and
    therefore games. So identity is *not* required.
  - If the games differ, the scores must agree within the per-game CI.
- Adopt `eval_concurrency = 32` and `eval_max_inference_batch = 32` for R15
  iff M3 is **≥ 1.3× faster** than M1 with 0 errors and peak VRAM < 3.0 GB.

## D56 — early material adjudication, shadow-mode validation (fixed before running)

Owner-approved 2026-09-27.

**Rule `early_material_v1`:** the side whose `material_v1` balance (P1 N3 B3 R5
Q9) is ≥ +5 for **40 consecutive plies** is adjudicated the winner at the
firing ply. The result is the one D53 assigns at the cap, reached earlier.

**Shadow mode (observe only):** every game plays to its natural end or the cap.
Each game records:
- the firing ply and leader, if any;
- the leader's minimum balance after firing (flip detection);
- the plies that would have been saved.

**Validation set:** the P0.5 M1 and M2 pairs (cycle-2 candidate vs reference and
vs parent), root_player_v1, paired RNG, K = 2, 32 sims, seed offset 2, 32 games
each. That is 64 games, all of them decided under D53.

**Adopt enforcement for R15 iff all of these hold:**
1. **Verdict agreement ≥ 95 %.** Among games where the rule fires, the leader
   equals the D53-adjudicated winner of the full game.
2. **No flip.** No fired game in which the leader's balance later falls to ≤ 0.
3. **Saved plies ≥ 25 %** of total arena plies.
4. **0 inference errors.**

Every disagreement is reported. If the gate fails, STOP and report; there is
no retuning of N or the threshold on the same data.

## Perf #11 / #13 — scheduling on the accepted D55 build (fixed before running)

Execution-only settings; neither enters the scientific identity.

- **#13 evaluation scheduling.**
  - Compare the M1 arena (`d0ee3ced` vs reference, root_player_v1, paired RNG,
    offset 2, 32 games) on the D55 build at c8 / batch 16 against the
    existing D55 c32 / batch 32 run. The c32 run is the 338 s shadow run;
    shadow mode does not change play.
  - Adopt `eval_concurrency = 32` and `eval_max_inference_batch = 32` for R15
    iff c32 is **≥ 1.3× faster** with **identical move digests** on all 32
    games and 0 errors.
- **#11 self-play concurrency.**
  - 16 games per cell at c8 / c12 / c16 (batch = 2 × c), trained `d0ee3ced`,
    32 sims, K = 2.
  - Adopt the smallest concurrency whose trainable positions/s is within 5 % of
    the best, with 0 errors and peak VRAM < 3.0 GB (the H3.3B rule).

## R15-P1 — per-arm R15 smokes (fixed before any R15 training)

Owner go-ahead: 2026-09-27 ("next steps, end to end"). P0 is complete
(`HP_R15_RESULTS.md`). D55 is accepted, D56 was rejected by its own gate, and
D53/D54 are in force.

**Arms:**
- `configs/hp/r15-smoke-r{1,2,4}.toml`: identical except `recurrence` (and
  `run_id`).
- Frozen R15 reference `385f4f27…` (semantic `17b03869…`), seed 1, CUDA FP32,
  head v2.
- D55 build (`--features cuda,fusion,autotune`) with
  `inference_candidate_buckets = true`.

**Scientific contract (all arms):**
- Search and self-play: the H3.6 contract. 32 sims, D41 exploration, K = 2,
  32 games/cycle, reuse 2.0, 32 × 4, continuous trainer, LR 370/37, and a
  256 per-cycle cap.
- Arena: D45 V2 with paired RNG, **root_player_v1 (D54)** and
  **promotion-v3 (D53)**.
- 3 cycles. D49 health stops plus `lr_schedule_end`.
- D52: truncated self-play stays untrainable. The truncated share is reported
  per arm.
- Scheduling (execution only): self-play at 12 games / batch 24 / 12 CPU workers (perf #11); evaluation at 8 games / batch 16 (perf #13 not adopted).

**Execution bounds (from P0.4 measurements, 2× estimate):**
- `run_budget_minutes`: R1 180, R2 240, R4 360.
- `position_budget` 45,000.

**Per-arm gates (as H3.6):**
- **GO iff:**
  - audit clean; 0 inference errors; no NaN/Inf;
  - VRAM plateau below 3.0 GB;
  - trainer continuous with lineage exact;
  - 0 zero-LR updates; the cap never binds; achieved reuse ≥ 0.8 × target;
  - no D49 stop.
- **CONDITIONAL if:** arena truncation > 0.10 in any arena, or truncated
  self-play share > 0.10 in any cycle, or the draw share rises by more than
  0.25 over the arm's cycle 0.
- **NO-GO:** a lifecycle regression, training instability, lineage
  corruption, or repeated CUDA failure.

**This is not a recurrence comparison.**
- Each arm is judged only against its own frozen reference. The report gives
  each arm's adjudicated score vs the reference, with CI, and flags every CI
  that includes 0.5.
- Any R1-vs-R2-vs-R4 strength claim waits for the P2 design: ≥ 96 games per
  comparison, ≥ 2 seeds, and sampler v2 pre-registered.
- Per-arm cost (wall per cycle, per phase) **is** reported, because it is the
  input to P2 budgeting.

**Order:** R1, then R2, then R4, sequentially, one GPU job at a time, under a
keep-awake hold. A NO-GO arm stops the sequence. A CONDITIONAL arm does not.

### R15-P1 amendment (2026-09-27, after R1, before R2/R4 ran)

- **The failure:** the R2 arm refused to start with "reference checkpoint
  recurrence differs from the run config" (`pilot.rs:417`).
  - The pilot requires the reference checkpoint's recorded recurrence to equal
    the run's.
  - P0.3's "one reference serves R1/R2/R4" was true of the **weights**, since
    recurrence is not a parameter, but not of the checkpoint **artifact**.
- **Fix, with the safety check kept:**
  - The same config was re-frozen with `recurrence = 2` and `recurrence = 4`
    (`configs/hp/r15-reference-v2-r{2,4}.toml`).
  - Both artifacts have semantic digest `17b03869…`, and 0 of 15,154,632
    elements differ from the R1 reference. Evidence:
    `docs/evidence/hp-r15-p1/reference-artifacts/`.
  - R2 pins `0cd0036c…` and R4 pins `6c41aec0…`.
- **Effect:**
  - All arms still start from identical weights.
  - `reference_model_id` now differs per arm, as an artifact identity only.
    The contract test compares scientific identities with recurrence and
    `reference_model_id` masked.
  - R1 is unaffected and not re-run.

## R15-P1b — R4 value-learning LR probe (fixed before running)

**Question:** is R4's value stall in P1 learning-rate-driven?

**Setup:**
- `bench-train`, 32 × 4, 120 timed updates plus 2 warmup, starting from
  optimizer step 0 (the P1 cycle-0 schedule: warmup 37, planned 370).
- Replay: the P1 R4 replay (`runs/hp-r15-smoke-r4/replay`, 96 games, all
  generated by the shared reference weights). All three cells start from the
  same reference weights (semantic `17b03869…`).
- Cells: **A** R4 @ LR 3e-4 (the P1 setting), **B** R4 @ 1.5e-4, **C** R1 @ 3e-4
  (control).

**Metric:** the mean WDL loss over the last 30 updates, plus the max grad norm.

**Interpretation:**
- C is expected to fall below A. If it does not, the P1 separation is data or
  noise, not recurrence, and is reported as such.
- **LR-driven** iff B's last-30 WDL is ≥ 0.05 lower than A's.
- **Not LR-driven** iff |B − A| < 0.05. In that case P2 keeps one shared LR,
  and the stall is treated as a recurrence property to measure.
- No LR is adopted from this probe. It only decides whether P2 needs an
  R-scaled-LR arm.

## R15-P2 — first recurrence comparison (fixed before any P2 training)

Owner-approved 2026-09-27: **2 seeds × 6 cycles**.

**Arms:**
- Configs: `configs/hp/r15-p2-{r1,r2,r4,r4lr}-s{1,2}.toml` (contract test
  `r15_p2_arms_share_one_contract`).
  - R1, R2, R4 at LR 3e-4.
  - **R4lr** is R4 at LR 1.5e-4, added by P1b.
- Seed 1 starts from reference weights `17b03869…` and seed 2 from
  `4e1beb4e…`. Each arm uses its recurrence-matched artifact.
- The contract is P1's plus: 6 cycles, LR schedule 800/80, sampler v2
  (`position_recency_v2`), position budget 90,000, and wall budgets
  150 / 210 / 300 / 300 min.
- D55 build, self-play c12 / b24, evaluation c8 / b16.

**Order:** seed 1 R1, R2, R4, R4lr, then seed 2 in the same order.
- Sequential, one GPU job, keep-awake hold.
- A crashed or NO-GO arm is recorded and the sequence **continues**, since
  arms are independent.

**Held-out set (fixed now):** `runs/hp-r15-p2-holdout-s1` and `-s2`.
- 32 games each, generated by each seed's reference at R1 with self-play
  seeds 777 / 778 (unused by any arm).
- Evidence and generation logs: `docs/evidence/hp-r15-p2/holdout/`.
- Correctness check: the untrained reference scores exactly ln 3 = 1.0986 on both sets (5,893 and 4,356 positions).
- Committed copies: `docs/evidence/hp-r15-p2/holdout/replay-s{1,2}/`. Shard SHA-256: s1 `a35de659…bf42`, s2 `04d9e824…3b80`.

**Primary metric:** the held-out WDL cross-entropy of each arm's **final trainer
checkpoint** (`checkpoints/trainer` after cycle 5), at the arm's own recurrence.
It is scored with `recur64 eval-value` on both held-out replays and
position-weighted into one number per arm per seed.

**Secondary metrics:**
- Held-out policy CE.
- Cross-arm arenas per seed: **R2 vs R1, R4 vs R1, R4lr vs R1**.
  - Final trainer checkpoints, 96 games each.
  - Per-side recurrence (`--reference-recurrence 1 --candidate-recurrence r`),
    under the arm's own seed's R1 config contract (root_player_v1, paired
    RNG, c8 / b16).
  - Reported as played, as T = draw, and adjudicated with CI.
- Per-arm P1-style gates and cost.

**Claim rules (fixed):**
- **Recurrence improves value learning:** an arm's held-out WDL CE is lower
  than R1's by **≥ 0.01 in both seeds**.
- **Recurrence hurts value learning:** R1 is lower by ≥ 0.01 in both seeds.
  This is judged separately for R4 at 3e-4 and for R4lr.
- **Strength claim:** the adjudicated cross-arm score CI excludes 0.5 in the
  same direction in **both seeds**.
- **Anything else is reported as "not shown."** P2 makes no claim beyond these
  rules, and no metric or threshold is changed after the runs start.

## R15-V1 — fast value-learning sweep (replaces the long P2 run as the inner loop; fixed before running)

**Why:** the owner rejected 7+ hour feedback loops (2026-09-27). The long P2 run
was stopped about 25 minutes into its first arm, with no results used. The
question P1 and P1b raised (how value learning depends on recurrence and LR)
is measurable in minutes on fixed data.

**Setup:**
- Every cell trains from its seed's recurrence-matched reference (seed 1
  semantic `17b03869…`, seed 2 `4e1beb4e…`) on **one fixed training replay**:
  `runs/hp-r15-smoke-r1/replay`, the P1 R1 replay of 96 games generated by the
  seed-1 reference.
- Training: `bench-train`, 32 × 4, 2 + 200 updates, the P2 config schedule
  (80 / 800).
- Scoring: `eval-value` on both held-out replays (P2 pre-registration),
  position-weighted.
- Cells, each at seeds 1 and 2: R1 / R2 / R4 at 3e-4, R1 / R2 / R4 at 1.5e-4,
  and R4 at 7.5e-5 (14 cells, about 3–7 min each). Seed 1 runs first,
  starting with R1@3e-4, R4@3e-4 and R4@1.5e-4.

**Primary metric:** held-out WDL CE after 200 updates.

**Rules:**
- An effect between two cells counts only if the difference is ≥ 0.01 **in
  the same direction in both seeds**. Anything else is reported as "not
  shown."
- The best LR per recurrence is reported, but no LR is adopted without a
  follow-up confirmation.
- This measures offline value learning, not self-play dynamics. A
  self-play confirmation of the winners is a separate, later step.

### R15-V1 amendment — the owner's fast loop (before any V1 cell ran)

- **Cells:** one cell per run, **100 updates** (not 200), about 2–3 min each
  including held-out scoring.
- **Order:** chosen iteratively from the previous result.
- **Status:** every V1 result is **EXPLORATORY**. An apparent effect is a
  finding only after a seed-2 run of the same cells reproduces its direction
  with ≥ 0.01.
- **Stopped P2 run:** partial outputs were deleted (never used). Its run
  directory is kept as `runs/hp-r15-p2-r1-s1-stopped`.

## R15-L1 — does the V1 LR finding make a stronger network in self-play? (fixed before running)

**Arms:**
- **A** = the existing P1 R1 run (`configs/hp/r15-smoke-r1.toml`, LR 3e-4,
  3 cycles). Its final trainer scores held-out WDL CE 0.9671 / 0.9777.
- **B** = `configs/hp/r15-lr-r1-7p5e-5.toml`, identical except LR 7.5e-5, run
  fresh for 3 cycles.

**Measures:**
- Held-out WDL CE (both sets, position-weighted) of B's final trainer.
- A **96-game head-to-head** of B's final trainer vs A's final trainer:
  R1 vs R1, root_player_v1, paired RNG, c8 / b16. Reported adjudicated, with
  CI.

**Rule:**
- "The lower LR improves self-play learning" iff B's held-out WDL CE is
  ≥ 0.01 below A's, **and** the head-to-head adjudicated CI lies above 0.5.
- One held-out gain without an arena gain is reported as "value improves,
  strength not shown."
- This is single-seed, so it is exploratory; adoption as the default LR
  needs a seed-2 repeat.

## P-1 — are 32 simulations the policy bottleneck? (fixed before running)

- **Starting network** M = the V1 R1 network (`runs/v1/R1-lr7.5e-5-s1-u400-d288`).
- **Oracle:** 8 self-play games by M at **256 sims, noise 0**, seed 901. The
  metric is `eval-value` policy CE against these deep-search visit targets.
- **Training sets:** 8 games by M at **32 sims** (seed 902) and at **128 sims**
  (seed 903), standard D41 noise.
  - M is trained on each for 40 updates at LR 7.5e-5 (32 × 4).
  - M, M+32 and M+128 are all scored on the oracle.
- **Rules:**
  - "Sims are the policy bottleneck" iff (oracle CE of M − M+128) − (M − M+32)
    ≥ 0.02.
  - "Not the cause" iff |that difference| < 0.01.
  - Otherwise the result is inconclusive.
  - No other claims.

## P-1r — size-controlled replication of P-1 (fixed before running)
- Same M, same oracle (`runs/p1x-oracle256`, seed 901), same training (40 updates,
  LR 7.5e-5, 32×4) and metric as P-1.
- **New data, new seeds:**
  - 32-sim: **16 games**, seed 904. This gives the 32-sim arm *at least as
    much* trainable data as the 128-sim arm, so the confound works against
    the hypothesis.
  - 128-sim: 8 games, seed 905.
- Trainable position counts are recorded. If the 32-sim arm still has fewer
  trainable positions than the 128-sim arm, the size control failed, and
  that is reported.
- **Rule** (same thresholds as P-1):
  - "Confirmed" iff gain128 − gain32 ≥ 0.02.
  - "Not replicated" iff it is < 0.01.
  - Otherwise inconclusive.
- **Only "confirmed" moves the pipeline** to higher simulation counts for
  training targets.

## P-2 — has the raw policy learned anything? (fixed before running)
- **Why:** held-out policy CE cannot measure policy learning. Its targets come
  from the **untrained** reference's 32-sim search (target entropy
  2.89 / 2.87), and the untrained net itself scores 3.163 / 3.175 on it (see
  `HP_R15_RESULTS.md` §P-1r).
- **Test:** a raw-policy arena between M (`runs/v1/R1-lr7.5e-5-s1-u400-d288`)
  and the untrained seed-1 reference (`runs/hp-r15-p0-ref-r15-v2`), both at R1.
  - Config `configs/probe/raw-policy-r1-s1.toml` is `r15-p2-r1-s1.toml`
    with only `simulations_per_move = 2`.
  - At 2 sims, one child is expanded and the move is the prior argmax
    (`puct.rs::select`, all q = 0).
  - `--noise-epsilon 0`, so play is deterministic given the opening.
  - 32 games = 16 openings × 2 colours, paired, root_player_v1.
- **Rule** (adjudicated material_v1 score, pair-level 95 % CI):
  - "The raw policy learned" iff the CI lower bound is > 0.5.
  - "Not shown" otherwise.

## P-3 — is the V1-trained network stronger than the untrained reference with search? (fixed before running)
- **Arena:** M (`runs/v1/R1-lr7.5e-5-s1-u400-d288`) vs the untrained seed-1
  reference (`runs/hp-r15-p0-ref-r15-v2`), both at R1.
- **Settings:** the standard `configs/hp/r15-p2-r1-s1.toml` contract
  (32 sims, D45 sampling and noise, paired_common_v1, root_player_v1,
  c8 / b16), **64 games**.
- **Rule** (adjudicated material_v1 score, pair-level 95 % CI):
  - "Training produced search strength" iff the CI lower bound is > 0.5.
  - "Training made it weaker" iff the CI upper bound is < 0.5.
  - "Not shown" otherwise.

## P-3b — baseline strength check, fresh 192 games (fixed before running)
- **Owner-approved policy (2026-09-28):**
  - Short runs (≤ 3 min) for everything except playing strength.
  - One large strength check per real change, with its time stated first.
  - This is the baseline check: *does the current training produce strength
    at all?*
- **Arena:** same as P-3 (M vs untrained seed-1 reference, standard R1
  contract, c8 / b16), **192 games**.
  - `--seed-offset 10000`, so the seeds (base 10001 + i/2) are disjoint from
    P-3's.
  - Estimated 39 min, under the keep-awake hold, as the only GPU job.
- **Primary result:** the fresh 192 games **alone**. P-3's result was already
  seen, so pooling would be optional stopping.
  - Adjudicated material_v1 score, game-level 95 % CI.
  - "Training produced search strength" iff the CI lower bound is > 0.5.
  - "Weaker" iff the CI upper bound is < 0.5.
  - "Not shown" otherwise.
- **Secondary (disclosed as pooled after a look):** P-3 + P-3b, 256 games.
