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
