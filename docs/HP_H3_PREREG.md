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
