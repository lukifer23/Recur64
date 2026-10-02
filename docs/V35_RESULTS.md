# V3.5 results: on-policy information-acquisition rescue

**Outcome (pre-registered rule, applied once on 2026-10-02): PARTIAL - CONTENT.**
Content-Use PASS, Gate VI PASS, Gate II FAIL, Gate III FAIL. HOLDOUT_C NOT EVALUATED. B16 NOT RUN (primary gates failed).
Plan: `docs/V35_RESEARCH_PLAN.md`. Decisions V35-D1..D7. Evidence: `docs/evidence/v35/` (`v35-gates.json` is the gate output).

All numbers are V3_TUNE_V1, KQRvK M3 (750 positions) unless stated, update 800, three seeds (5101/5102/5103), every arm the
same checkpoint. Branch `experiment/workstation-v35-onpolicy`, source HEAD `7e508df8d369fb34c58990774213bc51c762c504`.

## Runs and health

Three fresh runs from each seed's selected P5 final weights, fresh `adamw-v1`, peak LR 3e-4, 800 updates, micro16 x accum8,
CUDA FP32, health checks on; all exit 0, uninterrupted. Recipe digests: 5101 `742b193d...555f`, 5102 `b03afc32...2f9b`,
5103 `a14539d9...fa15a`. Training wall 1,225 / 1,348 / 1,360 s. Detached-rollout vs autodiff-replay policy difference was
**0.0** over every update of every run; all losses and gradients finite; VRAM 2,938-3,034 MiB across all checkpoints
(plateau rule: post-update-100 maximum within 5% of the earlier maximum: met). Gate VI: **PASS**.

## KQRvK M3 policy metrics (per seed: top-1 / correct mass / CE / entropy)

| seed | B0 | ACTIVE B2 | ACTIVE B4 | ACTIVE B8 | FIXED B8 (top-1, CE) |
|---|---|---|---|---|---|
| 5101 | 0.523 / 0.391 / 2.341 / 1.977 | 0.516 / 0.370 / 2.313 / 2.128 | 0.483 / 0.360 / 2.315 / 2.155 | 0.501 / 0.355 / 2.312 / 2.198 | 0.525, 2.317 |
| 5102 | 0.553 / 0.393 / 2.307 / 2.042 | 0.537 / 0.370 / 2.283 / 2.174 | 0.512 / 0.362 / 2.284 / 2.208 | 0.536 / 0.362 / 2.276 / 2.226 | 0.557, 2.308 |
| 5103 | 0.540 / 0.357 / 2.348 / 2.162 | 0.515 / 0.340 / 2.344 / 2.271 | 0.476 / 0.331 / 2.352 / 2.304 | 0.512 / 0.337 / 2.339 / 2.306 | 0.535, 2.338 |

B0 itself improved relative to the P5 checkpoints (the B0 arm is trained in the same updates): mean M3 B0 top-1 is 0.539 here
vs 0.457 for the P5 B0 reference. Top-1 trajectory (B0 / B8), update 0 -> 800: 5101 0.449/0.491 -> 0.523/0.501; 5102
0.489/0.525 -> 0.553/0.536; 5103 0.432/0.473 -> 0.540/0.512. At update 0 ACTIVE B8 was above B0; by update 800 it is below.
Intermediate evaluations were diagnostics only; no checkpoint selection used them.

## Gate II (same-weight compute): FAIL

`ACTIVE_B8 - B0`: pooled **-0.0222**, 95% CI [-0.0400, -0.0049], per seed -0.0213 / -0.0173 / -0.0280. Threshold +0.10; CI not
above 0; no seed positive. ACTIVE B8 is slightly WORSE than the same weights without queries.

## Gate III (learned selection): FAIL

`ACTIVE_B8 - FIXED_B8` (`fixed_bfs_actionid_v1`): pooled **-0.0227**, CI [-0.0396, -0.0058], per seed -0.0240 / -0.0213 /
-0.0227. Threshold +0.05. The learned selector is slightly worse than the fixed breadth-first order.

## Content-Use: PASS

`CE_ablated - CE_normal` on the same ACTIVE B8 paths: pooled **+0.0369** nats, 95% CI [+0.0185, +0.0553], per seed +0.0577 /
+0.0277 / +0.0252, so every seed is positive (rule: mean > 0, CI wholly > 0, every seed > 0; no magnitude threshold).
Normal vs ablated top-1 (per seed): 0.501 vs 0.513, 0.536 vs 0.544, 0.512 vs 0.529; CE 2.312 vs 2.369, 2.276 vs 2.304,
2.339 vs 2.365. **Removing the real queried content lowers the log-likelihood of the correct target but slightly RAISES top-1.**
Content is therefore used (the model's distribution depends on it, in the right direction on average), but the use is small
(0.037 nats against a CE of about 2.3) and does not translate into top-1 gains. Attribution ratio
`(normal_B8 - ablated_B8)/(normal_B8 - B0)` is not meaningful (null): `normal_B8 < B0`. Positions whose top-1 changes under
ablation, and the B2/B4 secondary content effects, are in `v35-run-seed*-eval-u0800.json`
(`query_content_ablation_active_paths`) and `v35-gates.json` (`ablation_B2_B4_secondary`).

## Selector / process diagnostics (ACTIVE B8, KQRvK M3, 750 positions x 8 queries per seed)

| seed | proof-adm. | refute-adm. | off-target | first query correct root | queries reducing residual | proofs complete at B8 | ideal Q* <= 8 |
|---|---|---|---|---|---|---|---|
| 5101 | 0.283 | 0.215 | 0.502 | 0.551 | 0.283 | 23 / 750 | 0.432 (324) |
| 5102 | 0.290 | 0.208 | 0.501 | 0.573 | 0.290 | 32 / 750 | 0.432 (324) |
| 5103 | 0.267 | 0.218 | 0.515 | 0.560 | 0.267 | 19 / 750 | 0.432 (324) |

Compared with the accepted P5.2 numbers (not a gate): the proof-admissible share rose from about 0.21-0.22 to 0.27-0.29, and the
refute-admissible share is now substantial (~0.21), as the learner enters wrong branches and the oracle labels them. Proof
completion on M3 remains 2.5-4.3% against the 43.2% ceiling (P5.2: 28-39/750), so on-policy relabelling improved the
selector's targeting a little but did not produce a search that completes proofs. Full per-position/depth/branch diagnostics
are in the committed evaluation files.

## Classification

Gate I (historical) PASS; Content-Use PASS; Gate II FAIL; Gate III FAIL; Gate VI PASS -> **PARTIAL - CONTENT**: the model uses the
returned state content, but B8 does not improve the task. Stop before HOLDOUT_C. No second DAgger iteration, no LR screen, no
extra updates, no B16 (the B16 diagnostic is conditional on Content-Use + II + III + VI all passing).

## What is established and what is not

- Established: on-policy training removes the teacher query-pattern leak and the model's output now depends on queried content
  (Content-Use passes on all three seeds, with an exact rollout/replay-parity training path). The detected shortfall is in
  turning that content into a better root move, and in selection: learned ACTIVE does not beat FIXED.
- Not established: that queries help at all (Gate II is negative); selective information acquisition; any extrapolation.
- Caveat (interpretation, not tested): B0 is trained in the same updates and improved by about 0.08 top-1; part of the
  deficit of B8 relative to B0 may be a capacity/optimisation trade-off between budgets rather than harm from queries
  per se. V3.5 contains no experiment that separates these.
- Not run: B16/Gate V, HOLDOUT_C, any additional training.

## V4 design diagnosis (memo only; nothing implemented)

The evidence supports the pre-registered V4 direction rather than another rescue: (1) the content path works but is weak
(+0.037 nats) and top-1 moves the wrong way under ablation, so integration should be structurally content-gated (ablated or
zero content = identity planner update) and evidence should be an explicit message generated from the returned state; (2)
ACTIVE queries remain ~50% off-target even after on-policy relabelling, with 2.5-4.3% proof completion, so the selector
needs a better learning signal than uniform-over-`A(S)` NLL, for example predicted query utility / residual reduction, with
learner-induced trajectories from the start; (3) keep one-edge StateQuery semantics and exact accounting; (4) make the
query-content counterfactual and an explicit B8-vs-B0 same-weight comparison first-class qualification tests during
training, not post-hoc; (5) consider decoupling the B0 and query-budget objectives so budget cannot trade off against B0.
