# Chimera V2 — experiment ledger

Branch `experiment/hp-r15-h3-integration`. Architecture contract: `HP_X2_ARCHITECTURE.md`.
Labels: MEASURED / INFERRED / NOT RUN. Nothing below is a result until it is under a
MEASURED heading with a log path.

## X2-PRE — pre-registration (committed BEFORE any confirmation-set evaluation)

### Question
Does exact progressive tool information (root facts → successor observations →
opponent-reply sets) let a *bounded gated neural planner* choose the forcing first move of
a mate-in-2 better than (a) seeing only root facts and (b) seeing all the same
information in one pass?

### Task and data (`recur64 x2 gen`, `docs/HP_X2_ARCHITECTURE.md`)
* White to move, pawnless KQK / KRK / KQQK / KQRK / KRRK, `fresh_no_history_v1`.
* No mate in one; ≥ 1 first move forces mate in two; correct fraction ≤ 0.15; at least
  one correct move shares its CandidateFactsV1 vector with an incorrect move.
* Splits: train 5 × 2400, tune 5 × 52, confirm 5 × 52; separate seeds
  (20262001 / 20262002 / 20262003); exact-FEN AND 8-symmetry-canonical disjoint from each
  other and from the V1 evaluation/training data. Confirm/tune/train labels are all
  re-verified 100 % through `GameState`. Chance top-1 = mean(correct/legal) is reported per
  set.
* MEASURED pool limits (`runs/x2/gen.log`, not committed): KQK and KRK have only a few
  hundred distinct symmetry classes satisfying the filters (sampled unique KQK ≈ 560, KRK
  ≈ 440), so their TRAIN counts are capped at 300 each (`--train-small-family-cap`); the
  other three families supply 3800 each: train = 12 000 (300/300/3800/3800/3800). Tune and
  confirm stay balanced at 52 per family (260 each). The root-fact ambiguity filter almost
  never rejects (quiet moves share all-zero fact vectors): non-ambiguous rejections were 0–30
  per thousands of candidates. Mean chance top-1: train 0.0589, tune 0.0542, confirm 0.0590.
  Capacity audit: max legal 60 (train), 54 (tune), 58 (confirm); max replies 8 everywhere;
  overflow 0; non-fresh 0.
* Capacity: `w_cap = 64`, `r_cap = 16`; the audit is recomputed from the positions at the
  start of every train/eval/bench and the run refuses on any overflow.

### Variants (visual OFF, 2 seeds each: 1, 2)
| | schedule | training budgets |
|---|---|---|
| A root-only | `root_only` | fixed T=1 |
| B all-info one-pass | `all_at_once` | fixed T=1 (Replies horizon at T1) |
| C progressive | `progressive` | `uniform_1_4_v1`: each update has one micro-batch per T=1..4, final-readout-only loss (`budget_final_v1`) |

Same data, optimizer, update count, seeds, loss and micro-batching for all variants.

### Loss (chosen before results)
Policy cross-entropy to the uniform-over-correct target, plus **0.1 ×** WDL cross-entropy
toward "win" (every exact mate-in-2 root is a win). Policy is the primary objective.

### Optimization
AdamW (project default), warm-up 30 updates, one LR for every budget. LR chosen from
{3e-5, 1e-4, 3e-4} on the **tune** set by best mean tune top-1 at the largest evaluated
budget of the progressive variant (for A/B: T=1); ties broken by lower tune CE. Train loss
alone is never used. The selected LR is applied to all three variants.

### Primary evaluation
Same weights, confirm set, T=1..4 for C; T=1 for A and B. Metrics per position: exact
top-1 (argmax ∈ correct set), probability mass on the correct set, CE to the uniform
target, entropy; planner RMS / delta / gate / attention shares.

### Decision rules (position-clustered paired bootstrap, 2000 deterministic resamples,
on seed-averaged per-position differences; "every seed positive" uses per-seed mean
differences)
* **Q1 tool-use signal**: progressive T3 or T4 − progressive T1 on exact top-1: mean > 0,
  95 % CI wholly > 0, both seeds > 0. (Two comparisons are examined; this is reported, not
  hidden.) Mass is reported alongside.
* **Q2 iterative-integration signal**: progressive T4 − all-info T1 on exact top-1: mean > 0,
  CI wholly > 0, both seeds > 0. A tie is "PARTIAL GO – TOOL": tool information works,
  recurrent integration has not earned itself.
* **Q3**: progressive T4 − T3; a clean positive is evidence of post-acquisition neural
  refinement; a tie means the final step is redundant (not a failure if Q2 holds).
* **Planner stable**: T1..T4 planner RMS finite and max/min < 4 for every seed.
* **Outcome**: FULL GO = Q1 ∧ Q2 ∧ stable. PARTIAL GO – TOOL = Q1 ∧ ¬Q2. PARTIAL GO –
  PLANNER = ¬Q1 ∧ progressive beats root-only (T1/T3/T4 vs A) ∧ stable (family breakdown
  reported). NO-GO otherwise, or if the planner cannot train stably after one bounded
  optimization check.
* **One rescue only** (structured process supervision: an intermediate head at T2/T3 for the
  fraction of replies with a mating continuation / all-replies-covered, final head still
  the forcing root move; normalized scales; one bounded comparison) — allowed only if the
  progressive model has the information but fails to use it. No further rescues.
* Visual A/B (off vs on, identical setup) only after a TOOL-USE SIGNAL.

### Compute honesty
Training reads a **cached** Replies-horizon world model (`tool_compute_mode = cached`,
recorded in `experiment.json`); it is capability training only. Every latency /
compute-frontier number comes from LIVE horizon-correct execution (`x2 eval`, `x2 bench`).
Cached and live numbers are never compared.

### Not done (by design)
No self-play, no PUCT, no retrieval, no external engine/data, no replay mixing.
The model is an analysis model under `fresh_no_history_v1`; it must not be used as a
self-play evaluator.

## Measured results
*(none yet — see `HP_X2_BUILD_RESULTS.md` for engineering gates)*

## X2-E1 — primary V2 confirmation (MEASURED; evidence in `docs/evidence/x2/`)

Rules were frozen in X2-PRE (commit 6c4800a) before any confirm evaluation. LR screen on
tune, progressive, 300 updates, seed 1 (`prog-lr*.log`): tune top-1 at T4 = 0.262 (3e-5),
1.000 (1e-4), 1.000 (3e-4); tie broken by CE 0.4443 < 0.4547 → **3e-4** for every variant.
All six runs: 300 updates × 128 positions, seeds 1 and 2, visual off, cached tool mode
for training, LIVE tool execution for evaluation.

Confirm set (260 positions, chance top-1 0.0590), seed-averaged:

| | T1 | T2 | T3 | T4 |
|---|---|---|---|---|
| A root-only | 0.608 | – | – | – |
| B all-info one pass | 1.000 | – | – | – |
| C progressive | 0.552 | 0.660 | 1.000 | 1.000 |

* Q1 (T3/T4 − T1, top-1): +0.448, CI [+0.394, +0.502], both seeds positive → **SIGNAL**
  (mass +0.649).
* Q2 (progressive T4 − all-info T1): 0.000, CI [0,0] → **no iterative-integration signal**
  (mass −0.0002).
* Q3 (T4 − T3): exactly 0 on every position → the final integration step is redundant.
* Progressive T1 is *below* root-only (−0.056, CI [−0.090, −0.023]); T3/T4 − root-only
  +0.392 (SIGNAL). Planner stable (RMS ratio < 4 in both seeds).
* **OUTCOME: PARTIAL GO – TOOL.** The exact world-model information is what solves the task
  and a progressively-trained network exploits it at T3; a one-pass network exploits the
  same information equally well. Recurrent integration has not earned itself.

Caveats (INFERRED): the reply-set records include next-player mate counts, so the task is
close to solvable from a single feature once replies are visible (ceiling at 1.000 for both
B and C); this task cannot discriminate recurrence from one-pass integration. No process
supervision rescue was run: the rescue is only permitted when the information is available
but unexploited, and here it is fully exploited. Visual A/B not run (allowed after a tool
signal; nothing indicates it is needed). Two seeds, 300 updates per run.

Compute (MEASURED, `bench.json`, 256 confirm positions, native/WASM byte-identical at every
horizon): Root 41.7 µs/pos native (33 root moves applied, 0 reply work), Successor 43.1,
Replies 536 µs/pos (98 reply moves applied, 3530 next-player moves enumerated); WASM
0.50 / 1.43 / 26.5 ms/pos. GPU forward per batch of 32 (steady): T1 31.0 ms, T2 39.2,
T3 58.2, T4 63.5; tool CPU 0.7 / 0.7 / 11.5 / 11.5 ms; board encoder runs 1 at every T;
peak VRAM 515 MiB forward, 1607 MiB training (batch 32 micro-batches).
