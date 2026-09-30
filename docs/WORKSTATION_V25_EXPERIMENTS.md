# Workstation V2.5 — experiment ledger

Each entry: QUESTION / HYPOTHESIS / PRE-REGISTERED RULE / CONFIG / DATA DIGEST / WALL /
PEAK VRAM / MEASURED / INFERRED / DECISION / NEXT ACTION. Rules are written BEFORE the
run. Entries are appended; they are never rewritten after results are seen.

## Pre-registered gates (fixed before any training)
- P0: all 20 architecture-correctness items must pass. No science if P0 fails.
- P0.5: if CandidateFacts costs >20% of end-to-end self-play evaluation wall, optimize
  without changing semantics.
- P0.6: training layout must be finite with peak VRAM <= 12 GB; failed cells stay visible.
- P1: CF, seed 1, LR in {3e-5, 7.5e-5, 1.5e-4, 3e-4}, ~75–100 updates, selection on TUNE
  policy CE / correct-move mass / stability (never train loss alone).
- P2 (L, C0, CF × 2 seeds, 400 updates, policy-only):
  - Q1 FACTS PATH GO iff CF M1 top-1 >= 0.95 AND CF > C0 paired 95% CI entirely > 0 AND both seeds positive.
  - Q3: CF M2 top-1 >= 0.75 and M3 top-1 >= 0.55 (both seeds at/near floors with pooled CI clear of chance).
  - One extension only: 400 → 800 updates if train/tune still improve with no overfit gap and healthy gradients. Then CONFIRM once more; if still short, STOP (no self-play).
  - Thresholds may be amended only BEFORE CONFIRM is seen.
- P3 (frozen conversion suite, argmax from ply 0, no noise, solver off): proof-trained CF
  beats corrected F10 by >= 10 wins/128 TARGET starts (searched), paired bootstrap 95% CI > 0,
  no catastrophic HEAVY regression; M3 CONFIRM playout conversion >= 80%.
- P3.5: one M4/M5 extension only if M1–M3 strong, M3 playout strong, transfer weak.
- P4: only if P3 passes; 2-cycle smoke at most; curriculum conversion < 70% for two
  consecutive cycles is a hard stop.

---

## E-DATA-1 — ProofTargetsV1 scale (PRE-REGISTERED before any dataset was generated)

QUESTION: how large can the hard-disjoint M1/M2/M3 datasets be under the pre-registered
filters (correct <= 15% of legal moves and CandidateFacts ambiguity for M2/M3; canonical
dedup; hard-disjoint splits)?

HYPOTHESIS: the requested ~1000/100/100 per (family, band) is not available for the
two-piece families.

MEASURED (exact exhaustive enumeration of every placement, symmetry-canonical classes that
pass the filters; `docs/evidence/v25/proof/proof-pool-*.json`):

| Family | M1 | M2 | M3 |
|---|---:|---:|---:|
| KQvK | 306 | 576 | 1,076 |
| KRvK | 189 | 532 | 438 |
| KQQvK | 95,649 | 174,163 | 4,409 |
| KQRvK | 111,273 | 306,595 | 211,215 |
| KRRvK | 41,612 | 122,082 | 108,086 |

Six of fifteen cells are below the 1,200 needed for 1000/100/100 (KQvK M1/M2/M3, KRvK
M1/M2/M3). The generator benchmark gate measured ~185–15,000 positions/s by band with a
projected full-scale wall under one minute, so the limit is pool size, not time.

PRE-REGISTERED RULE (all filters kept; nothing relaxed):
- A cell whose eligible pool P >= 1200 uses TRAIN/TUNE/CONFIRM = 1000/100/100.
- A cell with P < 1200 uses its whole pool split 80/10/10: TUNE = CONFIRM = floor(0.10 P),
  TRAIN = P - 2 floor(0.10 P). No position is reused across splits.
- Splits are selected from the exact pool by independent seeded shuffles (CONFIRM, then
  TUNE from the remainder, then TRAIN from the remainder), so they are hard-disjoint by
  exact FEN and canonical class by construction, and re-verified.
- Family x depth balance is therefore approximate, not exact. Results are reported pooled
  by depth AND by family; per-family confirm counts for KRvK M1 are small and are flagged.

EXPECTED SIZES: TRAIN 11,501, TUNE 1,208, CONFIRM 1,208.
DECISION: proceed. This is a scale decision inside the "approximately" latitude, not a
change to the exactness contract; if it should be treated as a material contract change the
owner can regenerate under a different rule in minutes (generation is cheap).

---

## Disclosure — CONFIRM touched once by a toy-model pipeline smoke test
While validating the `proof train / eval / compare` pipeline, a tiny throwaway model
(width 32, 2 blocks; 30 updates; CF and C0 variants) was evaluated on CONFIRM. Result:
CF M1 top-1 0.997 vs C0 0.10; M2/M3 near chance. It was a plumbing check only: it used no
V2.5-geometry model, informed no threshold, LR or selection, and the CF-vs-C0 M1 gap it
showed is the expected mechanical effect of the `mate` fact. It is disclosed here because
CONFIRM is otherwise reserved for the final evaluations. From this point CONFIRM is
evaluated only by the pre-registered P2 runs (with `--eval-confirm`) and the single
permitted extension.

## E-P1 — LR / training microscreen (PRE-REGISTERED before it was run)

QUESTION: which peak learning rate should the V2.5 fixed-data runs use (a new architecture;
the mainline 3e-4 is not assumed)?

HYPOTHESIS: a lower rate than 3e-4 is at least as good at 27M parameters, as the HP branch
found at 15M (there 7.5e-5). Not assumed.

CONFIG: CF (`configs/v25/candidate-v25-cf-cuda.toml`), seed 1, FP32, fixed TRAIN/TUNE
(digests in BUILD_RESULTS), policy-only exact-target loss (no WDL), effective batch 256
(64x4), 90 updates, warmup 9 (10%), cosine to 0 over the 90 updates, LR in
{3e-5, 7.5e-5, 1.5e-4, 3e-4}. No CONFIRM. Evaluate TUNE at update 45 and 90.

PRE-REGISTERED RULE:
1. A run is STABLE iff every update has finite loss and gradient norm AND its final
   TRAIN loss is below its first-update loss (no divergence).
2. Among STABLE runs, select the lowest final TUNE policy cross-entropy.
3. If the two lowest TUNE CEs are within 0.02 nats, prefer the run with the higher TUNE
   correct-set mass; if still tied, prefer the lower LR.
4. Train loss alone is never a selection criterion.
Known bias, stated in advance: short screens favor larger rates. The selected LR is used
for L, C0 and CF in P2; if the selected LR is the largest grid point, that is reported as
a boundary result, not silently accepted as optimal.

### E-P1 RESULT
CONFIG: as pre-registered (CF, seed 1, 90 updates, effective batch 256, policy-only).
DATA DIGEST: TRAIN 8d6440d2..., TUNE d48ccf86... (CONFIRM not evaluated).
WALL / VRAM: ~0.98 s/update; ~2.6 GB peak (500 ms sampling).
MEASURED (TUNE after 90 updates):

| LR | stable | TUNE CE | TUNE correct mass | TUNE top-1 | M1 / M2 / M3 top-1 | CE at update 45 -> 90 |
|---|---|---:|---:|---:|---|---|
| 3e-5 | yes | 3.179 | 0.088 | 0.306 | 1.00 / 0.01 / 0.04 | 3.286 -> 3.179 |
| 7.5e-5 | yes | 2.792 | 0.198 | 0.308 | 1.00 / 0.01 / 0.04 | 3.049 -> 2.792 |
| 1.5e-4 | yes | 2.568 | 0.329 | 0.378 | 1.00 / 0.09 / 0.16 | 2.673 -> 2.568 |
| 3e-4 | yes | 2.536 | 0.338 | 0.375 | 1.00 / 0.08 / 0.16 | 2.562 -> 2.536 |

Chance top-1 is 0.06. Untrained-ish start loss was 3.515 in every run.
RULE APPLIED: all four stable; lowest TUNE CE is 3e-4 (2.5363) vs 1.5e-4 (2.5682); the gap
0.032 exceeds the 0.02 tie band, so no tie-break applies.
DECISION: LR = 3e-4 for P2 (L, C0, CF).
BOUNDARY RESULT: 3e-4 is the LARGEST grid point, so the optimum may lie higher; this is a
selection under the pre-registered grid, not a claim of optimality. The 1.5e-4 vs 3e-4
gap is small (0.032 nats) and the 90-update horizon favors large rates.
INFERRED (not tested): M1 is already solved (1.00) at every LR because the `mate` fact
exposes it directly; M2/M3 are only just leaving chance after 90 updates, so the P2
horizon (400) is where the science question is decided.
NEXT ACTION: pre-registered P2 below.

## E-P2 — fixed-data architecture ablation (PRE-REGISTERED before it was run)

QUESTIONS: Q1 do exact facts work in this model (CF vs C0); Q2 do move tokens help beyond
capacity (C0 vs L on M2+M3); Q3 can the one-pass model learn deeper exact technique.

CONFIG: cells L (`large-legacy-cuda`), C0 (`candidate-v25-c0-cuda`), CF
(`candidate-v25-cf-cuda`); seeds 1 and 2; LR 3e-4; warmup 40 (10%); cosine to 0 over 400
updates; effective batch 256 (64x4); policy-only exact targets; FP32; TUNE evaluated every
50 updates; CONFIRM evaluated once at the end (`--eval-confirm`).

GATES (unchanged from the plan; thresholds may NOT be amended after CONFIRM is seen):
- Q1 FACTS PATH GO iff CF M1 top-1 >= 0.95 AND CF > C0 with the paired 95% CI entirely
  above 0 AND both seeds positive.
- Q3: CF M2 top-1 >= 0.75 and M3 top-1 >= 0.55; both seeds at/near the floors with a pooled
  CI clearly above chance.
- Q2 is reported cleanly (C0 vs L on M2+M3); it is not required to be positive.

THE ONE EXTENSION, defined now: if CF misses the M2/M3 gate but (a) TUNE CE fell by at
least 0.01 nats between update 350 and update 400, (b) TRAIN-slice CE is not more than 0.15
nats below TUNE CE (no overfit gap), and (c) every update had finite loss and gradient norm,
then run ONE extension: a NEW from-scratch run of 800 updates (cosine over 800, warmup 80,
same seed and LR) for CF, and re-evaluate the untouched CONFIRM. It is a fresh run, not a
continuation of a decayed schedule, so the schedule is not a confound. If the gate is still
missed: STOP, no self-play. C0 and L are extended identically if CF is, so the comparison
stays matched.


---

# ADDENDUM - POST-P1a HYGIENE AND P1b REQUALIFICATION (before P2)

## E-P1a - ORIGINAL LR SCREEN (preserved, not modified)
The section above titled "E-P1 RESULT" is the ORIGINAL LR SCREEN and is hereafter called
**E-P1a**. Its evidence (`docs/evidence/v25/p1/`) and measurements are unchanged.

P1a is a valid measured result under its actual contract: the ORIGINAL ProofTargets
TRAIN/TUNE split, uniform-position epoch shuffling, CF seed 1, 90 updates, LR grid
{3e-5, 7.5e-5, 1.5e-4, 3e-4}, no CONFIRM evaluation. It selected 3e-4 by the
pre-registered CE rule (final TUNE CE 3.179 / 2.792 / 2.568 / 2.536 across the grid);
M1 = 1.00 at every LR; M2/M3 moved materially only at the upper two LRs; 3e-4 was the top
edge of the grid and was recorded as a boundary result.

P1a answered: "which LR looked best for CF under the ORIGINAL split assignment and uniform
sampler?" Answer: 3e-4, boundary result. It is superseded ONLY as the LR-selection
authority for P2, because the hygiene pass below changes the split assignment and the
training sampling distribution. Its substantive findings remain evidence: all four LRs were
stable; CandidateFacts solve M1 quickly; M2/M3 needed more optimization; 3e-4 won the
original grid; 3e-4 was a boundary result. P1b (below) answers a different question under a
different optimization distribution; if the two differ, that is not a contradiction.

## Retired split assignment
The original CONFIRM was touched by the toy pipeline smoke (disclosed above). Because the
small KQvK/KRvK cells consume their entire eligible pools under the 80/10/10 rule, CONFIRM
cannot be replaced without reallocating TRAIN and TUNE, so the ENTIRE ORIGINAL SPLIT
ASSIGNMENT is retired. The exhaustive pools, filters, solver, families and depth
definitions are NOT retired; only the allocation into splits.

RETIRED AFTER P1a / BEFORE P2 (kept in history, not deleted):
- TRAIN   `8d6440d214e606a7f01ad0584d292cf6ac500495040bda96816fb9f599e74cab`
- TUNE    `d48ccf8616091437210e2c21f5ad33638a1847bd8220abf6c36b2f510c0185a3`
- CONFIRM `ed36378381eed0a05958c2302752b219a1d2b2f94996bb4e373e9a9510ccc101`

## Aborted P2 launch (disclosure)
P2 had been launched in the background before this addendum arrived. It was STOPPED on
receipt. One run (L, seed 1) had completed on the retired split, including an evaluation of
the RETIRED CONFIRM, and a second (C0, seed 1) was mid-training. Their outputs are
quarantined unread (`runs/v25/aborted/`, untracked) and are VOID as P2 evidence: they used
the retired split and the pre-hygiene contract. Nothing from them informed any decision.

## New split assignment (same pools, filters, rules; only the seeds changed)
| split | positions | seed | sha256 digest |
|---|---:|---|---|
| TRAIN | 11501 | 0x7A120001 | `1e5e121bc407e0d27b2956eac570c2d67485f8637b7d9e329d21e863427671d4` |
| TUNE | 1208 | 0x7A120002 | `f59d744a133e887d7c50a00923ee2d16b9bd7a4560330252a93574828bfc16e4` |
| CONFIRM | 1208 | 0x7A120003 | `8c4d2b723afb7f730d28a42f150935d0aab6ff6e98e25124c18febe71e0301df` |

Verified: 100% independent GameState audit (0 failures in all three splits); exact-FEN and
canonical-class disjoint; identical digests at 20 and 9 threads; digests differ from the
retired ones. Chance top-1 (M1/M2/M3): TRAIN 0.044/0.058/0.072, TUNE 0.044/0.060/0.072, CONFIRM 0.043/0.057/0.070.
Evidence: `docs/evidence/v25/proof-v2/proof-gen-report.json`.

replacement_confirm_status = "sealed_unseen" - with one precisely stated caveat: no model
has evaluated the new CONFIRM SET, and none will before the P2 commands. However, the new
and retired assignments draw from the SAME pools, so 47 of the 1,208 new CONFIRM positions
(3.9%; KQvK M1 6/30, M2 3/57, M3 16/107; KRvK M1 2/18, M2 14/53, M3 4/43; KQQvK M3 2/100)
were members of the retired CONFIRM that the disposable toy model evaluated once. That
model never trained on them and only aggregate numbers were read, so no label information
can reach a P2 model, but the set is not literally "never seen by anything". Also 252 new
CONFIRM positions were in the old TRAIN of the throwaway P1a models; P1a models are not
reused. `proof train --eval-confirm` and `proof eval --split confirm` now print
"CONFIRM DATASET IS BEING EVALUATED: <digest>" and append to `confirm-exposure.log` beside
the datasets (a hygiene guard, not a security mechanism). The log does not exist at P1b start.

## cell_balanced_v1 (sampler contract, code: `proof/sampler.rs`)
CELL = (family, mate_depth); 15 cells. A rotor visits the non-empty cells in sorted order;
global example g draws from cell g mod 15. Each cell owns an independent seeded shuffle
(seed = f(model seed, cell tag, local epoch)) consumed without replacement; on exhaustion
it reshuffles into its next local epoch. Nothing is duplicated on disk; oversampling small
cells is sampling, not extra unique data. The rotor runs across updates, so with effective
batch 256 (= 17x15 + 1) the cell receiving the extra example rotates and any two cells
differ by at most one example in the long run. The sequence is a pure function of (model
seed, sampler version, global example index). Tests: same seed reproduces, different seed
changes within-cell order but not the rotor's cell choice, all cells represented, exposure
equal within one, no duplicate within a local epoch, deterministic wrap and reshuffle,
family/depth fractions sum to one. The historical `uniform_v0` sampler is retained only to
reproduce P1a.

Expected exposure on the NEW TRAIN (400-update P2 = 102,400 examples = 6,826.7 per cell;
90-update P1b = 23,040 examples = 1,536 per cell). "uniform_v0 share" is what P1a's sampler
gave each cell.

| cell | eligible pool | TRAIN | TUNE | CONFIRM | uniform_v0 share | balanced share | local epochs @400 upd | @90 upd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KQQvK M1 | 95649 | 1000 | 100 | 100 | 8.69% | 6.67% | 6.8x | 1.5x |
| KQQvK M2 | 174163 | 1000 | 100 | 100 | 8.69% | 6.67% | 6.8x | 1.5x |
| KQQvK M3 | 4409 | 1000 | 100 | 100 | 8.69% | 6.67% | 6.8x | 1.5x |
| KQRvK M1 | 111273 | 1000 | 100 | 100 | 8.69% | 6.67% | 6.8x | 1.5x |
| KQRvK M2 | 306595 | 1000 | 100 | 100 | 8.69% | 6.67% | 6.8x | 1.5x |
| KQRvK M3 | 211215 | 1000 | 100 | 100 | 8.69% | 6.67% | 6.8x | 1.5x |
| KQvK M1 | 306 | 246 | 30 | 30 | 2.14% | 6.67% | 27.8x | 6.2x |
| KQvK M2 | 576 | 462 | 57 | 57 | 4.02% | 6.67% | 14.8x | 3.3x |
| KQvK M3 | 1076 | 862 | 107 | 107 | 7.50% | 6.67% | 7.9x | 1.8x |
| KRRvK M1 | 41612 | 1000 | 100 | 100 | 8.69% | 6.67% | 6.8x | 1.5x |
| KRRvK M2 | 122082 | 1000 | 100 | 100 | 8.69% | 6.67% | 6.8x | 1.5x |
| KRRvK M3 | 108086 | 1000 | 100 | 100 | 8.69% | 6.67% | 6.8x | 1.5x |
| KRvK M1 | 189 | 153 | 18 | 18 | 1.33% | 6.67% | 44.6x | 10.0x |
| KRvK M2 | 532 | 426 | 53 | 53 | 3.70% | 6.67% | 16.0x | 3.6x |
| KRvK M3 | 438 | 352 | 43 | 43 | 3.06% | 6.67% | 19.4x | 4.4x |

Scarce cells are oversampled heavily: KRvK M1 sees ~44.6 local epochs at P2 length
(vs 6.8 for a 1000-position cell). That is the intended trade (KQ/KR are central to the
later conversion question) but it raises memorization risk in exactly those cells; the
by-cell TRAIN-vs-TUNE gap is reported to make it visible.

## Metrics added
- Macro metrics on every evaluation: `macro_cell_*`, `macro_family_*`, `macro_depth_*`
  (top1, mass, ce, entropy): each non-empty group averaged separately, then group means
  averaged with equal weight; plus `by_cell`. Pooled metrics are unchanged. The
  pre-registered P2 gates stay on the ORIGINAL pooled depth metrics; macro metrics are
  mandatory diagnostics and cannot be used to redefine success after the result.
- `train_full_final` replaces `train_slice_final`. The old slice was the first 1,208 TRAIN
  positions and (as `by_family` showed) was a single family (KQvK n=1208); it is removed as
  a scientific diagnostic. Every run now evaluates ALL TRAIN positions (overall, by family,
  by depth, by family x depth, macro). P1a's `train_slice_final` fields are historical only.
- Evaluation files carry `model_seed` (from checkpoint metadata for saved models, the
  requested seed for fresh ones; never inferred from a filename) plus architecture,
  training updates, peak LR and sampler version where known.
- `proof compare --per-seed` pairs seeds by IDENTITY: both sides need unique model seeds
  and equal seed sets, else refusal (tests: same order pass, reversed order identical
  result, seed sets {1,2} vs {1,3} refused, duplicate seed refused, missing seed refused).

## P2 extension rule - CORRECTION (supersedes condition (b) ONLY)
Condition (b) of the E-P2 extension rule referred to the "TRAIN-slice CE", a diagnostic now
known to be biased. It is superseded by:
(b') the FULL-TRAIN pooled CE is not more than 0.15 nats below the TUNE pooled CE, AND
     no severe macro-cell overfit pattern: the macro-cell TRAIN CE must not beat the
     macro-cell TUNE CE by more than 0.25 nats.
Conditions (a) (TUNE CE falls by >= 0.01 between update 350 and 400) and (c) (all losses and
gradients finite) are unchanged. Extension semantics are unchanged: a FRESH from-scratch
800-update run (same LR, same seed, same data, same sampler; warmup 80; cosine over 800),
NOT a continuation of the finished 400-update cosine. The earlier text is not rewritten.

## E-P1b - FINAL-CONTRACT LR REQUALIFICATION (PRE-REGISTERED before it is run)
QUESTION: which LR governs the FINAL P2 contract (new splits, cell_balanced_v1)?
CONFIG: CF, seed 1, NEW TRAIN/TUNE, cell_balanced_v1, policy-only, effective batch 256
(64x4), 90 updates, warmup 9, cosine over 90, FP32; LR grid {3e-5, 7.5e-5, 1.5e-4, 3e-4}
(the grid is NOT expanded because P1a hit its upper edge: P1b requalifies the same planned
grid under the final contract; it is not a new search). CONFIRM is NEVER evaluated.
RULE (same as P1a): 1. STABLE iff every loss and gradient norm is finite and final TRAIN
loss < initial TRAIN loss. 2. Among stable runs the lowest final pooled TUNE CE wins.
3. If the two lowest are within 0.02 nats, the higher TUNE correct-set mass wins.
4. If still tied, the lower LR wins. Primary selection stays on pooled TUNE CE for
continuity with P1a; pooled, macro-cell, by-family and by-depth TUNE are all reported.
If P1b selects a different LR from P1a, P1b's LR is used for P2.


---

### E-P1b RESULT
CONFIG: as pre-registered (CF, seed 1, NEW TRAIN/TUNE, cell_balanced_v1, 90 updates,
effective batch 256, policy-only, FP32). CONFIRM not evaluated (no `confirm-exposure.log`;
`confirm.evaluated = false` in every run). Model seed 1 is recorded in each evaluation file.
DATA DIGEST: TRAIN 1e5e121b..., TUNE f59d744a... (new assignment).
WALL / VRAM: ~0.95-1.01 s/update; no VRAM problem.
Sampler check: every cell consumed exactly 1,536 examples (min = max over 15 cells);
KRvK M1 saw 10.0 local epochs in 90 updates.

MEASURED (TUNE after 90 updates; "gap" = TUNE CE minus full-TRAIN CE, pooled / macro-cell):

| LR | stable | pooled TUNE CE | pooled mass | pooled top-1 | macro-cell CE | macro-cell top-1 | M1 / M2 / M3 top-1 | gap |
|---|---|---:|---:|---:|---:|---:|---|---|
| 3e-5 | yes | 3.1739 | 0.0894 | 0.2964 | 3.0338 | 0.3398 | 1.000 / 0.002 / 0.020 | -0.003 / +0.003 |
| 7.5e-5 | yes | 2.7869 | 0.2024 | 0.2988 | 2.5969 | 0.3417 | 1.000 / 0.005 / 0.024 | +0.011 / +0.005 |
| 1.5e-4 | yes | 2.5750 | 0.3278 | 0.3576 | 2.3764 | 0.3911 | 1.000 / 0.078 / 0.116 | +0.020 / +0.008 |
| 3e-4 | yes | 2.5467 | 0.3380 | 0.4031 | 2.3382 | 0.4788 | 1.000 / 0.151 / 0.171 | +0.016 / +0.008 |

By family top-1 at 3e-4: KQQvK 0.357, KQRvK 0.353, KRRvK 0.367, KQvK 0.474, KRvK 0.632
(KRvK TUNE has only 114 positions, so that figure is noisy). At 1.5e-4 the family spread is
the opposite (0.41 / 0.40 / 0.41 / 0.23 / 0.19), i.e. the ranking of families is not stable
across LRs at this short horizon.

RULE APPLIED: all four runs stable; lowest pooled TUNE CE is 3e-4 (2.5467) vs 1.5e-4
(2.5750); the gap 0.0283 exceeds the 0.02 tie band, so no tie-break applies.
DECISION: LR = 3e-4 is authorized for P2.
BOUNDARY: 3e-4 is again the LARGEST grid point. The grid was deliberately not expanded;
this is a selection under the planned grid, not a claim of optimality.

### E-P1a vs E-P1b (different optimization distributions; not a contradiction)
| | P1a (old split, uniform sampler) | P1b (new split, cell_balanced_v1) |
|---|---|---|
| winner | 3e-4 | 3e-4 |
| final pooled TUNE CE at 3e-4 / 1.5e-4 | 2.536 / 2.568 | 2.547 / 2.575 |
| gap between the best two | 0.032 | 0.028 |
| M1 top-1 | 1.00 at every LR | 1.00 at every LR |
| M2 / M3 top-1 at 3e-4 | 0.083 / 0.158 | 0.151 / 0.171 |
MEASURED: both screens agree on the ordering of the four LRs and on M1 being solved; the
pooled CE curves are nearly identical. INFERRED (not tested): the changed split and
sampler did not materially move the LR ordering at 90 updates. Overfit signals are small:
full-TRAIN vs TUNE pooled CE gap is within +0.02 nats at every LR and the macro-cell gap
within +0.01, so no memorization of the heavily-oversampled scarce cells is visible yet
at this horizon; that is re-checked at 400 updates by the corrected extension rule.
Honest note: at 90 updates M2/M3 (0.15 / 0.17) are far below the P2 gates (0.75 / 0.55);
the science question is decided by the 400-update P2 runs, not by this screen.

## E-P2 - FINAL CONTRACT (supersedes the launch parameters of the earlier E-P2 block; the
## gates and the corrected extension rule stand exactly as written)
MODELS / SEEDS: L (`large-legacy-cuda`), C0 (`candidate-v25-c0-cuda`), CF
(`candidate-v25-cf-cuda`); model seeds 1 and 2.
TRAINING: 400 updates, policy-only, effective batch 256 (64x4), FP32, warmup 40, cosine
over 400, LR 3e-4 (E-P1b), `cell_balanced_v1`, TUNE evaluated every 50 updates.
DATA: NEW TRAIN / TUNE / CONFIRM (digests above).
CONFIRM: evaluated only at each run's final 400-update endpoint via `--eval-confirm`
(plus, if triggered, the single fresh 800-update extension). These are the first authorized
V2.5 evaluations of the replacement CONFIRM; each prints the exposure guard and is logged.
GATES (unchanged, pooled depth metrics): Q1 facts path: CF M1 top-1 >= 0.95 AND CF > C0 with
the paired 95% CI wholly above 0 AND both seeds positive (paired by model-seed identity).
Q3: CF M2 top-1 >= 0.75 and M3 top-1 >= 0.55, both seeds at/near the floors with pooled
evidence clearly above chance. Q2: C0 vs L on M2+M3 reported cleanly, not a gate.
DIAGNOSTICS REPORTED BESIDE THE GATES (cannot redefine success): macro-cell / family /
depth metrics, by-cell tables, full-TRAIN-vs-TUNE gaps, per-cell exposure.
EXTENSION: the corrected rule above ((a), (b'), (c)); a FRESH 800-update run, never a
continuation.
STATUS: E-P1b result and this contract are committed BEFORE any P2 run. No P2 run for the
replacement split has been launched.

---

## P2 statistics tooling patch (reporting only)
Per-seed comparisons now report the exact pre-registered groups (all / M1 / M2 / M3 /
M2+M3) for each metric (top1, mass, neg_ce), paired by model_seed identity, under
`<metric>.per_seed.seed_N.<group>`, using the SAME grouping and bootstrap function as the
pooled comparison. A seed's overall difference can no longer be mistaken for that seed's
M1 difference (the Q1 gate's "both seeds positive on M1" reads `top1.per_seed.seed_N.M1`).
No science contract, threshold, data, model, sampler, LR, bootstrap method or resample
count changed; the earlier P2 pre-registration is not rewritten.

Statistics patch commit: `1f03cfca8c051e5e74795fa388af6b9167d956a8` (P2 was launched only after this commit existed).

## P2 launch attempt 1 - infrastructure failure (documented; no result)
The first attempt of the frozen P2 contract (launched from commit `238f2f0`) failed
immediately in the RUNNER SCRIPT, not in the experiment: the PowerShell runner set
`$ErrorActionPreference = "Stop"`, and Windows PowerShell 5.1 converts the first stderr line
of a native program into a terminating `NativeCommandError`. The trainer logs progress to
stderr, so the script aborted at `update 0` of L seed 1. Consequences, verified afterwards:
no trainer process left running, GPU idle, `L-s1` output directory empty, log empty, NO
CONFIRM evaluation occurred (`confirm-exposure.log` absent), no metrics produced.
Remedy: the runner now redirects both streams with `cmd /c ... > log 2>&1` and does not use
`Stop`. The SAME cell is rerun from scratch with the SAME configuration; no hyperparameter,
data, sampler, seed, or order change. (A stale `runs/p2-C0-s1.log` from the earlier aborted
launch on the retired split was moved into `runs/v25/aborted/`.)


---

## E-P2 RESULT (frozen contract; run from commit `238f2f0` + doc-only commit `6e8c564`)
CONFIG: exactly the committed final P2 contract. L (`large-legacy-cuda`), C0, CF; model
seeds 1 and 2; LR 3e-4; 400 updates; warmup 40; cosine over 400; effective batch 256
(64x4); policy-only; FP32; `cell_balanced_v1`; NEW TRAIN `1e5e121b...` / TUNE `f59d744a...` /
CONFIRM `8c4d2b72...`; TUNE every 50 updates; CONFIRM only at update 400.
RUN HISTORY: attempt 1 failed in the runner script (documented above, no result); attempt 2
ran all six cells to completion, exit 0 each (`STATUS.txt`). CONFIRM hygiene: the exposure
log has exactly six entries, one per final endpoint, each with model name and seed; no
intermediate checkpoint evaluated CONFIRM. Exposure: every cell 6,826-6,827 examples;
KRvK M1 44.6 local epochs, 1,000-position cells 6.8.

Checkpoints (untracked, `runs/v25/p2/<cell>/checkpoint`; all with TRAIN/TUNE/CONFIRM digests above):
| cell | model_id | params | wall | peak VRAM |
|---|---|---:|---:|---:|
| L-s1 | 0372148dbc4f1bae51153432ca11f3494718348f7ac7c806c533c5befc23d284 | 26809944 | 348 s | 2551 MB |
| L-s2 | 48f3fb956d0a06283af651ee3ff4756b1d6df076ddcac3e6f5a943594be6a157 | 26809944 | 353 s | 2552 MB |
| C0-s1 | 04cf5c3fe7c41639a8b0b916033f6106c91b13334377d21878c21103d673ba96 | 27469204 | 366 s | 2679 MB |
| C0-s2 | ee04bec7739489f167cef5e33d938b3d73814b2f4124acf9e76c4df8462c25be | 27469204 | 359 s | 2679 MB |
| CF-s1 | 56d482b17ce2c56be094e6a3e24b7b312e07d7f2a9dab8709b1729cc118eeb5c | 27469204 | 367 s | 2680 MB |
| CF-s2 | 3b498cb3793b1e426bb267ab738c5502e6fc39d1a8cad12e8cf726db139ccd4b | 27469204 | 372 s | 2680 MB |

MEASURED - absolute CONFIRM, pooled (chance top-1 M1/M2/M3 = 0.043 / 0.057 / 0.070):
| model | seed | M1 top-1 | M2 top-1 | M3 top-1 | mass | CE | macro-cell top-1 | macro-cell CE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L | 1 | 0.629 | 0.639 | 0.624 | 0.493 | 1.881 | 0.681 | 1.653 |
| L | 2 | 0.693 | 0.661 | 0.638 | 0.527 | 1.813 | 0.712 | 1.572 |
| C0 | 1 | 0.632 | 0.610 | 0.589 | 0.452 | 2.008 | 0.658 | 1.759 |
| C0 | 2 | 0.626 | 0.634 | 0.620 | 0.473 | 1.947 | 0.671 | 1.698 |
| CF | 1 | 1.000 | 0.690 | 0.671 | 0.658 | 1.410 | 0.799 | 1.222 |
| CF | 2 | 1.000 | 0.688 | 0.647 | 0.654 | 1.436 | 0.791 | 1.244 |

CF CONFIRM by family top-1 (seed 1 / seed 2): KQQvK 0.81/0.81, KQRvK 0.69/0.67, KQvK 0.81/0.79,
KRRvK 0.76/0.77, KRvK 0.85/0.83. Weakest cell: KQRvK M3 (0.39 / 0.36); strongest: KRvK M3
(0.93 / 0.91, but n=43 and a heavily oversampled cell, so noisy).
Pooled CE, TRAIN / TUNE / CONFIRM: CF seed 1 1.296 / 1.405 / 1.410, seed 2 1.329 / 1.445 / 1.436;
L 1.669 / 1.822 / 1.881 and 1.642 / 1.796 / 1.813; C0 1.812 / 1.923 / 2.008 and 1.737 / 1.915 / 1.947.
No train/confirm overfit gap is visible in any model.

PAIRED COMPARISONS (B minus A, CONFIRM, paired bootstrap 95% CI; full tables with mass, CE and
per-seed groups in `docs/evidence/v25/p2/compare-*.json`):
| comparison | group | top-1 diff [CI] | seed 1 | seed 2 |
|---|---|---|---:|---:|
| CF - C0 | M1 | +0.371 [+0.326, +0.417] | +0.368 | +0.374 |
| CF - C0 | M2 | +0.067 [+0.030, +0.105] | +0.080 | +0.054 |
| CF - C0 | M3 | +0.054 [+0.021, +0.088] | +0.082 | +0.027 |
| CF - C0 | M2+M3 | +0.060 [+0.035, +0.085] | +0.081 | +0.040 |
| C0 - L | M1 | -0.032 [-0.062, -0.001] | +0.003 | -0.066 |
| C0 - L | M2+M3 | -0.027 [-0.047, -0.007] | -0.033 | -0.022 |
| C0 - L | M2 | -0.028 [-0.059, +0.001] | -0.029 | -0.027 |
| C0 - L | M3 | -0.027 [-0.054, +0.000] | -0.036 | -0.018 |
| CF - L | M1 | +0.339 [+0.295, +0.384] | +0.371 | +0.307 |
| CF - L | M2+M3 | +0.033 [+0.008, +0.058] | +0.049 | +0.017 |
| CF - L | M2 | +0.039 [+0.000, +0.079] | +0.051 | +0.027 |
| CF - L | M3 | +0.028 [-0.006, +0.061] | +0.047 | +0.009 |
Correct-set mass and -CE agree in sign with top-1 for every pooled group of CF-C0 and C0-L
(C0 is worse than L on mass by 0.040 [0.032, 0.048] and on CE by 0.142 on M2+M3).

GATES (thresholds unchanged, pooled depth metrics):
- Q1 FACTS PATH: PASS. CF pooled M1 top-1 = 1.000 (>= 0.95); CF - C0 M1 CI [+0.326, +0.417]
  is wholly above 0; both seeds positive on M1 (+0.368, +0.374, read from
  `top1.per_seed.seed_N.M1`, not from the overall metric).
- Q2 (descriptive, not a gate): candidate tokens WITHOUT facts did NOT beat the matched-capacity
  legacy head. C0 is slightly worse than L on M2+M3 (top-1 -0.027 [-0.047, -0.007], mass -0.040,
  CE +0.142), negative in both seeds; on M1 the seeds disagree (+0.003, -0.066). Reported, not rescued.
- Q3 DEEPER EXACT TECHNIQUE: NOT MET. CF M2 top-1 = 0.690 / 0.688 (pooled 0.689) is BELOW the
  0.75 floor in both seeds; CF M3 top-1 = 0.671 / 0.647 (pooled 0.659) clears the 0.55 floor in
  both seeds. The gate needs both, so it fails on M2. Both are far above chance (0.057 / 0.070).

EXTENSION CONDITIONS (as committed, conjunctive: (a) AND (b') AND (c)):
| | (a) TUNE CE 350 -> 400 (needs drop >= 0.01) | (b') pooled tune-train / macro tune-train (limits 0.15 / 0.25) | (c) finite |
|---|---|---|---|
| CF seed 1 | 1.4087 -> 1.4051, drop 0.0036: FAIL | +0.1086 / +0.1307: PASS | PASS |
| CF seed 2 | 1.4536 -> 1.4446, drop 0.0089: FAIL | +0.1159 / +0.1383: PASS | PASS |
DECISION: the extension does NOT trigger (condition (a) fails in both seeds). No 800-update run
was made. Per the plan, Q3 failed and the extension conditions fail: STOP before P3. Conversion
testing is not used to rescue a failed exact-technique gate.

CAVEAT ON CONDITION (a), stated for the owner and NOT used to change the decision: (a) is
measured over updates 350-400 of a cosine schedule whose LR is already <= 1.4e-5 and decaying
to 0, so a >= 0.01 drop is structurally hard to reach even when the model is far from converged
(TUNE CE fell from 2.64 at update 50 to 1.41 at 350 and was still falling slowly). Seed 2's 0.0089 is
within 0.0011 of the threshold. The rule was pre-registered; it is applied as written.

INTERPRETATION (three hypotheses kept separate):
1. Exact CandidateFacts add capability where they expose the answer: CF solves M1 (1.000 vs
   0.63-0.69 without facts) and also gives a smaller but significant, both-seeds-positive gain on
   M2+M3 over C0 (+0.060) and over L (+0.033).
2. Candidate-token representation by itself did not help under this contract: C0 <= L.
3. One-pass exact training reached M2 ~0.69 and M3 ~0.66 for CF (M3 above its gate, M2 below)
   with no overfit gap, so under this contract the one-pass model has not compiled M2 to the
   pre-registered floor. Whether more optimization (longer or differently scheduled) would close the
   M2 gap is NOT tested by this evidence.
P3 is not justified by the pre-registered rules. The next move is an owner decision.

DEFERRED PERFORMANCE NOTE (P4 scheduling qualification, not acted on): training GPU utilization
averaged ~85% with dips to 60-70% (see `gpu_util_busy_mean`). Candidate causes from code reading
(unprofiled): four host readbacks per micro-batch in `accumulated_update` (three are
reporting-only), CPU micro-batch assembly inline with the GPU work, and small-batch evaluation
passes. Fix later: one combined readback, background batch prefetch, per-update idle-fraction
logging.

---

## Owner decision after P2: P2.5 is a NEW research lineage
P2 remains a formal Q3 NO-GO: CF M2 0.690 / 0.688 (pooled ~0.689, floor 0.75), CF M3 0.671 /
0.647 (pooled ~0.659, floor 0.55). The old 800-update extension remains UNTRIGGERED (its
update-350-to-400 CE-drop condition failed in both seeds) and is closed. P3 remains
unauthorized under the old rule. No P2 number, threshold or decision is altered.
P2.5 (factorial + data scale + optional optimization horizon) is a separate, owner-authorized
experiment motivated by P2 evidence; it is neither the P2 extension nor a P2 rescue. All of
its rules are pre-registered in `docs/WORKSTATION_V25_P25_PLAN.md` before any P2.5 science.
