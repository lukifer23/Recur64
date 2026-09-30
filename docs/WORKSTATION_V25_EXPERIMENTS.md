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
