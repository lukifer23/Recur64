# Workstation V2.5 - P2.5 results

Plan (pre-registered): `docs/WORKSTATION_V25_P25_PLAN.md`. Starting SHA
`729c1e5183f2b36bc86d68288ef2efea51bd2d09`. P2 is frozen history and is not reinterpreted.
MEASURED = produced by a command in this repo; INFERRED = reasoning, not tested.

## Phase 1 - LF (`legacy_facts_v25`) built and gated
QUESTION: can the legacy head-v2 policy and the exact CandidateFacts be combined (the missing
cell of the 2x2)? MODEL: the unmodified `ProbeModel` (wrapped, so probe_v1 record layout and
hashes are untouched) + a candidate-local fact delta
`Linear(64->1)(GELU(Linear(8->64)(facts)))` added to the legacy policy logit before the masked
softmax; final layer small nonzero init (std 0.01); facts do not feed WDL.
MEASURED: 26,810,585 parameters = L 26,809,944 + 641. Engineering gates (CPU): finite forward;
policy sums to 1; padding exactly 0; terminal rows safe; changed facts change the policy; ZERO
facts reproduce the legacy policy (max diff < 1e-5); facts gradient finite and nonzero for both
facts layers on update 1; fresh entropy 0.998 x uniform, worst top-1 1.52 x uniform; WDL
logits exactly 0; exact checkpoint round trip; batched == sync inference; missing facts are
errors. Identity: distinct architecture id + `fact_delta_contract`; the full 3x3 cross-refusal
matrix (probe_v1 / candidate_v25 / legacy_facts_v25, every ordered pair, both loader/config
mismatch forms) refuses by architecture id on real checkpoint directories; the frozen P4.5 hash
test and the Probe-serialization tests are unchanged and green. CUDA (RTX 2000 Ada, FP32):
device guard passed, forward finite (p50 25.9 ms at batch 32), 64x4 training step finite (peak
2.6 GB), lifecycle growth 0 MiB. fmt clean, clippy 0, 299 workspace release tests pass.
Evidence: `docs/evidence/v25/p25/model-info-legacy-facts.json`,
`docs/evidence/v25/p25/factorial/lf-cuda-qual.json`.

## Phase 2 - heavy-family holdouts (generated and audited; NOT yet evaluated by any model)
Heavy families KQQvK / KQRvK / KRRvK x M1/M2/M3, 500 positions per cell per holdout, the same
exact solver, shortest-correct-move target, <= 15% correct fraction, facts-ambiguity rule,
canonical dedup and fresh_no_history contract as before. Exclusion is by CANONICAL CLASS of the
retired and the replacement P1/P2 TRAIN/TUNE/CONFIRM (six datasets, 24293 classes,
exclusion-manifest digest `29302432847909bf57bc37dd327ce12974abc2515313d5a9abd81c847f925661`) and of the other holdouts; the training extension (built later) excludes all of them.

| holdout | role | positions | seed | sha256 digest | chance top-1 M1 / M2 / M3 |
|---|---|---:|---|---|---|
| A | P2.5-F factorial | 4500 | 0x7A130001 | `384a528eb037b268d3ca214d5d9c90575d657c7a513a6f498afdfe76dd4e9b07` | 0.043 / 0.058 / 0.076 |
| B | P2.5-D data scale | 4500 | 0x7A130002 | `13c8e018beea71058dbbbf510c723ebbe85b3a057c7b0e4a845a1039e107df10` | 0.042 / 0.059 / 0.078 |
| C | P2.5-O horizon (only if triggered) | 4500 | 0x7A130003 | `4ab951c6edd8dd4f531bb87d2f4373895d1fddb70efdf052b24c09a1b71d87d5` | 0.042 / 0.059 / 0.076 |

MEASURED: 100% independent GameState audit (0 failures in A, B, C); holdouts mutually
disjoint and disjoint from every excluded split by canonical class and exact FEN; no
holdout-exposure log exists (no model has evaluated any holdout). Evidence:
`docs/evidence/v25/p25/holdouts/`.

### Per-cell pool accounting (the pre-registered pool-limit rule is now resolved)
| cell | eligible pool | excluded (earlier splits) | available | holdouts (A+B+C) | REMAINING for extension (target 4,000) |
|---|---:|---:|---:|---:|---:|
| KQQvK M1 | 95649 | 2390 | 93259 | 1500 | 91759 |
| KQQvK M2 | 174163 | 2394 | 171769 | 1500 | 170269 |
| KQQvK M3 | 4409 | 2078 | 2331 | 1500 | 831 |
| KQRvK M1 | 111273 | 2384 | 108889 | 1500 | 107389 |
| KQRvK M2 | 306595 | 2398 | 304197 | 1500 | 302697 |
| KQRvK M3 | 211215 | 2391 | 208824 | 1500 | 207324 |
| KRRvK M1 | 41612 | 2364 | 39248 | 1500 | 37748 |
| KRRvK M2 | 122082 | 2393 | 119689 | 1500 | 118189 |
| KRRvK M3 | 108086 | 2384 | 105702 | 1500 | 104202 |

MEASURED: eight of nine heavy cells have at least 37,748 classes left. **KQQvK M3 has only 831**
(pool 4,409; 2,078 already used by earlier splits; 1,500 to the holdouts). RULE APPLIED (fixed
before this measurement): holdouts take priority, the extension takes all that remains up to
4,000, nothing is relaxed. So P25_DATA_V1 will hold 1,000 + 831 = **1,831** unique KQQvK M3
TRAIN positions, not 5,000; the other eight heavy cells get 5,000. `cell_balanced_v1` keeps
per-cell EXPOSURE equal, so KQQvK M3 will be oversampled (~3.7 local epochs at 400 updates
vs ~1.4 for a 5,000 cell). INFERRED: this slightly dilutes the "5k vs 1k" contrast in exactly one
of nine heavy cells; it is reported, not hidden.


---

## CORRECTION (pre-science) - LF contract 2, scope narrowing, and an incident
Supersedes the LF size and gradient statements in Phase 1 above (the original text is kept).
- **Fix:** the final fact-delta layer's bias was inert (constant per row, cancelled by the
  softmax, zero gradient) and is removed; `FACT_DELTA_CONTRACT` 1 -> 2; exact LF size
  **26,810,584** = L + 640. MEASURED after the fix: per-parameter gradients for `facts1.weight`,
  `facts1.bias`, `facts2.weight` are finite and nonzero after update 1; the final layer has no bias;
  fresh entropy 0.994 x uniform, worst top-1 1.69 x uniform, WDL exactly 0; CUDA (RTX 2000 Ada):
  device guard, forward finite (p50 26.9 ms), 64x4 training step finite (peak 2.6 GB), lifecycle
  growth 0 MiB. Evidence: `docs/evidence/v25/p25/model-info-legacy-facts.json`,
  `docs/evidence/v25/p25/factorial/lf-cuda-qual.json` (the contract-1 run is kept as
  `lf-cuda-qual-contract1-engineering-only.json`).
- **Legacy-base identity (new structural regression test):** with the backend seeded identically,
  LF's wrapped Probe is BIT-IDENTICAL to an independently built Probe, and zero-fact LF equals its
  wrapped Probe (max diff < 1e-6). The test first failed when it shared a file with other tests:
  Burn's backend RNG is global process state, so a sibling test consuming random numbers between
  `seed()` and the constructions broke the comparison. Run alone it passes 3/3, so the legacy path is
  unchanged; the test now lives in its own integration-test file (its own process) and is stable.
- **Scope:** P2.5-O cancelled/deferred; HOLDOUT_C reserved but unused; STOP after P2.5-D (see the
  scope addendum in `WORKSTATION_V25_P25_PLAN.md`).
- **Incident (disclosed):** the agent had started LF training under contract 1 and had generated
  and committed HOLDOUT_A/B/C (`934210c`, after the `808943b` this addendum expected) before the
  addendum arrived. Stopped on receipt. LF seed 1 had COMPLETED (contract 1) and, as a descriptive
  check, evaluated the OLD P2 CONFIRM (the seventh entry in `confirm-exposure.log`); LF seed 2 was
  at update 325 with no outputs. Both are engineering-only and void as science, quarantined UNREAD
  in `runs/v25/aborted/lf-contract1-engineering-only/`. No holdout was evaluated
  (no `holdout-exposure.log`); the holdouts are deterministic data generated without any model, so they
  were kept as committed.


---

## P2.5-F RESULT - the completed 2x2 factorial on HOLDOUT_A
QUESTION: which combination of action representation and exact facts is best, and do they interact?
PRE-REGISTERED RULE: plan above (LF seeds 1/2 under the P2 contract; evaluate L, C0, CF, LF x 2 seeds on
HOLDOUT_A; selection by CF-LF / LF-CF on HOLDOUT_A M2+M3 top-1, CI wholly > 0 and both seeds positive).
CONFIG: LF = `legacy_facts_v25`, fact-delta contract 2, 26,810,584 parameters; TRAIN `1e5e121b...`,
TUNE `f59d744a...`, cell_balanced_v1, LR 3e-4, 400 updates, warmup 40, cosine over 400, batch 256
(64x4), policy-only, FP32. Commit of the runs: see the git log; git_sha in `MANIFEST-F.txt`.
DATA: HOLDOUT_A `384a528eb037b268d3ca214d5d9c90575d657c7a513a6f498afdfe76dd4e9b07` (4,500 heavy positions, never used for training or selection); chance
top-1 M1 / M2 / M3 = 0.043 / 0.058 / 0.076. Exposure: `holdout-exposure.log` has exactly 8 entries, all HOLDOUT_A.
MODEL IDS / RESOURCES (LF, descriptive old-P2-CONFIRM is a continuity check only; columns 6-7 are M1 on TRAIN / TUNE / old CONFIRM, then old-CONFIRM M1 / M2 / M3):
| run | model_id | wall | peak VRAM | loss | max grad norm | M1 TRAIN / TUNE / old CONFIRM | old CONFIRM M1 / M2 / M3 |
|---|---|---:|---:|---|---:|---|---|
| LF-s1 | 7866eb1f4841 | 351 s | 2576 MB | 3.447 -> 1.255 | 7.4 | 0.801 / 0.779 / 0.770 | 0.770 / 0.651 / 0.638 |
| LF-s2 | 1fa803be5716 | 355 s | 2576 MB | 3.439 -> 1.256 | 9.3 | 0.839 / 0.816 / 0.779 | 0.779 / 0.688 / 0.653 |

MEASURED - absolute HOLDOUT_A (pooled top-1; mass; CE; macro-cell top-1):
| model | seed | M1 | M2 | M3 | mass | CE | macro-cell top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| L | 1 | 0.637 | 0.584 | 0.583 | 0.435 | 2.149 | 0.602 |
| L | 2 | 0.679 | 0.621 | 0.545 | 0.458 | 2.109 | 0.615 |
| C0 | 1 | 0.613 | 0.553 | 0.537 | 0.398 | 2.245 | 0.568 |
| C0 | 2 | 0.616 | 0.555 | 0.548 | 0.407 | 2.236 | 0.573 |
| CF | 1 | 1.000 | 0.680 | 0.604 | 0.639 | 1.594 | 0.761 |
| CF | 2 | 1.000 | 0.691 | 0.598 | 0.638 | 1.613 | 0.763 |
| LF | 1 | 0.728 | 0.593 | 0.593 | 0.446 | 2.069 | 0.638 |
| LF | 2 | 0.761 | 0.630 | 0.558 | 0.478 | 2.006 | 0.650 |

MEASURED - the 2x2 (seed-averaged HOLDOUT_A top-1 M1 / M2 / M3 / M2+M3 | mass | CE):
| | NO FACTS | FACTS |
|---|---|---|
| legacy policy | L: 0.658 / 0.603 / 0.564 / 0.583 | 0.446 | 2.129 | LF: 0.744 / 0.612 / 0.576 / 0.594 | 0.462 | 2.038 |
| candidate-token policy | C0: 0.614 / 0.554 / 0.543 / 0.548 | 0.402 | 2.241 | CF: 1.000 / 0.686 / 0.601 / 0.643 | 0.639 | 1.603 |

MEASURED - effects (top-1, B minus A, paired bootstrap 95% CI; per-seed in the last column; full mass/CE tables in
`docs/evidence/v25/p25/factorial/compare-*.json` and `interaction.json`):
| effect | group | pooled [95% CI] | seed 1 / seed 2 |
|---|---|---|---|
| architecture effect without facts: C0 - L | M1 | -0.044 [-0.060, -0.028] | -0.025 / -0.063 |
|  | M2 | -0.049 [-0.065, -0.032] | -0.031 / -0.067 |
|  | M3 | -0.021 [-0.037, -0.007] | -0.046 / +0.003 |
|  | M2+M3 | -0.035 [-0.046, -0.024] | -0.038 / -0.032 |
| facts effect in candidate: CF - C0 | M1 | +0.386 [+0.363, +0.408] | +0.387 / +0.384 |
|  | M2 | +0.132 [+0.112, +0.152] | +0.127 / +0.137 |
|  | M3 | +0.058 [+0.041, +0.077] | +0.067 / +0.050 |
|  | M2+M3 | +0.095 [+0.081, +0.108] | +0.097 / +0.093 |
| facts effect in legacy: LF - L | M1 | +0.086 [+0.074, +0.098] | +0.091 / +0.081 |
|  | M2 | +0.009 [-0.002, +0.020] | +0.009 / +0.009 |
|  | M3 | +0.012 [+0.002, +0.021] | +0.010 / +0.013 |
|  | M2+M3 | +0.010 [+0.003, +0.017] | +0.010 / +0.011 |
| architecture effect with facts: CF - LF | M1 | +0.256 [+0.236, +0.276] | +0.272 / +0.239 |
|  | M2 | +0.074 [+0.054, +0.094] | +0.087 / +0.061 |
|  | M3 | +0.025 [+0.007, +0.044] | +0.011 / +0.040 |
|  | M2+M3 | +0.050 [+0.036, +0.064] | +0.049 / +0.051 |
| INTERACTION (CF - C0) - (LF - L) | M1 | +0.300 [+0.276, +0.323] | +0.297 / +0.303 |
|  | M2 | +0.123 [+0.101, +0.145] | +0.117 / +0.128 |
|  | M3 | +0.047 [+0.026, +0.067] | +0.057 / +0.037 |
|  | M2+M3 | +0.085 [+0.070, +0.100] | +0.087 / +0.082 |

SELECTION (pre-registered rule, primary group HOLDOUT_A M2+M3 top-1): CF - LF = +0.050 [+0.036, +0.064], both seeds
positive (+0.049 / +0.051) => **SELECT CF**. LF - CF is -0.050 [-0.064, -0.036].

LF HEALTH EXPECTATION (plan: LF M1 >= 0.95): NOT MET. LF M1 = 0.728 / 0.761 on HOLDOUT_A. MEASURED
diagnosis using TRAIN, TUNE and the old CONFIRM only (no holdout): LF is UNDERFIT on M1, not overfit and
not broken - M1 is ~ the same on TRAIN (0.80-0.84), TUNE (0.78-0.82) and old CONFIRM (0.77-0.78), and LF's
TUNE M1 was still rising at update 400 (seed 1: 0.18 -> 0.55 -> 0.64 -> 0.69 -> 0.76 -> 0.78), whereas CF has
M1 = 1.000 from update 50. Facts do change LF's policy, its gradients are nonzero, and it beats L on M1
(0.744 vs 0.658), so the facts signal is learnable through the scalar-delta path but LEARNS FAR TOO SLOWLY
under this contract. The plan said LF must not be selected for scaling until understood; it was not selected.

INFERRED (not tested): (1) Exact facts are far more useful through the candidate-token pathway than as a scalar
logit delta on the legacy head: the interaction is large and positive on every group (M2+M3 +0.085, M1 +0.300),
and CF beats both LF (M2+M3 +0.050) and L. (2) The LF numbers are conditional on LF as
parameterized and trained here; they do NOT show that legacy+facts is weak in principle - the small-init scalar
output layer behind a GELU hidden layer may simply need a different LR/init to be used as fast as CF uses facts. That
confound is OPEN and, per the owner scope, is not pursued in P2.5. (3) Candidate tokens alone remain slightly worse
than the legacy head (C0 - L M2+M3 -0.035, both seeds negative), so the candidate architecture's advantage is
specific to having facts to integrate.
DECISION: P2.5-D scales CF (seeds 1, 2) with the original P2 CF (1k heavy data) as the matched baseline.
NEXT ACTION: build P25_DATA_V1 (this result is committed first).


---

## P2.5-D dataset - P25_DATA_V1 (built and audited; no scaled model trained yet)
CF was selected by the P2.5-F rule, so only CF is scaled. QUESTION: is the remaining M2 gap primarily a
unique-data problem? Keep the exact small-family TRAIN unchanged, keep the existing 1,000 heavy positions
per cell, and add up to 4,000 new positions per heavy cell (plan: "P2.5-D").
DATA DIGEST: `3b25dc8549dd2fc9d47c30e294c273b3306aecb3eba91b964715326ddf74f2e6` - 44,332 unique exact TRAIN positions = 11,501 base + 32,831 added
(extension seed 0x7A140001). TUNE is the unchanged replacement P2 TUNE (`f59d744a133e887d7c50a00923ee2d16b9bd7a4560330252a93574828bfc16e4`).
EXCLUSIONS: every ADDED position avoids, by canonical class, the retired splits, the replacement TRAIN, TUNE
and CONFIRM and HOLDOUT_A/B/C (9 datasets; exclusion-manifest digest `60fbdd5e422bbd4696aea7492d21674a3c790fd9b10d661dc252bff29018ff58`).
MEASURED: independent audit of ALL 44,332 positions, 0 failures; the combined TRAIN shares no canonical class or
exact FEN with replacement TUNE/CONFIRM or any holdout.

| cell | eligible pool | excluded | available | ADDED |
|---|---:|---:|---:|---:|
| KQQvK M1 | 95649 | 3890 | 91759 | 4000 |
| KQQvK M2 | 174163 | 3894 | 170269 | 4000 |
| KQQvK M3 | 4409 | 3578 | 831 | 831 |
| KQRvK M1 | 111273 | 3884 | 107389 | 4000 |
| KQRvK M2 | 306595 | 3898 | 302697 | 4000 |
| KQRvK M3 | 211215 | 3891 | 207324 | 4000 |
| KRRvK M1 | 41612 | 3864 | 37748 | 4000 |
| KRRvK M2 | 122082 | 3893 | 118189 | 4000 |
| KRRvK M3 | 108086 | 3884 | 104202 | 4000 |

The pre-registered pool-limit rule applied exactly as written: eight heavy cells reach 5,000 unique positions; KQQvK
M3 adds only its 831 remaining classes (1,831 total). Total 44,332 vs the plan's ~47,501 estimate (which assumed a
full 5,000 in all nine heavy cells). Small families are unchanged (KQvK 1,570, KRvK 931).

| cell | unique TRAIN positions | local epochs at 400 updates (cell_balanced_v1) |
|---|---:|---:|
| KQQvK-M1 | 5000 | 1.37 |
| KQQvK-M2 | 5000 | 1.37 |
| KQQvK-M3 | 1831 | 3.73 |
| KQRvK-M1 | 5000 | 1.37 |
| KQRvK-M2 | 5000 | 1.37 |
| KQRvK-M3 | 5000 | 1.37 |
| KQvK-M1 | 246 | 27.75 |
| KQvK-M2 | 462 | 14.78 |
| KQvK-M3 | 862 | 7.92 |
| KRRvK-M1 | 5000 | 1.37 |
| KRRvK-M2 | 5000 | 1.37 |
| KRRvK-M3 | 5000 | 1.37 |
| KRvK-M1 | 153 | 44.62 |
| KRvK-M2 | 426 | 16.03 |
| KRvK-M3 | 352 | 19.39 |

INFERRED: per-cell EXPOSURE is unchanged by construction (~6,827 examples per cell at 400 updates), so the scaled run
changes UNIQUE data, not example counts: a 5,000-position heavy cell is seen ~1.37 times instead of ~6.8, and KQQvK M3
~3.7 times. Evidence: `docs/evidence/v25/p25/data-scale/dataset-meta.json`.
