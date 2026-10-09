# Recur64 V69 — D3 additional-update diagnostic on the full 768 fitting partition: results and decision

Contract: `D3_CONTRACT.md` (+ `d3_config.json`), frozen before any fit. Evidence: `evidence/d3/`. Labels: **MEASURED**, **INFERRED**, **NOT RUN**.
D3 is fitting-only (no validation/sealed use, no new data/seed/feature/architecture/optimizer/precision change). **Nothing here is a generalization or fresh-confirmation claim.**

## Decision

**With 12,000 updates (D2 prefix preserved, LR held at 5e-5 after update 1,999), D3-A escapes its prior-like plateau and strong-fits the full 768-example fitting partition; D3-M improves steadily but does not.**
- **D3-A (unchanged one-pass model):** balanced accuracy 0.995 (764/768 correct), BCE 0.0112, AUROC 1.000 at update 12,000 → *strong fit* (BA ≥ 95% and BCE ≤ 0.05), *not* exact memorization (4 errors).
  It stayed on the class prior through update 3,000 (as in D2), first met the partial-learning criterion at **update 4,000** (BA 0.628, BCE 0.648; confirmed at 5,000), and then improved monotonically.
- **D3-M (direct-board MLP):** balanced accuracy 0.897, BCE 0.282, AUROC 0.963 at update 12,000 (still improving: 0.764 at 2,400); *no* strong fit.
Per the pre-declared rules: fitting improved after a comparable D2 prefix, so **additional updates helped under this schedule**; this does not make budget the only cause of the earlier failure, and it says nothing about generalization.

## 1. Identity, integrity, provenance — MEASURED

- Consumer head for qualification, both fits and aggregation: `50f95b5f…` (clean, 0 dirty files); data producer `36a81508…` (distinct). Pushed to `origin` on this branch only; no merge.
- `d3/frozen_d3.json` holds 60 learner-group and 23 aggregator-group expected hashes; both fits logged "60 frozen hashes verified" before training and the aggregation "23 … verified". Each `provenance.json` binds source identity, contract/config/frozen hashes,
  s768 rows (D2's exact file), the extended order (`50c6a7ff…`), init (A: E1 canonical `3bc80e9f…`; M: D1 MLP `0228b6ea…`, tensor hashes verified), 12,000 completed updates, the actual model/optimizer/metadata file hashes and the prediction-file hash; the aggregator re-checked them.
- **Preservation (append-only receipts `evidence/d3/preservation_{start,end}.json`):** 235 files (E1, D1 and **D2** supplementary manifests + gen-001 manifest) re-hash identically before and after; nothing was modified. No fitted weights or moments of any earlier phase were loaded (hash-only). The documented E1 audit-receipt deviation is untouched.
- Example stream: `d2_train_order/768/<epoch>` extended to 192,000 samples (250 epochs); the first 38,400 equal D2's stream exactly (checked at construction, in qualification and by the aggregator); every example has exactly 250 exposures (checked independently from the frozen order and against each run's recorded exposures).
- Learning rate: the aggregator independently re-derived the rule and matched all 12,000 updates (|Δ| ≤ 1e-15) and the first 2,400 against D2's recorded rates.

## 2. Qualification — MEASURED (passed first attempt, 21 s)

D2/D1 machinery reused unchanged (all 20 qualified checks) plus D3 checks: LR rule for every u < 12,000 and boundaries; extended order (D2 prefix, epoch permutations, 250 exposures). Disposable weights only; limits respected (host 1,439 MiB, sampled device 1,187 MiB).
**Preserved failures:** none. Precision as measured: f32 storage/accumulation, matmul inputs possibly TF32; not strict FP32.

## 3. Endpoints and trajectories — MEASURED

| update | A: acc/BA | A BCE | A AUROC | M: acc/BA | M BCE | M AUROC |
|---|---|---|---|---|---|---|
| 2,400 (D2 endpoint) | 0.500 | 0.6931 | 0.522 | 0.764 | 0.4812 | 0.850 |
| 3,000 | 0.500 | 0.6931 | 0.546 | 0.760 | 0.4782 | 0.855 |
| 4,000 | 0.628 | 0.6478 | 0.688 | 0.777 | 0.4598 | 0.866 |
| 5,000 | 0.685 | 0.5793 | 0.763 | 0.789 | 0.4436 | 0.878 |
| 6,000 | 0.780 | 0.4266 | 0.878 | 0.810 | 0.4254 | 0.890 |
| 8,000 | 0.883 | 0.2459 | 0.964 | 0.840 | 0.3825 | 0.917 |
| 10,000 | 0.980 | 0.0606 | 0.998 | 0.868 | 0.3330 | 0.942 |
| 12,000 | **0.995** | **0.0112** | 1.000 | **0.897** | **0.2818** | 0.963 |

All 21 snapshots, confusion matrices, per-cell results and class-conditional margins are in `evidence/d3/d3_report.json`. A's logit spread rises from ≈ 0 (≤ 3,000) to 0.57 (4,000) and 10.6 (12,000), class means +10.75 / −9.85.
Per-cell accuracy at 12,000: A ≥ 0.984 in all 12 cells; M 0.84–0.97.
**A's plateau escape:** first snapshot meeting BA ≥ 0.60 and BCE ≤ 0.65 and confirmed at the next one = **4,000**; no temporary improvements before it. M met the criterion at 400 (confirmed). The escape occurred between updates 3,000 and 4,000 (snapshot resolution).
**Mechanics during A's escape:** A's pooled-representation variation (encoder/workspace) was 0.12/0.10 at init, 0.02/0.01 from update ≈ 100 through 3,000, then re-expanded: 0.44/0.23 (4,000), 0.83/0.54 (5,000), 1.34/0.98 (6,000), peaking ≈ 1.8/2.2 (8,000) and settling ≈ 1.55/1.95 (12,000).
Mean pre-clip gradient norm rose from ≈ 0.5 (plateau) to 6.6 (updates 4,000–6,000), 30–35 (6,000–10,000) and 22 (10,000–12,000), with 92–100% of updates clipped between 4,000 and 10,000 and 56% after: the escape and the fit occur under almost-continuous clipping. Total parameter movement from init: A 6.9% (2,400) → 14.5% (12,000); M 41.6% → 62.1%.
M's gradient norm stays ≈ 2.3–3.4 with ≈ 93–99% clipped throughout; loss falls steadily (0.481 → 0.282) and had not plateaued.

## 4. D2-prefix comparison — MEASURED (differences reported, not forced)

Same initialization, order and LR; autotuned CUDA kernels may differ. At the eleven D2 checkpoints the D3 and D2 runs agree to ≈ 3–4 digits in metrics (A BCE e.g. 0.7184 vs 0.7144 at 400, otherwise ≤ 0.0004 apart; BA/accuracy identical at every checkpoint; M metrics identical to 4 digits).
Per-example logit differences: A mean 4.5e-4 – 3.7e-2 (max 4.5e-2 at update 400), M 0 until update 200 then mean ≈ 4e-4 / max ≈ 2e-3. The trajectories are therefore near-identical through 2,400, so D3's later improvement is not explained by any difference in the first 2,400 updates; it occurs *after* update 3,000 and is attributable to the additional updates under this schedule.

## 5. Group and error analysis — MEASURED

- **256 → 768 expansion:** of the 512 added examples, 500 fell in connected groups already present at 256 (12 in new groups; 200 → 204 groups). So D2's size increase mostly added examples *within existing groups*; degradation at 768 is therefore not attributable solely to sample count or to new groups.
- **A at 12,000:** errors in 4 of 204 groups (193 groups have more than one example; all 4 erring groups are multi-example, one error each, positive mean margins 4.0–6.7). **M at 12,000:** errors in 65 of 204 groups (63 multi-example), smallest margins ≈ 0.5.
- **Complementarity with the D1 shallow baseline's FITTING predictions** (threshold 0; baseline fit BA 0.764; descriptive only; no retraining, ensemble, threshold choice or validation):

| endpoint | agree | both correct | both wrong | neural-only correct | baseline-only correct |
|---|---|---|---|---|---|
| A @12,000 | 583 | 583 | 0 | 181 | 4 |
| M @12,000 | 558 | 533 | 25 | 156 | 54 |

A is right on essentially every example the baseline gets right and on 181 more; M agrees with the baseline's shallow structure and is complementary on 156 vs 54. This is evidence about *fitting*, not generalization: A fits 768 examples to 99.5% where shallow features explain ≈ 76%.

## 6. Independent recomputation and resources — MEASURED

`v69-d3 aggregate` recomputed predictions-derived metrics (f64), exposures, the LR sequence and the D2-prefix comparison from serialized artifacts, requiring complete ids, 21 snapshots, matching hashes and agreement with in-process summaries (tolerance 1e-4).
Native fit time: A 2,781 s (mean update 230 ms), M 179 s (15 ms); total 2,960 s ≈ 49 min of the 2 h global ceiling; every fit within its cap (80 / 10 min); exit 0; 0 orphans. Peak host 461 / 289 MiB; sampled device 225 / 95 MiB (launcher nvidia-smi, ≈ 2 s sampling; may miss short peaks; allocator high-water unavailable).

## 7. MEASURED / INFERRED / NOT RUN

MEASURED: everything above. INFERRED: (a) A's non-learning in D2 was a *delay*: under this schedule its escape occurs after ≈ 3,000 updates at N = 768 and its escape time grows with N (D2: ≈ 100–200 at 64, ≈ 300 at 256, ≈ 3,500 at 768); (b) the escape coincides with the re-expansion of its pooled representation and a regime of persistent heavy gradient clipping — a mechanism is not identified;
(c) A reaching 99.5% fit where shallow features give 76% is consistent with memorization of the 768 fitting rows (D2 showed memorization at 64/256) and does not indicate generalization; (d) M's slower, shallow-structure-like improvement suggests it is closer to the shallow baseline's solution than to memorization at this budget.
NOT RUN: validation or sealed evaluation; any generalization or fresh-confirmation test; other horizons, LRs, optimizers, clipping thresholds, models or data; hierarchy comparisons; strict-FP32 matmul; seeds beyond the single canonical initializations.

## 8. Recommendation and next-agent prompt

The unchanged one-pass model can fit the full fitting partition, so the open question is no longer *whether it can fit* but *whether anything it learned transfers*. A fit at 99.5% against a shallow-feature fit of ≈ 76% is exactly the regime where memorization must be ruled out before any architecture claim.
Do not escalate horizons again. The next step is a **single, separately frozen generalization test** of the already-fitted, frozen D3 endpoints (D3-A and D3-M final checkpoints, plus the D1 baseline) on **fresh data**, which requires explicit owner approval because agents may not generate new data or redraw the master seed
(gen-001 validation is already-exposed development evidence). If approval is not available, the only permissible measurement is an exploratory, clearly-labelled evaluation on gen-001 validation, which would not be fresh confirmation.
A controlled optimization intervention (e.g. addressing the heavy-clipping regime) is a secondary question and should only be run after generalization is measured.

> Continue on branch `experiment/hp-v69-two-clock-workspace` (read AGENTS.md and docs/v69/{CONTRACT,MODEL_SPEC,RESULTS_E1_THREE_ARM,D1_CONTRACT,RESULTS_D1,D2_CONTRACT,RESULTS_D2,D3_CONTRACT,RESULTS_D3}.md). Do not rerun or continue E1–D3.
> Phase G1 (generalization test of frozen candidates) — **blocked on owner approval for a fresh data draw**; ask for it first. If approved: write a G1 contract before generating anything; draw a new master seed from OS entropy (record before generation); use the audited V69 generator/teacher/partition code unchanged to produce a fresh,
> independently custody-checked evaluation partition (document the identical-domain restriction and the no-historical-comparison limitation); freeze candidates = D3-A final, D3-M final (hash-verified, loaded for inference only) and the D1 baseline (no refit); define metrics, group-bootstrap uncertainty and decision rules before any model touches the new data; evaluate each candidate exactly once;
> report fit-vs-fresh gaps, per-cell and group results, derangement/erasure controls from the E1 protocol, and baseline comparison. No retraining, tuning, threshold or candidate selection on the fresh data; no sealed-test use; no hierarchy claims. Stop after G1; do not execute follow-ups.
