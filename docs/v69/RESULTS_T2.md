# Recur64 V69 — T2 results: a neural model clearly beats the shallow baseline

Contract: `T2_CONTRACT.md`, `t2_config.json`, `T2_AMENDMENT_A1.md`. Evidence: `evidence/t2/` (machine-written report, selection, receipts, frozen hashes, logs). Labels: **MEASURED** (read from a machine-written artifact), **INFERRED** (my interpretation), **NOT RUN**.
Scope: same-domain generalization on exactly-labelled immediate nonterminal children of KQQvK/KQRvK/KRRvK (attacker to move at roots, mate within n = 1 or 2). No claim about architecture, recurrence, hierarchy, search or policy.

## 1. Headline (MEASURED, sealed T2 test, 6,144 examples, 1,315 connected groups)

| Candidate | Balanced acc. | 95% CI | BCE | AUROC | Real − derangement (pp) |
|---|---|---|---|---|---|
| **M\*** `M_k6000_sym_lr3e-3_e30` (MLP, symmetry-averaged training) | **0.912** | 0.904–0.921 | 0.347 | 0.967 | 41.7 |
| **P\*** `M_k6000_d8_lr3e-3_e60` (MLP, random-symmetry augmentation only) | 0.848 | 0.837–0.859 | **0.343** | 0.922 | 35.2 |
| B (frozen D1 shallow baseline) | 0.776 | 0.766–0.787 | 0.476 | 0.855 | 27.8 |
| B2 (same baseline refit on T2 train) | 0.779 | 0.769–0.790 | 0.473 | 0.856 | 28.1 |
| A\* `A_k2000_d8_lr5e-4_e54` (E1 one-pass workspace model) | 0.690 | 0.677–0.703 | 0.578 | 0.764 | 18.7 |

Paired cluster-bootstrap (5,000 resamples, identical group draws, balanced-accuracy difference, 95% interval):

| Comparison | Gain (pp) | 95% interval (pp) | BCE difference |
|---|---|---|---|
| M\* − B | +13.6 | 12.4 to 14.8 | −0.129 (better) |
| M\* − B2 | +13.3 | 12.2 to 14.5 | −0.126 (better) |
| P\* − B | +7.2 | 6.0 to 8.4 | −0.133 (better) |
| P\* − B2 | +6.9 | 5.7 to 8.2 | −0.130 (better) |
| M\* − P\* | +6.4 | 5.5 to 7.4 | +0.004 (interval −0.027 to 0.037, no difference) |
| A\* − B2 | −9.0 | −10.6 to −7.4 | +0.105 (worse) |
| B2 − B | +0.3 | −0.2 to 0.8 | −0.003 |

**Pre-registered decisions (MEASURED):**
- **M\*: transfer met; practical improvement over both B and B2 met** (gain ≥ 5 pp, interval excludes 0, BCE no worse). Significant-gain flag also met.
- **P\*: transfer met; practical improvement over both B and B2 met** (+7.2 / +6.9 pp), so the gain does not depend on the symmetry-averaged objective.
- **A\*: transfer not met** (balanced accuracy 0.690 < 0.75, BCE 0.578 > 0.55). It is worse than both baselines.
- Integrity passed (`integrity_ok: true`; derangement donor predictions consistent for every candidate).

## 2. Controls (MEASURED)
- **Derangement** (board swapped within family × budget cell, scored against recipient labels): balanced accuracy 0.495–0.503 for all five candidates, i.e. chance. Predictions are driven by the board, not by cell identity.
- **Erasure** (OOD diagnostic): all candidates 0.50 balanced accuracy; P\* predicts positive for 100% of erased inputs, the others 50%.
- **G1 panel (secondary, report-only, 1,536 examples):** same ordering and the same flags: M\* 0.912 / BCE 0.357, P\* 0.835 / 0.359, B 0.764 / 0.481, B2 0.774 / 0.474, A\* 0.684 / 0.598. M\* − B2 +13.7 pp, P\* − B2 +6.1 pp, both intervals exclude zero.
- **8-fold TTA (report-only):** M\* unchanged by construction (its registered inference is already the 8-view mean). P\* rises to 0.866 / BCE 0.305 and A\* to 0.724 / 0.543 — not used for any decision.

## 3. Where the gain is (MEASURED, per-cell accuracy, 512 per cell)
M\* beats B2 in all 12 family × budget × class cells; the largest gains are in the harder n = 2 negative cells (e.g. KRRvK n2 neg 0.936 vs 0.701, KQRvK n1 neg 0.938 vs 0.732). P\*'s weakest cells are KQQvK n2 neg (0.689) and KQRvK n2 neg (0.705); A\* is 0.59–0.81 everywhere.

## 4. Training and selection (MEASURED, validation only; 13 registered runs, all completed)
Final-update validation BCE / balanced accuracy:

| Run | Val BCE | Val BA |
|---|---|---|
| M k6000 sym LR 3e-3 e30 (**M\***) | 0.323 | 0.915 |
| M k6000 d8 LR 3e-3 e60 (**P\***) | 0.331 | 0.843 |
| M k6000 sym LR 1e-2 e30 | 0.335 | 0.850 |
| M k6000 d8 LR 3e-3 e30 | 0.340 | 0.844 |
| M k2000 sym LR 1e-2 e30 | 0.411 | 0.815 |
| M k2000 d8 LR 3e-3 e30 | 0.424 | 0.804 |
| M k6000 d8 LR 1e-2 e30 | 0.489 | 0.759 |
| M k2000 sym LR 3e-3 e30 | 0.574 | 0.869 |
| M k6000 sym LR 3e-3 e60 | 0.685 | 0.900 |
| M k2000 d8 LR 1e-2 e30 | 0.608 | 0.652 |
| A k2000 LR 5e-4 e54 (**A\***) | 0.566 | 0.707 |
| A k6000 LR 1e-3 e18 | 0.627 | 0.641 |
| A k6000 LR 5e-4 e18 | 0.636 | 0.620 |

Validation → test: M\* 0.915 → 0.912 BA, BCE 0.323 → 0.347; P\* 0.843 → 0.848, BCE 0.331 → 0.343; A\* 0.707 → 0.690. The validation selection did not overfit the test.
Grid wall time 9,237 s (≈ 2.6 h) plus data generation, audit, refits, evaluation and builds; within the 4 h native limit. Peak host 1.06 GiB, sampled device 801 MiB.

## 5. Interpretation (INFERRED — descriptive, not causal proof)
- With 3× more exactly-labelled data, a small direct-board MLP clearly exceeds a 23-feature shallow baseline on this domain; T1's "tied at the baseline level" result was data/objective limited. Both more data (k = 6,000 beat k = 2,000 in validation) and the symmetry-averaged objective (+6.4 pp over random augmentation alone) contributed; the data comparison is validation-only and confounded with the LR grid.
- M\*'s mean negative logit is about −23 against +7 for positives (MEASURED), so it is extremely confident on negatives; its BCE advantage over P\* is nil even though its accuracy is 6 pp higher. Calibration of the sym-trained model is poor; the 60-epoch sym run (val BCE 0.685) shows the same tendency.
- A\* remains the weakest candidate even with a 4.5× longer schedule (it improved from the T1 test BA 0.580 to 0.690). A\* was only tried with LR 5e-4/1e-3 and 18–54 epochs; I cannot tell whether it is intrinsically weaker or under-trained. This does not say anything about recurrence or workspace architectures in general.

## 6. Deviations and incidents (MEASURED / disclosed)
1. **Amendment A1** (`T2_AMENDMENT_A1.md`): strict "any prior child" exclusion saturated the finite mate-in-2/3 domain once the T1 pool was in the prior (96% of generated roots excluded); the first generation was stopped (kept under `attempt1_infeasible_strict_exclusion/`) and the exclusion changed to root-level identity plus example-level child identity, with the same seed. Written and committed before the regeneration; the audit then passed with 0 failures (81,216 examples).
2. **Evaluator row-count constant.** The first `eval` invocation failed on a hard-coded test row count (3,072, a T1 value; T2 test has 6,144). It failed at the row-count check, before any prediction or metric was computed. The test file had been read, hashed and featurized by that process (no labels were used). The fix was a one-constant source change, committed (`5f807f4`), and `source` / `verify-evaluator` / `freeze-final` were redone (failed copies in `evidence/t2/attempt_freeze2_*`). Each candidate was then evaluated once.
3. **Aggregator fix after evaluation.** The aggregator wrongly treated the P candidate as non-neural and rejected its TTA rows; the fix (`cand == "P"` treated as neural) is analysis-only, committed after the predictions existed, and changes no prediction or decision rule. The end preservation receipt (837 files, 0 mismatches) was written before this fix and records source digest `d829…ebffa`.
4. Smoke-test runs (`*_smoke`) are excluded from selection and from the 13 registered runs.
5. Precision: CUDA f32 storage/accumulation, matmul inputs possibly TF32 — **not** strict FP32. Baselines are host f64.

## 7. NOT RUN
Wider/deeper MLPs, other optimizers or schedules, A with symmetry-averaged training or longer schedules, calibration/temperature fitting, out-of-domain positions (other material, depth ≥ 4, attacker not to move), the gen-001 sealed test (still closed), any hierarchy/search/self-play. Test and G1 are now spent for selection purposes.

## 8. Repository note
Earlier build artifacts (~106 MB) were accidentally committed in `3fd8ad6` and removed in `27612ed`; the blobs remain in the V69 remote history. No force-push has been done without approval; a history rewrite can be done on request.
