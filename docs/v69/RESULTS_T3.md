# Recur64 V69 — T3 results: held-out material-family transfer

Contract: `T3_CONTRACT.md` (committed before any T3 model was trained). Evidence: `evidence/t3/`. Labels: **MEASURED** (machine-written report), **INFERRED**, **NOT RUN**.
Setup (MEASURED): for each fold the unchanged M\* recipe (**S**: symmetry-averaged, LR 3e-3, 30 epochs) and P\* recipe (**D**: random-symmetry, LR 3e-3, 60 epochs) were trained on the T2 train rows of two families (48,000) and evaluated on all 24,000 T2 train rows of the third family, never trained on in that fold. No tuning, selection or early stopping. Baseline B2F = the shallow baseline refit on the same two families. No sealed data was read (T2 test, G1, gen-001 sealed, T1 test). 6 runs, all completed (≈ 2.6 min each for D, ≈ 3.9 min for S).

## Result: no fold passes for either recipe — "family-specific" (MEASURED)

| Held-out family | Recipe | Balanced acc. | BCE | B2F BA / BCE | Gain vs B2F (pp), 95% CI | Derangement drop (pp) | In-family M\*/P\* on T2 test (ref.) |
|---|---|---|---|---|---|---|---|
| KQQvK | S | 0.710 | 3.90 | 0.746 / 0.521 | −3.6 (−7.6, +0.4) | 20.6 | 0.893 |
| KQQvK | D | 0.700 | 0.845 | 0.746 / 0.521 | −4.6 (−7.7, −1.2) | 19.5 | 0.808 |
| KQRvK | S | 0.798 | 0.998 | 0.762 / 0.492 | +3.6 (+0.4, +6.3) | 29.6 | 0.898 |
| KQRvK | D | 0.793 | 0.461 | 0.762 / 0.492 | +3.1 (+0.5, +5.4) | 29.1 | 0.843 |
| KRRvK | S | 0.769 | 1.33 | 0.813 / 0.467 | −4.4 (−9.4, +0.0) | 26.3 | 0.946 |
| KRRvK | D | 0.774 | 0.540 | 0.813 / 0.467 | −3.9 (−8.6, −0.2) | 27.5 | 0.894 |

Pass rule (per fold): BA ≥ 0.75, BCE ≤ 0.55, gain ≥ 5 pp over B2F with the interval excluding zero and BCE no worse, derangement drop ≥ 15 pp. Folds passing: S 0/3, D 0/3.

## What this shows (MEASURED facts, INFERRED reading)
- **MEASURED:** Out of family, both recipes fall to the level of a shallow baseline fit on the same data (BA 0.70–0.80 vs B2F 0.75–0.81), i.e. 5–18 points below their in-family accuracy. The only fold where a model exceeded B2F with an interval excluding zero is KQRvK (+3.1 / +3.6 pp, below the 5 pp requirement and BCE no better for S); in KQQvK and KRRvK the models are at or below B2F.
- **MEASURED:** The board-swap control still drops accuracy to chance-level (drops 19–30 pp), so out-of-family predictions are still board-driven; the models are not failing by ignoring the board.
- **MEASURED:** Out-of-family calibration is poor, worst for the symmetry-averaged recipe (BCE 1.0–3.9 even when BA is ~0.7–0.8); D is better calibrated (0.46–0.85). Validation on the held-out family (diagnostic) matches the held-out accuracy (e.g. KQRvK 0.79–0.80), so the fold numbers are not a quirk of the large held-out set.
- **INFERRED:** T2's large gain over the baselines is mostly family-specific pattern knowledge, learned per material combination, not a general forced-mate concept shared across KQQ/KQR/KRR. With only two training families per fold the test is also a hard one (the held-out family has different piece interactions), so this is a lower bound on cross-family generalization, not a statement that nothing general is learned. The consistent weakness is on n = 2 positives (e.g. KQQvK 0.39–0.45, KRRvK 0.50–0.64), where forced mate needs two-move lookahead.

## Caveats
- KQQvK and KRRvK held-out sets have only 30 and 28 connected groups (very large connected components), so their cluster-bootstrap intervals rest on few independent units and the 5 pp rule is a weak test there; KQRvK has 230 groups. The qualitative picture (models ≈ baseline out of family) holds in all three folds.
- The in-family references are T2 test per-family accuracies from the spent T2 report, models trained on all three families with more data than any fold; they are descriptive, not a controlled comparison.
- Folds hold out whole families, so no claim is made about other kinds of shift (new depth, different material, attacker-side changes).
- Precision: CUDA f32 storage/accumulation with possibly-TF32 matmul (not strict FP32); baselines host f64.

## NOT RUN
Training on all families with a family-id-free regularizer, larger models/data aimed at cross-family transfer, hierarchy/search/policy, new material families, depth ≥ 3, calibration/temperature fitting, the sealed gen-001 test (still closed).
