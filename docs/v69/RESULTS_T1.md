# Recur64 V69 — T1 results: scale + exact symmetry augmentation vs the shallow baseline

Contract: `T1_CONTRACT.md` (+ amendments A1, A2). Evidence: `evidence/t1/`. Labels: **MEASURED** (from run artifacts), **INFERRED** (my reading), **NOT RUN**.
Scope: same-domain generalization (KQQvK/KQRvK/KRRvK, attacker to move, exact M2/M3 child labels). No claim about architecture, recurrence, hierarchy or move selection.

## 1. Headline (MEASURED, sealed T1 test, 3,072 rows, 752 connected groups)
| candidate | bal. acc | BCE | AUROC | real − derangement (pp) | 8-fold TTA bal. acc / BCE (report-only) |
|---|---|---|---|---|---|
| **M\*** (MLP, k=2000, d8, LR 3e-3) | **0.7656** | **0.4754** | 0.852 | 25.6 | 0.7998 / 0.4280 |
| **A\*** (E1 one-pass, k=2000, d8, LR 5e-4) | 0.5801 | 0.6659 | 0.615 | 6.8 | 0.5798 / 0.6637 |
| B (frozen D1 shallow baseline) | 0.7725 | 0.4805 | 0.852 | 27.9 | – |
| B2 (same baseline refit on T1 train) | 0.7734 | 0.4760 | 0.854 | 28.1 | – |

Paired cluster bootstrap (5,000 resamples, identical group draws), balanced-accuracy differences with 95% intervals:
M\*−B −0.7 pp [−2.7, +1.3]; M\*−B2 −0.8 pp [−2.8, +1.2]; A\*−B −19.2 pp [−21.3, −17.2]; A\*−M\* −18.6 pp [−20.4, −16.7]; B2−B +0.1 pp [−0.5, +0.8].
Spent G1 panel (secondary, report-only, 1,536 rows): M\* 0.770 / BCE 0.465; A\* 0.581 / 0.673; B 0.764; B2 0.771 (M\*−B +0.6 pp [−2.1, +3.3]).
Erasure (all-empty board) gives balanced accuracy 0.500 for every candidate on both sets (MEASURED).

## 2. Pre-registered decisions (MEASURED against the registered rules)
- **M\* meets the transfer criterion** (BA ≥ 0.75, BCE ≤ 0.55, derangement drop ≥ 15 pp, integrity pass). **A\* does not** (BA 0.58, BCE 0.67, drop 6.8 pp).
- **Neither neural candidate improves on the baselines** (rule: gain ≥ 5 pp with the paired interval excluding 0 and BCE no worse). M\* is statistically indistinguishable from B and B2; A\* is clearly worse.
- Same-domain read-out (INFERRED): with ~30× more data and exact D8 augmentation the direct-board MLP *does* learn a transferable mate-classification signal (0.50 → 0.77 BA; G1 panel 0.53 for the 768-row MLP → 0.77), but it only **matches** the 23-feature shallow baseline; it does not exceed it. The shallow baseline is not improved by more data (B2 ≈ B), suggesting its feature ceiling (~0.77) is the limiting level that M\* reaches.
- A (E1 one-pass workspace model) did not generalize in its registered budget (4,500 updates at most): INFERRED that the registered A grid was under-trained/under-sized relative to D3, which needed ~4,000 updates only to *fit* 768 rows; this does not show A cannot learn it with a longer schedule (NOT RUN: longer A schedules, other LRs).

## 3. Validation-only grid (MEASURED; descriptive, not causal)
Final-update validation (1,536 rows), selection by lowest BCE per family:
- **M, k=2000:** d8 beats off at every LR (LR 3e-3: BCE 0.443/BA 0.784 vs 0.961/0.736; LR 1e-3: 0.481/0.773 vs 0.615/0.745); LR 3e-4 under-fits (0.620/0.659).
- **M scaling (d8, LR 3e-3):** k=250 BCE 0.666 (BA 0.579) → k=1000 0.514 (0.750) → k=2000 0.443 (0.784). Without augmentation the high LR overfits badly (train loss 0.12–0.17 with val BCE 0.96–1.33), so augmentation and more data both matter; the best configuration is at the edge of the grid (largest k, highest LR) — INFERRED that more data/higher LR could still help (NOT RUN).
- **A:** k=2000 d8 0.662/0.590; k=2000 off 0.677/0.546; k=1000 d8 and k=250 d8 stayed at chance (0.693, BA 0.500).
Test-time 8-fold symmetry averaging lifted M\* test BA from 0.766 to 0.800 and BCE 0.475 → 0.428 (report-only variant, **not** a registered candidate; selected nothing on the test) — a pointer for a follow-up, not a result.

## 4. Data and integrity (MEASURED)
- One fresh draw (seed fingerprint b52ad95c925672dc); T1 train 24,000 / val 1,536 / test 3,072; per cell exact quotas 2,000/128/256; per-root cap 2/class; nested subsets k = 250/1000/2000.
- Identity-level exclusion against gen-001 and the G1 pool: 73,004 of 108,387 accepted roots excluded (67%); audit found 0 canonical overlaps with prior data; 0 cross-partition canonical-child identities.
- Audit (`audit_pass: true`, 0 failures): all 28,608 examples re-queried exactly, 8,544 contributing roots re-analysed with an empty cache, 216 independent-reference child checks, 30 M3 minimal-depth checks, **exact label invariance under all 8 board transforms on a 1,500-example sample (10,500 fresh oracle queries)**.
- Preservation receipts start and end: 363 files, all hashes unchanged. Frozen training inputs (`frozen_train.json`) and final inputs (`frozen_final.json`) enforced by hash at run time; evaluator source frozen and matched by digest; evaluator verified on a validation-only fixture (max logit difference vs recorded ≤ 2e-15).
- Precision: f32 storage/accumulation, matmul inputs possibly TF32 (not strict FP32); baselines host f64. CUDA only; no fallback.

## 5. Deviations and failed/unrun items (kept visible)
1. **A1** (before any row existed): component-level partitioning was infeasible (percolation); replaced by root-level keyed 80/10/10 partitioning.
2. **A2** (after the A1 rows existed): the first full audit **failed** (4 failures). Causes were mine: the cross-partition taken-set was keyed on (child, budget) so a position could occur in two partitions at different budgets; and the nested-scale check used 24k instead of 12k rows per scale. Fix: taken-set keyed on the canonical child; audit formula corrected; same seed reused (only generation/audit statistics had been seen; no model had been trained); data regenerated; the failed attempt is preserved (`evidence/t1/audit_attempt_A1_failed.log.out`, `audit_receipt_A1_failed.json`). A2 was written after, not before, the failed audit — this is a post-hoc fix of a specification defect, disclosed here.
3. First smoke test of the trainer ran without the NVRTC PATH and produced garbage (discarded; smoke runs `*_smoke` excluded from selection); first `freeze-final` listed two files the evaluator/aggregator roles may not read, so verification failed **before any test or G1 data was opened**; the lists were corrected, the unusable freeze/source/verification files were archived, and everything was re-run (second pass used for all results).
4. Build artifacts were accidentally committed earlier in the branch history (commit 3fd8ad6, `crates/target-v69`, ~106 MB) and removed in 27612ed; the blobs remain in remote history because no force-push was authorized.
5. NOT RUN: A with longer/other schedules; M beyond k=2000 or LR > 3e-3; TTA as a registered candidate; hierarchy/search/self-play; independent sealed gen-001 test.
6. Total native training time ≈ 40 minutes (22 runs), far under the 3.5 h limit; host ≤ 1.0 GiB, device ≤ 0.8 GiB sampled.

## 6. Recommendation (INFERRED)
The ceiling observed so far is the shallow-feature level (~0.77 BA). Next logical step, to be registered separately: (a) extend M along the observed trends (larger fresh training sets, LR, symmetry-averaged inference trained end-to-end) to test whether it can pass the baseline, and (b) give A a longer, properly scaled schedule with the same augmentation. Only if a neural model clearly exceeds the shallow baseline does a representation/architecture claim become supportable.
