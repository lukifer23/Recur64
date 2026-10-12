# Recur64 V69 — T3 contract: does the winning recipe transfer to an unseen material family?

Frozen before any T3 model is trained. Authorization: the owner, after `RESULTS_T2.md` ("proceed with what you think we should do next, I trust you"). Stop after T3; no follow-up is executed.

**Motivation (`RESULTS_T2.md`).** M\* (symmetry-averaged MLP, 0.912 balanced accuracy) and P\* (random-symmetry MLP, 0.848) clearly beat the shallow baselines on the sealed T2 test, but test positions share their three material families with training. This leaves open whether the models learned something family-general about forced mate or family-specific patterns.
**Question.** Train the *unchanged* M\* and P\* recipes on two of the three families (KQQvK, KQRvK, KRRvK) and evaluate on the third family, never seen in that fold's training. Does the model still classify forced mate better than a shallow baseline fit on the same two families? Same-domain only (mate within n = 1 or 2, attacker-relative inputs); no architecture, recurrence, hierarchy, search or policy claim.

## 0. Data (no new generation, no sealed data)
Only T2 files that were already open for training/validation: `t2/rows/train.jsonl` (72,000 rows; 24,000 per family; 6,000 per family × budget × class cell), `t2/rows/val.jsonl`, and their metadata (`family`, `group_id`). **Not read:** the T2 test rows/metadata, the G1 panel, T1/gen-001 sealed data, anything under `*/sealed*`. The T2 audit already proved partition isolation and exact labels. Families have disjoint material, so a fold's held-out rows share no position or root with that fold's training rows.
For fold F ∈ {KQQvK, KQRvK, KRRvK}: **train** = T2 train rows of the other two families (48,000); **held-out set H** = all 24,000 T2 train rows of family F (never trained on in this fold); **diagnostic** = T2 val rows of the two trained families (2,048) and of F (1,024).

## 1. Runs (pre-registered; nothing else is run; no tuning, no selection, no early stopping)
Recipe unchanged from T2 (batch 64, AdamW wd 1e-4 on weights, global clip 1.0, 200-update warmup then cosine to 0.1·peak, stable BCE, final update used, keyed per-epoch permutations, D1 MLP init 842→128→64→1):
- **S** (the M\* recipe): mode sym (BCE of the 8-symmetry mean logit; inference = 8-view mean), peak LR 3e-3, 30 epochs.
- **D** (the P\* recipe): mode d8 (per-sample keyed random symmetry), peak LR 3e-3, 60 epochs.
3 folds × 2 recipes = 6 runs. Baseline per fold: **B2F** = the D1 baseline recipe (23 features, 1000 GD steps, lr 0.05, L2 0.01, standardized on its fitting rows) refit on that fold's training rows, host f64. The frozen D1 baseline B (fit on gen-001, all families) is report-only.
Controls on H (label-independent): board derangement (cyclic shift of a keyed shuffle within budget cells; score against recipient labels) and the in-family reference: M\* / P\* per-family balanced accuracy on the T2 test taken from `evidence/t2/t2_report.json` (already-spent report-only reference, cells are equal-sized).

## 2. Metrics and decision rules (prospective)
Balanced accuracy, accuracy, BCE, Brier, AUROC, confusion, per budget × class cell, on H. 5,000 paired cluster-bootstrap resamples over `group_id` (identical draws for all comparisons).
**Held-out-family transfer (per fold and recipe):** BA(H) ≥ 0.75, BCE(H) ≤ 0.55, and gain over B2F ≥ 5 points with the paired 95% interval excluding zero and BCE no worse; derangement drop ≥ 15 points.
**Summary claim:** "family-general" for a recipe only if all three folds pass; "partially family-general" if at least one but not all pass; otherwise "family-specific". The T2 in-family reference is described, not tested.

## 3. Limits and provenance
≤ 2 h native execution in total (bounded launcher, native exit codes, process-tree cleanup); host ≤ 8 GiB; sampled device ≤ 3,072 MiB. CUDA only (f32 storage, matmul possibly TF32; no CPU fallback), baselines host f64. The T3 tool refuses to open any path containing `test`, `g1`, or `sealed`. It requires a clean git tree and records source digest, input hashes and output hashes in append-only per-run provenance; reruns of a run id are refused. Failed or unfinished runs are reported. Findings are descriptive; effects are not causal proofs.
