# Recur64 V69 — G1 (revision R1): fresh same-domain generalization of three frozen candidates — results and decision

Supersedes the stopped attempt in `RESULTS_G1.md`. Contract: `G1_CONTRACT.md` + `G1_AMENDMENT_R1.md` (+ `g1_config_r1.json`), frozen before the R1 seed was drawn. Evidence: `evidence/g1r1/`. Labels: **MEASURED**, **INFERRED**, **NOT RUN**.
G1 is **same-domain generalization on one fresh evaluation-only panel**; it is not an architecture, recurrence, hierarchy, chess-reasoning or move-selection claim. Nothing was trained, tuned, calibrated, ensembled or thresholded.

## Decision

**The two neural candidates do not generalize; the shallow baseline does.** On 1,536 fresh, audited, gen-001-disjoint examples:

| | balanced acc | BCE | AUROC | fit BA | fit→G1 gap | derangement drop (real − recipient) | transfer criterion |
|---|---|---|---|---|---|---|---|
| **A** (D3-A, 1.88 M) | **0.503** [0.480, 0.525] | 4.628 [4.37, 4.88] | 0.508 | 0.995 | 49.2 pp | 0.7 pp [−2.8, 4.1] | **fail** |
| **M** (D3-M MLP) | **0.529** [0.509, 0.548] | 1.543 [1.43, 1.67] | 0.533 | 0.897 | 36.8 pp | 3.8 pp [0.5, 7.0] | **fail** |
| **B** (D1 shallow logistic) | **0.764** [0.741, 0.785] | 0.481 [0.459, 0.504] | 0.850 | 0.764 | 0.1 pp | 27.5 pp [24.2, 30.6] | meets BA/BCE thresholds |

(95% intervals: 5,000 paired cluster-bootstrap resamples over the 393 connected G1 groups; identical draws for every candidate and comparison.)
Paired differences (95% CI): **A − B** balanced accuracy −0.261 [−0.293, −0.229], BCE +4.147 [+3.893, +4.398]; **M − B** −0.235 [−0.263, −0.206], BCE +1.062 [+0.946, +1.190]; **A − M** −0.026 [−0.052, −0.000], BCE +3.085 [+2.843, +3.325].
No candidate improves on the baseline (A and M are 23–26 points *worse*). By the pre-declared rule — *neural models fail while the baseline transfers* — the priority is **representation and generalization, not more fitting.**
Accuracy transfer and calibration are reported separately: the neural models fail both (near-chance accuracy and BCE far above the 0.693 chance level: confident wrong answers); the baseline passes both. D3's "A fits the fitting set" result was memorization of 768 rows.

## 1. Frozen protocol, identities, provenance — MEASURED

- Source: reviewed head `7afc00a` (stopped G1) → amended, tested and committed as `13227499…` (clean tree) before the R1 freeze; the evaluator ran at exactly that source (digest `f0a0e8c2…`, enforced at launch).
  Pushed to `origin` on this branch only; never merged.
- **Order of events:** R1 amendment + config committed → R1 protocol frozen (`g1r1/frozen_protocol.json`, sha `8b18e4f5…`, 306 prior files verified, **no seed existed**) → fresh seed drawn → generation → audit → intervention map frozen → evaluator verified → evaluator source + data frozen (`frozen_g1.json`, sha `7eb64440…`) → one evaluation per candidate → aggregation.
- **Seed policy (agent's choice, owner-delegated): fresh draw.** Fingerprint **`8a82104fcb055419`**, recorded 2026-10-09T03:12:26Z before generation, write-once. Reason: R1 was selected after observing the first seed's generation statistics; a fresh draw removes the question of seed-informed design. The first seed (`2fef6b12a6cc5faa`) is abandoned and was never used for data.
- Candidates (hash-verified at launch; provenance binds candidate bytes, evaluator source, data manifest `0b45df62…`, protocol and intervention map `74d4d568…`): A = D3-A final update-12,000 `model.mpk`/`meta.json`; M = D3-M final; B = `d1/baseline/model.json` with its stored standardization (not recomputed). Threshold logit 0. Neural candidates loaded for inference only (the evaluator role cannot read optimizer files, G1 metadata, gen-001 data or the exclusion index; tested).
- **Preservation (append-only receipts):** 306 files (E1, D1, D2, D3 and attempt-1 `g1/` manifests + gen-001 manifest) re-hash identically before and after. gen-001's sealed test was never opened by model code.
- Precision: CUDA f32 storage/accumulation, matmul inputs possibly TF32 (not strict FP32); baseline inference is host-side f64 (authorized, not a fallback).

## 2. Fresh data, exclusion, audit — MEASURED

Generation (deterministic rounds of 1,000 attempts per family, 5,000,000-node teacher bound): feasible at round 3 in **1.6 s**; per family attempts 4,000 (KQQvK: 885 M2 + 172 M3 accepted, 348 M1 rejected; KQRvK: 627 + 525, 271 rejected beyond 3; KRRvK: 596 + 439), illegal-state rejections (overlap/adjacent kings/defender in check) counted, duplicates 1/2/3 (symmetry-equivalent roots), **0 unresolved or timed-out teacher calls**.
Accepted roots 3,244 in 1,730 connected groups. **R1 exclusion removed 1,285 roots (40%)** — 4 by root identity, 1,285 by child identity — leaving 1,959 roots in 1,392 groups (largest 18 roots; 1,089 singletons); **no quota shortfall at round 3** (under the original whole-component rule the same domain was infeasible).
Panel: 1,536 examples, 128 per family × remaining-budget × class cell, from 456 roots in 393 connected groups (per-root cap ≤ 2 per class honoured); model rows (`g1_rows.jsonl`, sha `54ec00f7…`) are stored separately from root/group metadata.
**Audit (data-only, before any inference) — PASS, 0 failures:** manifest hashes; exact quotas/ids; row↔metadata; canonical identities recomputed from FEN; root–child membership and stored statuses; terminal exclusion; **empty-cache re-analysis of all 456 contributing roots and a fresh exact re-query of all 1,536 targets**; **0 canonical overlaps with gen-001** (full 65-byte keys; the stored exclusion index was rebuilt from the gen-001 pool and matched); groups rebuilt; independent-reference child checks (96) and a bounded sample of **24 M3 minimal-depth checks** — all agreed.
Reference shared dependencies: the `cozy-chess` move generator/legality (cross-checked against brute-force `is_legal` in tests), the FEN parser and the domain definition; **not** the search, cache, node budget or `rules::classify` code.
**Limitations:** zero accidental overlap with the forbidden historical V5/V6 datasets is **not certified** (they were not inspected). R1 guarantees identity-level separation from gen-001, not transitive group-level separation; candidates trained only on the 768 gen-001 fitting rows.

## 3. Evaluator qualification — MEASURED (before G1 contact)

On the fitting-only fixture (D2 `s768` rows): A and M reproduce the recorded D3 update-12,000 fitting predictions (max |Δlogit| 1.8e-15; 768/768 classifications agree; tolerance 2e-3), the baseline reproduces its recorded D1 fitting predictions (8.9e-16; tolerance 1e-9), features are independent of labels and ids, and the erasure definitions are consistent. New append-only receipt (`evaluator_verification.json`); old predictions untouched; no G1 label or output was used.

## 4. Controls — MEASURED

- **Derangement** (map frozen before any prediction; label-independent, bijective, no fixed points, within family × budget cells; label agreement of donor map 48.8%): every deranged prediction equalled the donor's ordinary prediction exactly (max |Δlogit| 0.0 for all three; tolerances 2e-3/1e-9).
  B: balanced accuracy 0.489 vs recipient labels, **0.764 vs donor labels** — it reads the board, predicting the donor's label. A: 0.496 recipient / 0.503 donor; M: 0.491 / 0.529 — **the neural models' predictions are no more aligned with the board's own label than with a random board's**, i.e. nothing board-specific transferred.
- **Erasure (OOD diagnostic, definitions in the contract):** A and M predict 0% positive (BA 0.500; BCE 2.75 / 0.74); B (all board-derived features replaced by fit means) is also at 0.500 (predicts 50% positive; BCE 0.734). Not causal proof.
- **Confidence on errors:** A mean |logit| 9.26 on errors vs 9.48 on correct (logit sd ≈ 10 in both classes); M 2.91 vs 3.00; B 0.68 vs 1.41. The neural models are as confident when wrong as when right; the baseline is less confident when wrong.
- Per-cell accuracy: A 0.34–0.63, M 0.45–0.66 (all near chance, no stable cell); B 0.62–0.89 (every cell above chance; KQQ/n1/pos 0.89, KQR/n1/pos 0.88; weakest KQR/n2/neg 0.62).

## 5. Cost — MEASURED (separating startup/JIT; batch size stated)

| | startup (CUDA init + load) | first full pass incl. JIT (1,536 rows, batch 16) | warmed synchronized batch-16 median | warm 1,536-row inference only | peak host / sampled device (MiB) |
|---|---|---|---|---|---|
| A | 160 ms | 3,219 ms | 7.40 ms | 733 ms | 348 / 159 |
| M | 93 ms | 724 ms | 0.33 ms | 39 ms | 265 / (sample 0) |
| B (host f64) | 0.5 ms | n/a | n/a | feature extraction incl. FEN parse and legal-move/capture features 1.7 ms for 1,536 rows + 0.03 ms logits | ~0 (launcher sample) |

End-to-end evaluator process wall: A 5.9 s, M 1.1 s, B 0.14 s (includes row parsing/feature extraction). The baseline is the cheapest *and* the best candidate. Memory sampling limitation: launcher polls whole-GPU nvidia-smi every ≈ 2 s and the host process at 500 ms; short peaks can be missed; allocator high-water is unavailable. Total native time for G1 ≈ 25 s (generation 1.6 s, audit 0.7 s, evaluations 9 s) of the 2 h budget; every cap respected, all exits 0, 0 orphans.

## 6. Independent recomputation — MEASURED

`v69-g1 aggregate` (separate code from the evaluator) recomputed every metric in f64 from serialized predictions and G1 reporting metadata, verified 12 frozen hashes, prediction-file hashes against each provenance, the frozen-file binding, donor consistency and the frozen map; the same 5,000 resamples were used for all comparisons.

## 7. MEASURED / INFERRED / NOT RUN

MEASURED: §1–§6. INFERRED: (a) the neural candidates' high fitting accuracy came from memorizing the 768 rows (fit→G1 gaps of 37–49 points, no board-specific signal under derangement); (b) a shallow, global mobility/capture feature set is learnable and transferable from 768 examples, whereas raw-board networks need either far more data or an inductive bias that makes legal-move/attack structure easy to read;
(c) a 12,000-update run that "fits" is not evidence of a useful representation. NOT RUN: any training, tuning, calibration, ensemble, threshold change, repeat evaluation, additional draw, hierarchy fit, policy, search, self-play, gen-001 sealed-test inference, historical V5/V6 comparison.

## 8. Recommendation and next-agent prompt

By the pre-declared rules: **prioritize representation and generalization rather than more fitting.** Do not extend the D-series; the 768-row set is the bottleneck, and the G1 panel is now a spent one-shot evaluation (reusing it for selection would contaminate it).
Concrete next phase (separately registered): a **representation/generalization campaign built on exact, cheap data**, evaluated on a *new* sealed test drawn the same way:
(1) draw a new multi-partition dataset (the V69 generator makes thousands of exact-label roots per second): large fitting set (e.g. ≥ 20k examples), validation, and a sealed test — with R1-style identity exclusion against *all* existing V69 data including the G1 panel;
(2) use the **exact label-preserving 16-fold symmetry group** (8 board symmetries × attacker-colour relabelling) as data augmentation, and add the baseline's global features (legal-move count, attacked/captured major pieces) as an auxiliary-target or input channel only if pre-registered as a representation intervention;
(3) compare models on the same data under frozen candidates (A, M, baseline), one change at a time, with cluster-bootstrap uncertainty and the derangement/erasure controls; keep the shallow baseline as the standing bar.

> Continue on branch `experiment/hp-v69-two-clock-workspace` (read AGENTS.md and docs/v69/{CONTRACT,MODEL_SPEC,RESULTS_E1…D3,G1_CONTRACT,G1_AMENDMENT_R1,RESULTS_G1,RESULTS_G1_R1}.md). Do not reuse the spent G1 panel for selection or tuning. **Do nothing that draws new data until the owner authorizes it in the session.** If authorized: write a T1 contract (data partitions with sealed test, identity exclusion against gen-001 and G1, augmentation definition and tests for label-preservation under the 16-fold symmetry, models/recipes, metrics, decision rules, limits) and freeze it before drawing a new OS-entropy seed; generate/audit with the existing teacher; train only the pre-registered configurations; evaluate once on the sealed test; report against the shallow baseline. No hierarchy claims, search, policy or self-play.
