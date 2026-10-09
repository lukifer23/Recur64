# Recur64 V69 — G1 contract: fresh same-domain generalization evaluation of three frozen candidates

Frozen **before the G1 generation master seed is drawn**. Machine-readable parameters: `g1_config.json` (hash-enforced).
Authorization: the owner explicitly authorized **one** fresh evaluation-data draw and the registered evaluations below; no training or tuning is authorized.
G1 is **same-domain generalization** (KQQvK/KQRvK/KRRvK, attacker to move at roots, immediate nonterminal children, exact forced-mate-within-n labels). It is
not an architecture proof, not evidence about recurrence/hierarchy, broad chess reasoning or move-selection utility.

## 0. Boundaries

E1–D3 and gen-001 artifacts are preserved byte-for-byte (append-only preservation receipts before and after). Gen-001's sealed test stays closed to all model code. No
historical V5/V6 positions, labels, caches or model outputs are used; **zero accidental overlap with those forbidden datasets is not certified** (they are not inspected). No
retraining, tuning, calibration, ensemble, candidate/checkpoint choice, threshold search, repeat evaluation for favourable results, gen-001 sealed inference, hierarchy fit, policy, search or
self-play. Precision: CUDA f32 storage/accumulation, matmul inputs possibly TF32 (not strict FP32); the baseline runs host-side in f64 (authorized; not a GPU fallback).

## 1. Candidates (frozen; hashed; no alternatives)

A = D3-A final (update-12,000) checkpoint; M = D3-M final (update-12,000) checkpoint; B = the D1 shallow logistic baseline with its original standardization statistics and weights
(`d1/baseline/model.json`, 23 features, threshold logit 0). Actual model/metadata files and their D3/D1 provenance are hashed in `g1/frozen_protocol.json` and re-verified at every launch.
Neural candidates load for inference only (no optimizer state).

## 2. Fresh evaluation data

One new G1 master seed (32 bytes from the OS CSPRNG via `scripts/v69/new_seed.ps1`), fingerprint recorded **before** generation, never redrawn or overwritten. Domain-separated streams:
`generation/<family>` (attempt index), `selection/root`, `selection/child`, `g1/intervention/<family>/<budget>`, `g1/bootstrap`. No model-init/training stream exists.
Generator, teacher and symmetry code are the audited V69 code (`sample_root`, `analyze_root` with a ≤ 5,000,000-node-per-query bound, `canonical_key`, `build_groups`): same domain (pawnless, no castling/ep,
halfmove 0, no history, legal placements, defender not in check), exact minimal mate depth M ∈ {2,3} with absence below proven, complete legal root move set, children classified by `rules::classify` first (terminal children excluded),
remaining budget n = M − 1, target = original attacker forces mate within n (UNKNOWN/resource-limited ⇒ root discarded, never negative). No neural selection. A narrow evaluation-only adapter (`g1.rs`) performs
exclusion and selection; it does not change legality, targets, sampling distribution, class quotas or symmetry definitions.
**One evaluation-only panel:** 1,536 examples = 128 per family × remaining-budget × class cell; deterministic keyed selection; per-root cap ≤ 2 positive and ≤ 2 negative children; a canonical child already
selected at the same budget is skipped. The panel is never called training data and no training partition is created.

## 3. Independence and exclusion

A data-only role builds a contamination-exclusion index from gen-001's accepted roots (canonical root keys) and **all** their immediate children (canonical child keys, every legal move, terminal included), full 65-byte
identities (not hashes). New roots are grouped into connected components (shared canonical children); **any component containing a root whose canonical root key or any child key occurs in the gen-001 index is removed
entirely.** The index supplies no targets or features. Reported: attempts, illegal/duplicate/rejected/unresolved/timed-out counts, exclusions (roots, groups), accepted roots, group counts and sizes, cell coverage, and the effect of
exclusion on sampling. Symmetry-equivalent positions are deduplicated within G1; related roots stay in connected groups; model rows are stored separately from root/group metadata.
If exact construction or audit fails, stop before inference: no reduced quotas, substitutions, changed seed or retry.

## 4. Audit (data-only, before any inference)

Manifest hashes; exact quotas and ids; row ↔ metadata correspondence; canonical identities recomputed from FEN; root-child membership and stored statuses; terminal exclusion; empty-cache re-analysis of every contributing root and
a fresh exact re-query of all 1,536 targets; cross-gen-001 exclusion re-verified against the index with full keys; independent-reference (`reference.rs`, no cache/budget/`rules`) child checks on a deterministic per-cell sample, and a bounded
sample of M3 minimal-depth checks. **Reference shared dependencies:** the `cozy-chess` move generator and legality (cross-checked against brute-force `is_legal`), the FEN parser, and this crate's definition of the domain; it does not share the search, caching or terminal-rule code.

## 5. Roles and evaluator

Roles: **G1 builder/auditor** (reads gen-001 pool for exclusion verification, the G1 seed; writes `g1/`), **frozen-candidate evaluator** (reads only the G1 model rows, the frozen candidate files and the frozen intervention map; denied root-bearing G1 metadata and all of gen-001 validation/sealed/pool and
any historical data), **metric aggregator** (joins predictions with G1 reporting metadata; also reads the D3/D1 fitting reports for the fit-to-G1 gap). Targets and ids never enter model features; existing feature encodings are unchanged; the baseline uses its exact extractor and stored standardization (never recomputed on G1).
**Evaluator verification before G1 contact:** on the fixed existing fitting-only fixture (the 768 fitting rows, D2 `s768`), verify candidate loading, feature encoding, target separation and forward agreement with the recorded D3 update-12,000 fitting predictions (neural: |Δlogit| ≤ 2e-3, the documented TF32-class tolerance; baseline: 1e-9 against its recorded fitting predictions); new append-only receipt; old predictions untouched; no G1 labels or outputs may be used to repair the evaluator.
The verified evaluator source is frozen (executable-source digest recorded; the evaluator refuses to run on a different digest or a dirty tree).

## 6. Registered evaluation (one invocation per candidate)

Each candidate gets exactly one invocation producing: ordinary G1 predictions; **fixed board derangement**; **fixed board erasure**. The derangement map is frozen (label-independent, bijective, no fixed points, within family × remaining-budget cells, keyed stream
`g1/intervention/<family>/<budget>`, complete board instance replaced coherently; recipient budget kept) before any candidate prediction exists. Reported against recipient and donor labels, with a check that each deranged prediction equals the donor's ordinary prediction within tolerance (neural 2e-3, baseline 1e-9).
**Erasure:** A, M — all 64 squares set empty, retaining only the legitimate task metadata (remaining budget, attacker-to-move indicator, the constant rule scalars); B — every board-derived feature replaced by its fit mean (no mobility/capture/geometry information retained), budget indicator kept, budget interactions recomputed from those constants.
Erasure is an out-of-distribution diagnostic, not causal proof. A failed invocation stays visible; stop and report; no repeats.

## 7. Metrics and comparisons

Per candidate: balanced accuracy, accuracy, BCE, Brier, AUROC, confusion, per family/budget/class results, logit distributions and confidence on errors, fit-to-G1 gap (fitting numbers from the D3/D1 reports), derangement and erasure results.
**5,000 paired cluster-bootstrap resamples over connected G1 groups, the same resamples for all candidates and comparisons** (rows are never resampled independently). Paired differences with 95% intervals (accuracy, balanced accuracy, BCE):
A − B, M − B, A − M. Latency: warmed-up synchronized inference (state batch size), plus end-to-end cost including parsing and feature extraction (notably the baseline's legal-move computation), with startup/JIT costs reported separately.

## 8. Prospective decision rules

*Preliminary same-domain transfer (A and M):* G1 balanced accuracy ≥ 75%; G1 BCE ≤ 0.55; real − derangement-vs-recipient balanced accuracy ≥ 15 points; all engineering, custody and integrity checks pass.
*Practical improvement over the baseline:* balanced-accuracy gain ≥ 5 points, paired 95% interval excludes zero, and BCE no worse than the baseline. Accuracy transfer and calibration are reported separately; missing one threshold is not proof of no useful signal; passing proves nothing about recurrence, hierarchy,
broad chess reasoning or move selection. Outcomes → A transfers and improves: fresh controlled architecture/generalization campaign; M transfers better or is comparable at much lower cost: M is the practical control/candidate; neural fail while B transfers: representation and generalization first, not more fitting; all fail: audit distribution, domain and sampling first.

## 9. Resources, provenance, preservation

≤ 2 h total native execution (excl. compilation) with a remaining-budget check: generation + audit ≤ 60 min, each candidate evaluation ≤ 10 min, aggregation ≤ 20 min; teacher ≤ 5,000,000 nodes/query; host ≤ 8 GiB; sampled device ≤ 3,072 MiB; bounded launcher, native exit codes, process-tree cleanup, orphan checks; sampling limitations reported.
`g1/frozen_protocol.json` (candidates, contract/config, prior-phase manifests) is frozen before the seed; `g1/frozen_g1.json` (data manifest, rows, map, evaluator source digest, protocol) before inference; every launch re-verifies its group. Predictions are bound to candidate-file bytes, evaluator source identity, data manifest, protocol and intervention identities.
Append-only preservation receipts (E1, D1, D2, D3 manifests, gen-001 manifest) are written before and after. Stop after G1; no follow-up is executed.
