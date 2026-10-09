# Recur64 V69 — G1 fresh generalization evaluation: STOPPED at data construction (no inference run)

Contract: `G1_CONTRACT.md` (+ `g1_config.json`), frozen before the seed was drawn. Evidence: `evidence/g1/`. Labels: **MEASURED**, **INFERRED**, **NOT RUN**.

## Decision

**G1 is blocked before any candidate saw G1 data: the frozen exclusion rule cannot produce the required 1,536-example panel.** Per the registered stop rule ("if exact construction or audit fails, stop before
inference; no reduced quotas, substitute examples, changed seed or automatic retry"), I stopped. **No G1 rows, labels, predictions or comparisons exist; no candidate was evaluated on G1; no result about
generalization was obtained.** A concrete protocol revision for the owner to decide on is in §4; nothing was changed unilaterally.

## 1. What was completed — MEASURED

- Source: reviewed head `0a69c2cb…` → this report is bound to `258e987…` (clean tree). Pushed to `origin` on this branch only; no merge.
- **Protocol frozen before the seed existed** (`frozen_protocol.json`, sha `29917466…`): contract, config, candidate files (D3-A and D3-M final checkpoints + provenance + recorded fitting predictions; the D1 baseline model with its stored standardization), prior-phase manifests (E1, D1, D2, and a new D3 supplementary manifest), fitting-only fixture.
  The tool refuses to freeze if a seed already exists; the freeze preceded the draw.
- **One fresh G1 master seed drawn from the OS CSPRNG, fingerprint `2fef6b12a6cc5faa`**, recorded 2026-10-09T02:38:17Z before generation, write-once. It has not been redrawn. It has been exposed only to generation statistics (§2); no labels were used for any modelling and no model output exists.
- New code (all tested; 3 G1 tests + the earlier suites pass): G1 roles (data builder/auditor, frozen-candidate evaluator, metric aggregator) with runtime denial of root-bearing metadata, exclusion index, gen-001 data and optimizer states to the evaluator; exclusion-aware generation/selection; audit (exact quotas, row/metadata, canonical identities, root–child membership, terminal exclusion, empty-cache re-analysis, fresh exact re-query, full-key cross-gen-001 exclusion, independent-reference child checks and bounded M3 minimal-depth checks); frozen label-independent derangement map; paired 5,000-resample cluster-bootstrap aggregation; frozen-candidate evaluator (CUDA for A/M, host f64 for the baseline).
- **Evaluator verification (fitting-only fixture; append-only receipt `evidence/g1/evaluator_verification.json`) — PASS:** A and M reproduce the recorded D3 update-12,000 fitting predictions (max |Δlogit| 1.8e-15, 768/768 classifications agree; tolerance 2e-3), the baseline reproduces its recorded D1 fitting predictions (max |Δ| 8.9e-16; tolerance 1e-9, stored standardization, no recomputation),
  features are independent of labels/ids, and the erasure definitions are consistent. No G1 data was involved. (The receipt name is now taken; a revised run must use a new receipt name, and the verified evaluator source digest will change with any code change.)
- **Preservation (append-only receipts `preservation_{start,end}.json`):** 282 files (E1, D1, D2, D3 manifests + gen-001 manifest) re-hash identically before and after. gen-001's sealed test was never opened by model code.

## 2. Why construction failed — MEASURED

The registered rule: group new roots into connected components (roots sharing any canonical immediate child, all legal moves) and **remove every component containing a root whose canonical root key or any child key occurs in gen-001's roots-and-all-children index** (1,628 roots, 62,229 distinct canonical children).
The generation loop (deterministic rounds of 1,000 attempts per family) never reached the quotas. Per round (accepted roots in the pool → roots kept after exclusion; examples still short of 1,536):

| round | pool | directly overlap gen-001 | largest component (pre-exclusion) | kept (strict rule) | short | kept (per-root variant, statistics only) | short |
|---|---|---|---|---|---|---|---|
| 0 | 797 | 41% | 8 | 435 | 231 | 471 | 209 |
| 3 | 3,100 | 41% | 65 | 1,195 | 132 | 1,833 | **0** |
| 5 | 4,668 | 40% | 160 | 1,373 | **100 (best)** | 2,788 | 0 |
| 8 | 6,997 | 41% | 420 | 1,301 | 160 | 4,126 | 0 |
| 20 | 16,425 | 40% | 1,137 | 415 | 744 | 9,882 | 0 |
| 39 | 31,123 | 40% | 2,191 | 98 | 1,212 | 18,697 | 0 |

(first attempt, run to round 159 / 119,002 roots before I terminated it at 600 s with 1 root kept; the 40-round replay above is deterministic and identical in its early rounds.)
Two mechanisms: (1) **about 40% of all fresh roots share at least one canonical child with gen-001** (its 62k children already cover a large fraction of this domain's small child space), so many roots are individually excluded; (2) connected components **percolate** as the pool grows
(largest component 8 → 2,191 roots), and any gen-001 overlap inside a component deletes it whole, so the strict rule *shrinks* the usable pool after round ≈ 6. At its best (round 5) the only short cell is **KQQvK remaining-budget 2 (M3 roots)**: 78 of 128 for each class; every other cell was full.
The strict rule therefore cannot supply the M3 KQQvK quota at any pool size. This is a property of the frozen protocol and this domain, not an engineering failure of the generator, and no reduced quota, substitution or retry is permitted.
**Process note (disclosure):** I terminated the first generation attempt myself at ≈ 10 minutes (it was clearly degenerate: 1 root kept of 119k) instead of letting it reach its 3,300 s limit; its log is preserved. The diagnose-only replay wrote statistics only — **no rows, labels, panel or model output were produced**, and the per-root counts above were not used to build anything.

## 3. MEASURED / INFERRED / NOT RUN

MEASURED: §1–§2. INFERRED: the transitive component exclusion is stricter than needed to prevent identity-level overlap with gen-001, and its infeasibility is driven by percolation plus the high direct-overlap rate; under a per-root identity exclusion the quotas would be feasible by round 3 (statistics only, not built).
NOT RUN: G1 panel construction, audit of any panel, derangement/erasure maps, any candidate inference, paired comparisons, latency on G1, decision rules, aggregation.

## 4. Concrete revision proposal (owner decision required)

**R1 — identity-level exclusion, then grouping.** Remove each new root whose canonical root key, or any of whose child keys, appears in the gen-001 index (full 65-byte keys); *then* form connected groups among the remaining roots for deduplication and the cluster bootstrap. Guarantee: no G1 root or child canonical identity occurs in gen-001 (checked exhaustively by the audit).
What is weaker: a G1 component may still be adjacent to (not identical with) gen-001 positions — but candidates were trained only on the 768 fitting rows and all G1 positions are identity-disjoint from every gen-001 root and child, so this does not reintroduce identical-position leakage; it should be stated as a limitation.
Seed: the already-drawn seed has been exposed only to generation statistics (no labels used, no model outputs); reusing it with R1 is defensible but is the owner's call — a fresh draw is the conservative alternative. All other quotas, selection, controls, metrics and decision rules stay as frozen. R1 requires a contract amendment, a new append-only protocol freeze, re-verification of the evaluator at the amended source digest and a new G1 run; none of this was done.

## 5. Next-agent prompt

> Continue on branch `experiment/hp-v69-two-clock-workspace` (read AGENTS.md and docs/v69/{CONTRACT,…,RESULTS_D3,G1_CONTRACT,RESULTS_G1}.md). G1 stopped at construction under the frozen strict exclusion rule (see RESULTS_G1 §2). **Do nothing until the owner states in the session whether revision R1 (identity-level exclusion, then grouping) is approved and whether to reuse the drawn seed (fingerprint 2fef6b12a6cc5faa) or draw a new one.**
> If approved: (1) write a G1 amendment (do not edit the frozen contract; add `G1_AMENDMENT_R1.md` and an updated config) specifying R1; (2) freeze it in a new append-only protocol file before any generation; (3) implement R1 as a narrow change to `apply_exclusion` (per-root identity exclusion first, groups afterwards), keeping all quotas/selection/audit/controls/metrics/decision rules unchanged, with tests; (4) re-verify the evaluator at the new source digest (new receipt name); (5) generate under the 60-minute cap with the approved seed policy, audit exhaustively (incl. full-key cross-gen-001 exclusion), freeze, evaluate each candidate exactly once (ordinary + derangement + erasure), aggregate with 5,000 paired cluster-bootstrap resamples, apply the prospective rules, and write preservation receipts before and after.
> If not approved: stop and report. No retraining, tuning, calibration, ensemble, threshold change, gen-001 sealed inference, hierarchy fit, policy, search or self-play.
