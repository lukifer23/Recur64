# Recur64 V69 — initial three-arm semantic learnability experiment (E1): results and decision

Branch `experiment/hp-v69-two-clock-workspace`. Start head `681e3f1`; this report is bound to the
commits listed in §1. Labels: **MEASURED** (observed here), **INFERRED** (reasoned from measurements,
not tested), **NOT RUN**. Scope guard honoured: no extra seeds, longer fits, alternative losses,
architecture rescue, sealed-test inference, policy training, search, self-play, or merge.

## Decision (one line)

**All three arms FAIL the preliminary feasibility gates, including the fitting gate.** No arm learned
anything beyond the class prior: fit and validation balanced accuracy 50.0%, BCE 0.6932 (= ln 2), every
example predicted negative. There is no recurrent advantage to evaluate (nothing separates A, B, C).
Per the pre-declared interpretation rule: *all fail → diagnose task/representation/optimization; no
bigger campaign.* A passing result was never obtained; nothing here supports claims about reasoning,
hierarchy, or chess strength.

## 1. Source lineage, identities, custody

| Item | Value |
|---|---|
| Fits and endpoint evaluations ran at consumer head | `fb72a81020a9df38ab86aaf76ebbcb75616c1bb9` (clean tree, 0 dirty files); executable-source digest `3b31c994…bee0ea` (identical at qualification commit `ad685ba` — only docs/scripts changed between) |
| Post-hoc diagnostic + post-run audit | head `7b163c4518a9cbd953c4725a82df48bfe3033041` (adds one inference-only diagnostic command; digest differs by that code) |
| **Data producer** (gen-001) | `36a81508b456ede1cb682f2f03fe678fd08db70f` (kept distinct from the consumer; never stamped onto data files) |
| gen-001 data | unchanged: fit `14c7167f…`, val `705bea66…`, sealed test `5de7631e…` (SHA-256 prefixes), manifest `3da360e6…` — verified byte-identical after the whole campaign (post-run audit) |
| Seed fingerprint | `5710d34e89d9dd35` (no new seeds drawn); streams used: `model_init/0`, `train_order/<epoch>`, `intervention/<partition>/<family>/<budget>`, `bootstrap/<arm>`, `qualification_init` + `qualification_direction` (disposable) |
| Canonical init | `init/canonical_init.bin` sha256 `3bc80e9f…`, tensors sha256 `f21959ae…`; every arm's loaded tensors verified equal to it before training |
| Frozen spec | `spec/frozen.json` (`evidence/e1/frozen.json`): MODEL_SPEC `09070427…`, CONTRACT `e2a5dbfe…`, intervention map `0d1aab88…` |

Changed/added files since `681e3f1` (all under `crates/recur64-v69*`, `scripts/v69`, `docs/v69`, `Cargo.*`,
`.gitignore`): new crate `recur64-v69-model` (model, init, inventory, trainer, qualification, f64 reference,
data loader, `v69-fit` CLI); in `recur64-v69`: `access.rs` (role policy), `audit2.rs`, `features.rs`,
`metrics.rs`, `provenance.rs`, `v69-aggregate`; tests `audit_corruption.rs` (18), `features_metrics.rs` (7),
`model_boundary.rs` (2); launcher `run_limited.ps1` + self-test, `freeze_spec.ps1`.

## 2. Integrity repair (Section A) — MEASURED

New source-bound receipt `evidence/qualification/audit_receipt_v2.json` (pre-fit, head `ad685ba`) and
`evidence/e1/audit_receipt_v2_postrun.json` (post-campaign, head `7b163c4`): PASS, 0 failures. The original
receipt/manifest/generation report were not touched. Checks: manifest hashes via custody-routed opens; every
model row ↔ metadata by unique id (fen, budget, label, partition, family; no missing/extra/duplicate); root,
child and example canonical identities recomputed from FEN; groups, partition assignments and the whole
selection **rebuilt from pool + master seed** and compared; each example verified as the claimed root's child
with the stored status and a **fresh exact re-query of all 1,536 targets**; all 453 contributing roots
re-analysed from an empty cache; quotas. 18 corruption tests prove detection of: row/meta label, FEN and budget
mismatch, duplicate/missing/extra rows, mis-assigned partition, wrong group association, wrong target even when
row and metadata agree, corrupted canonical identity, example not a child of its root, unknown fields, manifest
hash mismatch (audit stops on unverified inputs). Role policy (learner = seed + `data/fit.jsonl`; evaluator =
fit + val rows; aggregator = fit/val metadata + predictions) is enforced at runtime and tested; a source scan
forbids the withheld paths and audit roles in the model crate. Model code never opened `sealed/`, `pool/`
or root-bearing metadata.

## 3. Model and parameters — MEASURED

Exactly as `MODEL_SPEC.md` (width 192, 6 heads, FFN 576, GELU, zero dropout, pre-norm LayerNorm ε 1e-5,
residual scale 0.1, 2 encoder blocks, shared fast/slow blocks with the encoded-board refresh, head
mean-pool → LN → MLP(192,192,1)). **1,876,417 parameters in 107 tensors** (ceiling 4 M; no width/depth
added): piece_emb 2,496; square_emb 12,288; scalar_proj 1,728; budget_emb 768; att_emb 384; enc0 370,944;
enc1 370,944; enc_ln 384; slot_emb 1,536; ws_pool_ln 384; ws_pool_proj 37,056; fast 519,936; slow 519,936;
head_ln 384; head1 37,056; head2 193. Decay group 41 rank≥2 tensors, no-decay group 66 bias/norm tensors.
Complete inventory: `evidence/qualification/param_inventory.json`.

## 4. CUDA qualification — MEASURED (all failures preserved)

Final run: A, B, C qualified, 15 checks each, exit 0, 8 GiB / 3,072 MiB limits respected, 0 orphans
(`evidence/qualification/`). The suite first failed (stack overflow on the Windows main thread; a one-sided
gradient of `clamp_min` at exactly z = 0 — a real defect, fixed; over-strict per-tensor accumulation check;
diagnostic skipping zero-gradient tensors; fp32 finite differences unusable) — details and the evidence of each
failed attempt are in `MODEL_SPEC.md §9`; no scientific fit existed at that time.
**Precision finding:** Burn 0.21 / cubek-matmul autotune can run f32 matmuls as **TF32 tensor-core** products;
measured max |logit error| vs an independent f64 host implementation of the model: 5.7e-4 (A), 9.3e-4 (B),
9.1e-4 (C). The graph is "CUDA f32 storage, matmul inputs possibly TF32" — **not validated as strict FP32**,
and cannot be forced to be without custom kernels (prohibited). Gradients match exact f64 finite differences to
≤ 1.2e-3 relative; 8×2 accumulation matches the batch-16 gradient to ≤ 1.2e-6; credit assignment through the
full unroll verified with leaf probes, a detach negative control, and shared-gradient change when early calls
are cut; checkpoint restores weights, both AdamW states and schedule (continued-training diff 0.0; fresh-
optimizer control differs 3.9e-4); no state leakage; labels/ids never inputs; bit-identical init across arms.

## 5. Fixed fits — MEASURED

Each arm: 600 updates, microbatch 2 × 8 accumulation, AdamW, LR 5e-4 → 5e-5 (20-update warmup, cosine),
wd 1e-4 (decay group only), global clip 1.0 applied once per accumulated gradient, identical shuffled
sample stream (9,600 samples; every example seen 12–13 times), fixed update-600 endpoint, no validation during
training, no restarts. All completed (status complete, exit 0, 0 orphans).

| | A (1F+1S) | B (4×(2F+1S), reads 1,2,4) | C (6×(F+S), reads 2,4,6) |
|---|---|---|---|
| block calls / readouts | 2 / 1 | 12 / 3 | 12 / 3 |
| fit wall (launcher) | 152 s | 520 s | 501 s |
| mean update (ms) | 249 | 863 | 832 |
| first-20 / last-200 train loss | 0.729 / 0.694 | 0.712 / 0.694 | 0.722 / 0.694 |
| clipped updates | 30.3% | 28.7% | 23.5% |
| peak host / sampled device (MiB) | 367 / 161 | 399 / 193 | 397 / 193 |
| val inference ms per batch of 16 | see `eval_provenance_*.json` (28.4 for B) | | |

B and C have equal block-call counts, and measured update latency agrees within ~4%; no FLOP claim is made.
Device memory is whole-GPU nvidia-smi sampling at ~2 s (launcher) or 100 ms (qualification) and can miss peaks;
allocator high-water values are unavailable.

## 6. Endpoint evaluation — MEASURED, recomputed independently

Independent aggregation (`v69-aggregate`, f64, separate crate, fit/val metadata + serialized predictions only)
agrees with the in-process f32 summary to ≤ 1.2e-9 for all arms.

| (identical for A, B, C) | fit | validation |
|---|---|---|
| balanced accuracy @ logit 0 | 50.00% | 50.00% |
| ordinary accuracy | 50.00% | 50.00% |
| final-readout BCE | 0.6932 | 0.6932 |
| Brier | 0.2500 | 0.2500 |
| confusion (tn, fp, fn, tp) | 384, 0, 384, 0 | 192, 0, 192, 0 |

Intermediate readouts (B, C): BCE 0.6932 and BA 50.0% at every readout. Per-cell (family × budget × class)
accuracies are 1.00 for every negative cell and 0.00 for every positive cell (constant "negative" output);
cell BCE 0.680–0.707 across all arms, partitions and cells. Fixed derangement (map frozen before any result; sha `0d1aab88…`; label-agreement of the
donor map 49.5%): accuracy vs recipient labels 50.0%, vs donor labels 50.0%, deranged − real gap **0.0 pp**
(cluster-bootstrap 95% CI [0.0, 0.0] over 100 validation groups; BCE CI 0.693–0.694); every deranged input's
logits equal the donor ordinary-input logits (max difference 0.0). Board erasure: BA 50.0%, 0% predicted
positive (uninformative — the models already ignore the board). **Gates: fit ≥ 95% ✗, val ≥ 75% ✗, val BCE
≤ 0.55 ✗, derangement drop ≥ 15 pp ✗; engineering/custody gates ✓ → feasibility NOT established for any arm.**

## 7. Why — diagnosis (post-hoc, inference-only; `evidence/e1/diag_signal.json`)

MEASURED: across validation inputs the output spread (logit std) is 0.13 / 0.16 / 0.13 (A/B/C) at the
canonical initialization (label-uncorrelated input dependence) and **1.3e-4 / 4.2e-4 / 2.1e-4 after 600
updates**; the across-example variation of the pooled workspace fell from ≈10% of its mean norm to ≈1%,
and of the pooled encoder output from 12% to 3%. Training loss never left the ln 2 plateau although
gradient norms stayed ≈ 0.5–1 (clipped 14–30% of updates, mostly early). So the optimizer did not fail to
move; it found no label-predictive direction and removed the input-dependent noise.
INFERRED (not tested; testing is outside this authorization): (i) the task as presented — forced mate within
n of an immediate child, from piece placement alone — may require search-like computation that a 1.9 M-parameter
model cannot acquire from 768 examples in 600 updates; (ii) because even the 768 fitting examples were not
memorized, optimization/signal-path weakness is at least as likely as task difficulty: the label signal must
traverse mean pooling, 0.1-scaled residual sublayers, a LayerNorm and a 2-layer head, and cheaper gradient
descent directions (shrinking variance) exist; (iii) TF32 matmuls are a minor suspect (error ≈ 1e-3 on logits of
0.02–0.3 at init) and cannot explain a complete failure to fit. No claim is made about which of these holds.

## 8. Interpretation and recommendation

All arms fail → **do not scale or continue the hierarchy comparison on this recipe.** Simpler recurrence cannot
be judged better than hierarchy when nothing is learned; "keep one-pass for simplicity" is not earned either —
there is no success to simplify. Recommended next step is a small, pre-registered **diagnostic** phase, not a
bigger campaign: (1) one-pass-only overfit-ability test on 16–64 fitting examples to separate optimization
failure from task difficulty; (2) a representation audit of the target itself with trivially computable
features (e.g. attacker piece hanging / defender king mobility) via a non-neural baseline on the *same* gen-001
fit/val split to measure how much of the label is explainable without search; (3) only then decide between
signal-path changes (e.g. per-token readout instead of mean pooling, residual scale) and a changed task.
Any such change must be a new pre-registration with a new fresh fitting/validation draw if gen-001 validation has been
looked at (it has: this report). The sealed test stays unopened.

## 9. MEASURED / INFERRED / NOT RUN

MEASURED: everything in §1–§7 tables and figures. INFERRED: the explanations in §7 (i)–(iii) and the
recommendation. NOT RUN: sealed-test inference; any additional seed, longer fit, alternative loss, retuned
hyper-parameter or architecture variant; strict-FP32 matmul; BF16; allocator high-water measurement; FLOP
counting; a non-neural baseline; any claim about hierarchy, reasoning, or chess strength.

## 10. Next-phase prompt (diagnostic, bounded)

> Continue on branch `experiment/hp-v69-two-clock-workspace` (read AGENTS.md, `docs/v69/{CONTRACT,MODEL_SPEC,
> RESULTS_E1_THREE_ARM}.md`). E1 failed all gates; do not rerun E1. Phase D1 (diagnostic only, ≤ 2 h total, no
> new data draw yet, sealed test closed): (a) implement a non-neural exact baseline on gen-001 fit/val using only
> model-visible features (material/attack/king-mobility counts) with a fixed simple learner, reporting
> balanced accuracy and per-cell metrics; (b) pre-register and run an overfit test of the one-pass arm A on 32
> fixed fitting examples with the same recipe except the number of updates (≤ 2,000) and report fitting accuracy
> only; (c) report whether the label is explainable without search and whether the model can memorize.
> Stop after reporting. Pre-register any follow-up (new draw of fit/val from a fresh seed, signal-path variants)
> before running it; keep qualification, custody, role access, bounded launcher and provenance unchanged.
