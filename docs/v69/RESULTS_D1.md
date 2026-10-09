# Recur64 V69 — D1 bounded diagnostic of learning failure: results and decision

Branch `experiment/hp-v69-two-clock-workspace`. Contract: `D1_CONTRACT.md` (+ `d1_config.json`), frozen before any fit.
Evidence: `evidence/d1/`. Labels: **MEASURED**, **INFERRED**, **NOT RUN**. D1 is a retrospective diagnostic on gen-001;
its validation partition is already-exposed development evidence, so nothing here is fresh confirmation.

## Decision

**Both neural models memorize the 32-example fitting panel (rule: BOTH_MEMORIZE).** D1-A (the unchanged E1 one-pass
model) and D1-M (a direct-board MLP) each reach 32/32 correct and BCE ≈ 0 by update 100 and stay there through update
2,000 (criterion: 32/32 and BCE ≤ 0.05). Basic tiny-panel trainability is therefore established for the current
architecture, recipe and input/target/loss/update mechanics; **larger-set learning remains open.** Separately, a
23-feature shallow logistic baseline reaches **validation balanced accuracy 77.9%, AUROC 0.863, BCE 0.472** — i.e. the
label has substantial shallow predictive structure that the E1 neural models did not find.
This does **not** show that the task is learnable by the E1 recipe at scale, and does not show that search is
unnecessary.

## 1. Lineage, integrity, provenance

- Consumer head for all D1 fits and the baseline: `d39d787f…` (clean tree, 0 dirty files), D1 started from `c99b45b`.
  Data producer remains `36a81508…` (distinct). Pushed to `origin` on this branch only; no merge.
- **E1 files preserved:** after D1, all 59 files of the supplementary E1 manifest (`d1/e1_supplementary_manifest.json`:
  fits, optimizers, metadata, predictions, reports, specs, init, intervention map, qualification, audits) re-hash identically;
  gen-001 data/pool/sealed unchanged. The manifest is a **verification taken at D1 start, not a launch-time receipt.**
- **Finding while verifying E1 (a real provenance deviation, mine):** `v69-d1 verify-e1` first FAILED: the pre-fit audit
  receipt hashed in E1's `spec/frozen.json` (`3b213343…`) had been overwritten in place by the post-campaign audit re-run
  (`5c2633b3…`; both receipts PASS and differ only in source head/time). The pre-fit copy survives in committed evidence
  (`evidence/qualification/audit_receipt_v2.json`) and was copied to `d1/e1_preserved/`. Verification now enforces the frozen
  hash against the preserved copy and requires the live file to equal the preserved post-run copy; every other E1 frozen
  dependency matched exactly. No E1 file was modified to repair this.
- **Enforced hashes:** `d1/frozen_d1.json` (sha `6fdf6d2c…`) holds 37 expected hashes in three role groups; every D1
  launch re-hashed its group through the role-restricted access layer (learner: 23 hashes incl. D1 contract/config, panel
  rows, example stream, canonical E1 init, MLP init, E1 frozen file, E1 manifest, all 12 E1 final checkpoint/metadata
  files [hash-only, never loaded], dataset manifest and fit rows; baseline: 6; aggregator: 8) and aborts on mismatch.
  Each fit's `provenance.json` binds source identity, contract/config/frozen hashes, panel rows (`bf958392…`), example stream
  (`e7b1415d…`), init (D1-A: E1 canonical `3bc80e9f…`; D1-M: `0228b6ea…`), completed updates (2,000) and the
  SHA-256 of the actual checkpoint, optimizer, metadata and prediction files; the aggregator re-checked them.
- Learner role never read validation, sealed, pool or metadata; the model crate's source-scan test still passes.

## 2. Panel — MEASURED

32 gen-001 **fitting** examples, 16 positive / 16 negative, 32 distinct connected group_ids, selected by the keyed
D1 stream (`d1/panel`) with no model output: 24 (two per family × budget × class cell) + one positive and one negative in each of four
keyed strata (KQQvK n2, KQQvK n1, KRRvK n2, KQRvK n2). Capacity was sufficient; no change to the selection was needed.
Example stream: 32,000 samples, every panel example exactly 1,000 times, identical for both models.

## 3. Qualification — MEASURED (passed first attempt; 21 checks, 23 s)

Run under the bounded launcher on the RTX 2050 with the documented process-scoped CUDA environment; limits respected
(host peak 1,437 MiB, sampled device 1,187 MiB, no orphans). D1-M (842 → 128 → 64 → 1, **116,225 parameters**): forward vs
independent f64 reference max error 1.5e-4; gradients match exact f64 finite differences (≤ 3.4e-4 relative); stable BCE
(logits ±88); 8×2 accumulation vs batch-16 (error/scale 1.2e-7); one clip operation per update, clip to norm 1.0 verified;
parameter groups (3 decay / 3 no-decay tensors; zero-gradient lr = 1 test); checkpoint + both optimizers restore (continued
diff 0.0; fresh-optimizer control 3.2e-4); state independence (0.0); labels/ids never inputs. D1-A: forward at the E1 canonical
init is bit-identical to the E1 one-pass code path and within 8.1e-4 of the f64 reference; same machinery checks through the D1
trainer. Qualification weights were disposable (`d1_qualification_init`). **Preserved failures:** none in qualification.
During development only: the model-crate source-scan test flagged a comment containing a forbidden token (reworded), and the
E1 deviation above. Precision adopted as measured: f32 storage/accumulation, matmul inputs possibly TF32; **not strict FP32.**

## 4. Fixed neural fits — MEASURED

Identical mechanics (2,000 updates, microbatch 2 × 8, AdamW, LR 5e-4 → 5e-5 with the contract's indexing, global clip once),
serial, no early stopping/selection/restart; both COMPLETE (exit 0, 0 orphans).

| | D1-A (E1 one-pass, 1,876,417 p) | D1-M (MLP, 116,225 p) |
|---|---|---|
| panel @0 (init): correct / BCE / AUROC | 16/32 / 0.6997 / 0.512 | 19/32 / 0.6889 / 0.621 |
| @100 | 32/32 / 1.0e-4 / 1.000 | 32/32 / 0.0125 / 1.000 |
| @500 | 32/32 / 0.0000 / 1.000 | 32/32 / 0.0002 / 1.000 |
| @1000 | 32/32 / 0.0000 / 1.000 | 32/32 / 0.0001 / 1.000 |
| @2000 | **32/32 / 0.0000 / 1.000** | **32/32 / 0.0000 / 1.000** |
| logit spread @0 → @2000 (sd) | 0.13 → 13.0 | 0.07 → 10.4 |
| class means @2000 (pos / neg) | +13.1 / −12.9 | +10.4 / −10.3 |
| memorization criterion | **met** | **met** |
| grad norm (pre-clip), updates 0–20 / 20–100 / 100+ | 2.9 / 0.98 / ≈0 | 0.48 / 0.53 / ≤0.016 |
| clipped updates | 2.0% (all early) | 0.4% |
| parameter movement from init (relative L2) @100 / @2000 | 6.7% / 6.9% total | 22.8% / 32.6% total |
| largest component movement (D1-A @2000) | square_emb 0.45, piece_emb 0.34, att/budget emb ≈0.12, encoder 0.08–0.10, fast/slow 0.05–0.06 | fc1 0.30, fc2 0.39, fc3 0.31 |
| fit wall / mean update | 463 s / 231 ms | 31 s / 15.5 ms |
| peak host / sampled device (MiB) | 444 / 225 | 281 / 95 |
| exposure | 1,000 per example | 1,000 per example |

(Panel snapshots are fitting diagnostics only; neither neural model saw validation or sealed data. Class-conditional
distributions, per-cell results and full quantiles are in `evidence/d1/d1_report.json`.) Total native execution ≈ 11 min
(≪ 2 h). No resource limit was approached.

## 5. Shallow feature baseline — MEASURED, retrospective exploratory

Fit once on all 768 fitting rows (23 pre-specified features; objective 0.6931 → 0.5062, 1,000 steps, lr 0.05, L2 0.01),
evaluated once on the 384 validation rows; threshold logit 0; no search on validation.

| | BCE | balanced acc | AUROC | 95% group-bootstrap CI (BA / AUROC / BCE) |
|---|---|---|---|---|
| fit (204 groups) | 0.490 | 76.4% | 0.845 | [73.1, 79.7]% / [0.817, 0.872] / [0.459, 0.521] |
| validation (100 groups) | 0.472 | 77.9% | 0.863 | [73.2, 82.4]% / [0.824, 0.902] / [0.424, 0.523] |

Largest standardized weights: fewer legal defender moves → mate (−1.49), legal defender captures of an attacker major →
not mate (−0.76), minimum major–king distance (−0.28). Per-cell validation accuracy 0.66–0.88 in all 12 cells.
These numbers would clear the E1 validation-only gates (BA ≥ 75%, BCE ≤ 0.55) but the baseline was never a candidate
for them, validation is exposed, and a threshold on fit/val closeness (76% vs 78%) shows no overfit — this is evidence of
**shallow predictive structure**, not of generalization claims, and weak/strong baseline results do not establish whether search is necessary.

## 6. Independent recomputation — MEASURED

`v69-d1 aggregate` recomputes every metric in f64 from serialized predictions and metadata (complete ids, labels, counts,
artifact hashes, no incomplete runs). Agreement with the training process's own f32 panel summaries: max |difference| over
BCE and correct counts 1.4e-20 across both models and all five snapshots.

## 7. MEASURED / INFERRED / NOT RUN

MEASURED: §1–§6. INFERRED: (a) E1's failure on 768 examples was **not** a broken signal pathway, loss, update rule,
input encoding or label wiring — the same architecture, recipe and code memorized 32 examples within 100 updates while
moving only ~7% from initialization; (b) E1's non-learning is therefore more plausibly about learning the label's structure
from many examples within 600 updates (scale/capacity-for-generalization/optimization dynamics) than about the pipeline;
(c) the label contains shallow structure (mobility and capture features) that the E1 networks did not discover, so E1 is not
evidence that the task is hard; (d) memorizing 32 examples is cheap and says nothing about generalization (the model could
memorize via embeddings alone). NOT RUN: any larger-set fit, scaling ladder, longer E1-style run, new validation draw, fresh
confirmation, strict-FP32 matmul, sealed-test inference, hierarchy comparison, search, self-play, policy training.

## 8. Recommendation and next-agent prompt

Do not expand to a hierarchy comparison yet. The decisive open question is *where* and *why* fitting breaks between 32 and
768 examples. Next (pre-registered, fitting-only, still no validation use): a **fit-set scaling ladder** — the same two models,
same recipe, N = 32, 64, 128, 256, 512, 768 fixed nested fitting subsets, with the update horizon declared in advance
(e.g. 4,000 updates), measuring when memorization fails and whether the failure is update-budget (loss still falling) or
optimization collapse (input dependence lost, as in E1); then, only if a model fits 768, a **single fresh validation draw**
requires explicit owner approval (new data generation is forbidden to agents without it). A shallow-feature-informed
auxiliary input (legal-move count / capture indicators) is a *later* representation experiment, not part of the ladder.

> Continue on branch `experiment/hp-v69-two-clock-workspace` (read AGENTS.md and docs/v69/{CONTRACT,MODEL_SPEC,
> RESULTS_E1_THREE_ARM,D1_CONTRACT,RESULTS_D1}.md). Do not rerun E1 or D1. Phase D2 (fitting-only scaling ladder): write a
> separate D2 contract and immutable namespace; nested fixed subsets of gen-001 fitting rows (N = 32, 64, 128, 256, 512,
> 768) chosen by a keyed deterministic stream with group-distinct preference; models D1-A and D1-M with the D1 qualification
> machinery re-verified; horizon declared in advance (≤ 4,000 updates), schedule indexing defined, same optimizer
> recipe; measure fit accuracy/BCE/AUROC, logit spread, input-dependence retention, gradient norms and movement at
> pre-declared updates; hash-enforced inputs, bound provenance, bounded launcher, 3,072 MiB / 8 GiB / total wall caps;
> no validation or sealed use by neural code, no new data, no seed redraw, no extra seeds. Report where memorization
> breaks and whether the break is budget- or collapse-like; stop and recommend; do not execute follow-ups.
