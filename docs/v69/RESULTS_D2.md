# Recur64 V69 — D2 fitting-set scaling diagnostic: results and decision

Contract: `D2_CONTRACT.md` (+ `d2_config.json`), frozen before any fit. Evidence: `evidence/d2/`. Labels: **MEASURED**, **INFERRED**,
**NOT RUN**. D2 is fitting-only: no neural code touched validation or sealed data; no new data, seed, feature, architecture or backend change.

## Decision

**NEITHER model strong-fits the full 768-example fitting partition under D2** (strong fit = balanced accuracy ≥ 95% and BCE ≤ 0.05).
Both fit 64 and 256 examples exactly; both degrade between 256 and 768. They degrade differently:
- **D2-A (unchanged one-pass model): collapse-like at 768.** At 768 it stays on the class prior for all 2,400 updates (BA 0.500, BCE 0.693, logit spread ≈ 0,
  input-dependent variation of its pooled representation stays ≈ 0.02/0.008). At 64 and 256 it first collapses the same way, then *escapes* and memorizes.
- **D2-M (direct-board MLP): partial learning at 768** — BA 0.764, AUROC 0.850, BCE 0.481 at update 2,400, slowly improving, not memorizing. Its fitting numbers are
  essentially those of the shallow feature baseline's fitting numbers (0.764 / 0.490), suggesting it learned shallow structure rather than memorizing.
This is not evidence that more training would succeed, and not evidence about generalization.

## 1. Identity, integrity, provenance — MEASURED

- Consumer head for all D2 qualification, fits and aggregation: `5d7734c6…` (clean tree, 0 dirty files). Data producer `36a81508…` (distinct).
- Frozen: `d2/frozen_d2.json` (evidence copy) holds expected hashes in two role groups; every fit re-verified the 37-hash learner group (logs: "37 frozen hashes verified"; the aggregator verified its own 18)
  through the role-restricted layer and would abort on mismatch; hash-enforcement failure is unit-tested.
  Each of the six `provenance.json` files binds source identity, contract/config/frozen hashes, subset-rows/example-stream hashes, init hashes (A: E1 canonical `3bc80e9f…`; M:
  D1 MLP `0228b6ea…`, each loaded-tensor hash verified), completed updates (2,400 in all six), actual checkpoint/optimizer/metadata file hashes and the prediction-file hash;
  the aggregator re-checked them and the recorded exposure counts against the frozen streams.
- **Preservation (append-only receipts, `evidence/d2/preservation_{start,end}.json`):** 127 files (E1 manifest, D1 manifest, gen-001 manifest) re-hash identically before and after the campaign;
  no E1/D1/gen-001 file was modified. E1/D1 fitted checkpoint files were hashed only, never loaded. The documented E1 audit-receipt deviation is preserved untouched (D2 never writes to `audit/`).
- Neural role never read fit metadata, validation, sealed or pool; source-scan and runtime access tests pass (28 data-crate and 2 model-crate tests green).

## 2. Subsets — MEASURED

64 ⊂ 256 ⊂ 768 with the 32 D1 panel examples in the 64 subset; exact quotas q = 5 (+ 2 keyed pairs: KRRvK n1, KQQvK n1), 21 (+ 2 pairs: KQQvK n1, n2), 64; every stratum balanced.

| N | distinct groups | repeated group memberships | distinct roots | pos / neg |
|---|---|---|---|---|
| 64 | 63 | 1 | 63 | 32 / 32 |
| 256 | 200 | 56 | 203 | 128 / 128 |
| 768 | 204 | 564 | 228 | 384 / 384 |

The fitting partition contains only 204 connected groups, so group-distinctness is impossible beyond that (as the contract allowed); group preference kept 63/64 and 200/256 distinct.
Streams: 38,400 samples per size, per-epoch permutations, identical for A and M; exposures: 600 (N64), 150 (N256), 50 (N768) per example.

## 3. Qualification — MEASURED (passed first attempt, 22 s, 20 checks + 2 D2 checks)

The qualified D1 machinery was reused unchanged; the D2-specific additions passed (LR boundaries: u=0 2.5e-5, u=19/20 5e-4, u=1,999 and 2,000–2,399 exactly 5e-5, monotone; example-stream shape, epoch-permutation and within-update-distinctness
checks for all three sizes). Disposable weights only. **Preserved failures:** none in qualification; Host peak 1,490 MiB, sampled device 1,187 MiB.
Precision as measured: f32 storage/accumulation, matmul inputs possibly TF32; not strict FP32.

## 4. Six fixed endpoints (update 2,400) — MEASURED

| | N | acc/BA | BCE | AUROC | strong fit | exact | fit wall | update ms | clipped | peak host/dev (MiB) |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 64 | 1.000 | 0.0000 | 1.000 | yes | yes | 640 s | 264 | 4.5% | 444 / 488 |
| A | 256 | 1.000 | 0.0001 | 1.000 | yes | yes | 719 s | 296 | 30% | 448 / 288 |
| A | 768 | 0.500 | 0.6931 | 0.514 | **no** | no | 756 s | 309 | 12.7% | 451 / 225 |
| M | 64 | 1.000 | 0.0001 | 1.000 | yes | yes | 56 s | 22 | 1.0% | 282 / 167 |
| M | 256 | 1.000 | 0.0010 | 1.000 | yes | yes | 52 s | 21 | 13.7% | 282 / 95 |
| M | 768 | 0.764 | 0.4812 | 0.850 | **no (partial)** | no | 94 s | 38 | 75% | 282 / 95 |

Total native fit time 2,317 s (≈ 39 min, global limit 2 h); every fit within its cap; exit 0; 0 orphans. Full snapshot tables (updates 0…2,400), confusion matrices, per-cell results, class-conditional margins:
`evidence/d2/d2_report.json`. Device memory is launcher nvidia-smi sampling (~2 s) and can miss peaks.

**First update with strong fit / exact memorization** (snapshot resolution): A N64 200/200, A N256 1000/1600; M N64 200/200, M N256 600/600; none at N768.
**Escape from the prior plateau (A):** N64 between updates 100 and 200 (BA 0.50 → 1.00), N256 between 200 and 400 (0.50 → 0.77), N768 not within 2,400 updates.

## 5. Equal-update and equal-exposure comparisons — MEASURED (with the schedule confound)

Equal update 600 (balanced accuracy / BCE): A: N64 1.000/0.0000, N256 0.855/0.320, N768 0.500/0.693. M: 1.000/0.0004, 1.000/0.017, 0.673/0.584.

Equal average exposure (the learning rates at the checkpoints differ — warmup/decay position — so this is **not** a pure causal attribution):

| avg exposure/example | model | N64 | N256 | N768 |
|---|---|---|---|---|
| 12.5 (updates 50 / 200 / 600) | A | 0.500 / 0.694 | 0.500 / 0.698 | 0.500 / 0.693 |
| | M | 0.969 / 0.522 | 0.941 / 0.175 | 0.673 / 0.584 |
| 50 (updates 200 / 800 / 2,400) | A | 1.000 / 0.004 | 0.965 / 0.104 | 0.500 / 0.693 |
| | M | 1.000 / 0.004 | 1.000 / 0.005 | 0.764 / 0.481 |

At fixed exposure both models fit worse as N grows, so exposure alone does not account for the degradation. Reading it differently, at fixed N more updates help (A N256: 0.50 → 0.97 → 1.00; M N768: 0.67 → 0.76), so budget matters too.

## 6. Training dynamics and collapse-like behaviour — MEASURED

- **A, pooled-representation variation** (‖std over examples‖/‖mean‖ of encoder / workspace): initialization 0.121 / 0.101. By update 50–100 it has contracted to ≈ 0.04 / 0.02 in **all three sizes** (the same
  collapse seen in E1; logit spread falls from 0.13 to ≈ 0.003). It then re-expands at N64 (update 200: 1.14/1.29; later 1.83/2.26), at N256 (update 400: 0.82/0.61; peak 1.79/1.69 at 800, settling at 1.31/1.57), and does **not** re-expand at N768 (0.022 / 0.008 from update 600 to 2,400).
- **A gradients/movement:** at N768 mean pre-clip gradient norm ≈ 0.45–0.50 with 5–10% clipping after update 600 and total movement from init only 7.1% (vs 9.9% at N64, 15.5% at N256); loss stays 0.693–0.694.
- **M:** no representation collapse; loss falls monotonically at all sizes. At N768 the gradient norm stays ≈ 2.3 with ≈ 85–90% of updates clipped after update 1,200 and the loss is still falling slowly (0.549 → 0.507 → 0.491 → 0.489 across windows 600–1,200, 1,200–2,000, 2,000–2,200, 2,200–2,400 at the held LR 5e-5); movement 41.6% of init.
- Latency (synchronized): update ms in table above; full-subset inference median A 35/161/424 ms, M 3.9/13/61 ms for N = 64/256/768.

## 7. Independent verification — MEASURED

`v69-d2 aggregate` recomputed every metric in f64 from serialized predictions and subset metadata, requiring complete ids, 11 snapshots per run, exposures equal to the frozen stream, matching hashes, and agreement with the in-process summaries
(tolerance 1e-4; the aggregation would have aborted otherwise).

## 8. MEASURED / INFERRED / NOT RUN

MEASURED: everything above. INFERRED: (a) for A, the time to escape the early prior plateau grows with fitting-set size and exceeds 2,400 updates by N = 768; (b) because both A and M degrade at fixed exposure and
M reaches the shallow baseline's fitting level at 768 without memorizing, the limiting factor looks like how fast gradient descent finds label-relevant structure in a larger set, rather than the pipeline (consistent with D1);
(c) M's still-falling loss at 768 means extra budget is a *hypothesis*, not evidence that more training will fit; (d) A's non-escape is "collapse-like behaviour" — representation variation contracts toward the prior and does not recover — without a cause identified.
NOT RUN: any run longer than 2,400 updates, other learning rates or schedules, other sizes (adaptive sizes were forbidden), validation/sealed evaluation, generalization claims, feature augmentation, recurrence/hierarchy comparisons, larger models or data.
The shallow baseline's exposed-validation result remains retrospective and was neither refit nor used.

## 9. Recommendation and next-agent prompt

Do not enlarge the model or the dataset. The sharpest open question is whether the failure at 768 is a *budget* effect (plateau escape time) or a *dynamics* effect that more updates cannot fix. Next (D3, fitting-only, pre-registered):
the same two models on the **768 subset only**, with one declared horizon extension (e.g. 12,000 updates: D2 schedule shape stretched with an explicit, frozen indexing), recording A's escape time (if any), the variation/collapse trace, and M's fit trajectory.
If A escapes and M memorizes, the failure was budget; if A still does not escape, dynamics are the issue and the next step is a *separate* pre-registered optimization-path intervention (one change at a time).
A fresh validation draw for any generalization claim needs explicit owner approval.

> Continue on branch `experiment/hp-v69-two-clock-workspace` (read AGENTS.md and docs/v69/{CONTRACT,MODEL_SPEC,RESULTS_E1_THREE_ARM,D1_CONTRACT,RESULTS_D1,D2_CONTRACT,RESULTS_D2}.md). Do not rerun or continue E1/D1/D2.
> Phase D3 (fitting-only horizon test): write a separate D3 contract/config/namespace and freeze before fitting. Models: unchanged D2-A and D2-M from their UNTRAINED initializations (hashes enforced; no fitted weights).
> Data: the 768 fitting rows only (reuse the D2 s768 rows and an extended deterministic shuffled epoch stream; hash it). Schedule: declare in advance a stretched horizon of 12,000 updates (state the exact warmup/cosine/hold indexing and note it differs from E1/D1/D2),
> same optimizer, microbatch 2 × 8 accumulation, clip 1.0, wd rules. Snapshots at pre-declared updates (include the D2 ones through 2,400, then every 1,000 to 12,000); report accuracy/BA/BCE/AUROC, logit spread, gradient/clip stats, parameter movement and, for A, pooled encoder/workspace variation.
> Limits: 3,072 MiB sampled device, 8 GiB host, ≤ 2 h total native (A ≈ 62 min at ~310 ms/update; M ≈ 8 min), bounded launcher, append-only receipts, hash enforcement, bound provenance, preservation receipts for E1/D1/D2/gen-001 before and after.
> No validation/sealed use, no new data/seed/feature/architecture changes, no extra horizons or learning rates. Report endpoints, escape time, interpretation per pre-declared rules; stop and recommend; do not execute follow-ups.
