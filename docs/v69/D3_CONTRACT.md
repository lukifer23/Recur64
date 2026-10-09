# Recur64 V69 — D3 contract: bounded additional-update diagnostic (full 768 fitting partition)

Frozen before any D3 fit. Machine-readable parameters: `d3_config.json` (hash-enforced at launch).
Question: *does additional training under the unchanged D2 schedule and example stream produce useful full-set fitting, or escape from A's prior-like
plateau?* (D2: A stayed on the class prior at N = 768 for 2,400 updates; M reached BA 0.764 with loss still falling.) D3 is a new registered experiment,
not a continuation of E1/D1/D2 artifacts. Fitting-only; no validation or sealed inference; no new data, master seed or initialization seed; no feature,
architecture, optimizer-setting, precision, backend, hierarchy, search or self-play change.

## 0. Boundaries and precision

E1, D1, D2 and gen-001 artifacts are preserved byte-for-byte; no fitted weights or moments are loaded (their checkpoint files are hashed only). Only D2's exact
`s768` rows are used (the complete accepted fitting partition); no example is selected using losses or predictions. Precision is the measured CUDA description:
**f32 storage/accumulation, matmul inputs possibly TF32; not strict FP32**; no precision/backend change; no CPU fallback, custom autodiff/kernels, Python training,
driver change or paid compute.

## 1. Integrity, provenance, append-only receipts

`d3/frozen_d3.json` holds expected SHA-256 of every input per role group (learner, aggregator), re-verified through the role-restricted access layer at every launch
(abort on mismatch): D3 contract/config, the complete extended sample order, D2 `s768` rows and order, E1 canonical init, D1 MLP init, E1/D1/D2 frozen files and manifests, the
start receipt, dataset manifest and fit rows, and (hash-only) every E1/D1/D2 final checkpoint/optimizer/metadata file; aggregator additionally the D2 s768 predictions, provenance
and trace, the D1 baseline's **fitting** predictions and the fit/s256/s768 metadata. Each fit's `provenance.json` binds executable source identity (git head, dirty count, source
digest), contract/config/frozen hashes, rows/order/init identities, completed updates, the actual final model/optimizer/metadata file hashes and the prediction-file hash; the aggregator
re-checks them. All receipts under `d3/receipts/` are append-only (writes refuse to overwrite). Preservation of E1, D1, D2 and gen-001 is verified (manifest re-hash) before and after.
The documented E1 audit-receipt deviation (overwritten in place by the E1 post-campaign audit; pre-run copy preserved) remains untouched; D3 never writes to `audit/`.

## 2. Models and initialization

D3-A: unchanged one-pass workspace model (E1 arm A; 1,876,417 parameters). D3-M: unchanged direct-board MLP (842 → 128 → 64 → 1; 116,225). A starts from the E1 canonical
UNTRAINED initialization; M from the D1 UNTRAINED MLP initialization; tensor hashes verified; fresh independent optimizer states; independent model storage.

## 3. Example stream (D2 prefix preserved)

The existing deterministic stream `d2_train_order/768/<epoch>` (per-epoch Fisher–Yates permutation of the 768 rows sorted by id) is extended to 12,000 × 16 = 192,000
samples (250 epochs). **No new shuffle stream.** The first 38,400 sample indices must equal the frozen D2 stream exactly (checked at construction, qualification and aggregation).
Every example receives exactly 250 exposures (checked independently by the aggregator). Update u uses samples [16u, 16u+16); microbatch m uses [2m, 2m+2).

## 4. Fixed recipe (12,000 updates)

Microbatch 2 × 8 accumulation (effective 16); stable BCE-with-logits (microbatch loss /8); global accumulated-gradient clip once at 1.0 (factor min(1, 1/(norm+1e-6)) applied only when
norm+1e-6 > 1); AdamW β 0.9/0.999, ε 1e-8; two instances for groups (decay wd 1e-4 on rank ≥ 2 tensors; wd 0 on biases/normalization). **Learning rate (0-based update u):**
u < 20: 5e-4·(u+1)/20; 20 ≤ u ≤ 1,999: 5e-5 + ½(5e-4−5e-5)(1+cos(π(u−20)/1979)); u ≥ 2,000: 5e-5 (the cosine is **not** stretched; the first 2,400 learning rates equal D2's exactly —
verified independently). No early stopping, best-checkpoint selection, restart, extra updates or alternative horizon. Serial runs.

## 5. Measurements (fitting partition only; inference mode)

Snapshots at updates 0, 50, 100, 200, 400, 600, 800, 1000, 1600, 2000, 2400 (the D2 checkpoints) then 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000. Reported:
accuracy, balanced accuracy, BCE, AUROC; confusion and per-cell metrics; logit spread and class-conditional margins; gradient norms, clipping, LR; parameter movement by component; exposures, latency and
memory; for A the pooled encoder/workspace variation (‖std over examples‖/‖mean over examples‖). **D2-prefix comparison:** same initialization, order and LR do not guarantee identical CUDA trajectories
(autotuned kernels may differ); differences are reported (metrics and per-example logit differences at matching snapshots), not forced to agree; any improvement before update 2,400 cannot be attributed
to the additional updates. Descriptive criteria: **strong fit** BA ≥ 95% and BCE ≤ 0.05; **exact memorization** accuracy 100% and BCE ≤ 0.05; **A's plateau escape** = first snapshot with BA ≥ 60% and
BCE ≤ 0.65 that is confirmed (also met) at the next scheduled snapshot; temporary improvements are reported separately. These are fitting descriptions, not generalization claims.

## 6. Group and error analysis

At the final endpoint: errors and margins by connected `group_id` and by family/budget/class. Document that D2's expansion 256 → 768 mostly added examples within existing groups (computed from metadata);
degradation is not attributed solely to sample count. **Complementarity with the D1 shallow baseline's FITTING predictions** (threshold logit 0): agreement, both correct, both wrong, neural-only correct,
baseline-only correct. The baseline is not retrained, no ensemble is formed, no threshold chosen, no validation used; complementary errors are descriptive evidence only.

## 7. Qualification and limits

Reuse the qualified D1/D2 mechanics unchanged; additionally verify the extended order (exact D2 prefix, epoch permutations, 250 exposures), the LR rule for all 12,000 updates and boundaries, plus the
standard accumulation/clipping/checkpoint/state-independence/role/hash machinery (disposable weights never initialize a fit). Process-scoped CUDA environment as before. Limits: sampled device ≤ 3,072 MiB;
host ≤ 8 GiB; ≤ 2 h total native execution (excluding compilation) with a global remaining-budget check; qualification ≤ 30 min; A fit ≤ 80 min; M fit ≤ 10 min; bounded launcher with native exit codes,
process-tree cleanup and orphan checks; memory sampling limitations reported. Partial snapshot predictions are rewritten after each snapshot so a timed-out run leaves evidence; timeouts are INCOMPLETE,
not scientific failures; no retries or limit changes.

## 8. Independent recomputation and interpretation

`v69-d3 aggregate` recomputes predictions-derived metrics, exposures and the LR sequence independently (f64, from serialized predictions/traces and metadata), requiring complete ids, snapshots and artifact hashes
and agreement with the in-process summaries. Interpretation: fitting improves after a comparable D2 prefix → additional updates helped under this schedule (budget is not claimed as the only cause of earlier failure);
A still prior-like → the current path failed within 12,000 updates (not claimed permanently untrainable); M strong-fits → propose a separately frozen generalization experiment; neither strong-fits → propose one
controlled representation or optimization intervention supported by the measured failure (not another horizon-only escalation). No hierarchy, generalization or fresh-confirmation claims. Stop after D3; no follow-up is executed.
