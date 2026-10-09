# Recur64 V69 — D2 contract: bounded fitting-set scaling diagnostic

Frozen before any D2 fit. Machine-readable parameters: `d2_config.json` (hash-enforced at launch).
Context: E1 failed to fit 768 examples; D1 showed both models memorize a 32-example panel and that shallow features carry
signal. D2 asks only: *do the unchanged one-pass model (D2-A) and the unchanged D1 MLP (D2-M) fit larger fitting subsets
(64, 256, 768), and how do sample exposure, dataset size and collapse-like behaviour separate?* Fitting-only; no
validation/sealed inference by any neural code; no new data, seed redraw, feature augmentation, architecture change,
hierarchy comparison, search, self-play or policy training. D2 is a **new diagnostic recipe**: comparisons with E1 do not isolate
"extra updates" (E1 used a different decay horizon).

## 0. Boundaries and precision

E1, D1 and gen-001 artifacts are preserved byte-for-byte; no fitted E1/D1 weights or moments are loaded (their checkpoint files
are hashed only). The shallow baseline's exposed-validation result is retrospective and is not refit or used. Precision is the
measured CUDA description: **f32 storage/accumulation, matmul inputs possibly TF32; not strict FP32**; backend/precision never
change. No CPU substitution, custom autodiff/kernels, Python training, driver changes or paid compute.

## 1. Integrity, provenance, append-only receipts

`d2/frozen_d2.json` holds expected SHA-256 of every input per role group (learner, aggregator) and is re-verified through the
role-restricted access layer at every launch (abort on mismatch): D2 contract/config, subset rows and example streams, E1
canonical init, D1 MLP init, E1 and D1 frozen files, E1/D1 supplementary manifests, the start receipt, dataset manifest/fit
rows, and (hash-only) all E1 and D1 final checkpoint/optimizer/metadata files. Each fit's `provenance.json` binds executable
source identity (git head, dirty count, source digest), contract/config/frozen hashes, subset/order/init identities, completed
updates, actual checkpoint-file hashes and the prediction-file hash; the aggregator re-checks them. Receipts under `d2/receipts/`
are **append-only** (writes refuse to overwrite). `v69-d2 preserve --label start|end` re-hashes the E1 and D1 manifests and the
gen-001 manifest. **Documented E1 deviation (preserved, not repaired):** the E1 pre-fit audit receipt was overwritten in place by the E1 post-campaign
audit re-run; the pre-run copy exists in committed evidence and `d1/e1_preserved/`. D2 never writes to `audit/`.

## 2. Nested fitting subsets (sizes 64, 256, 768; fitting partition only)

64 ⊂ 256 ⊂ 768; the 32-example D1 panel ⊂ 64; 768 = the complete accepted fitting partition. Construction (host side, depends
only on the keyed ordering and allowed fitting metadata — family, budget, class, group id — never on model output): for size N,
q = ⌊N/12⌋ examples per family × budget × class cell (12 cells); the remaining N − 12q examples are distributed as
positive/negative **pairs** to strata (family × budget) in a fixed keyed order (SHA-256-keyed hash of the stratum under label `d2/strata/<N>`;
64 → q=5 plus 2 pairs, 256 → q=21 plus 2 pairs, 768 → q=64). Equal positives and negatives in every stratum is checked.
Starting from the previous subset, each cell is filled to its quota choosing candidates by (previously unused connected group_id first,
then keyed hash of the id under label `d2/subset`). Group-distinctness is **preferred**, not required. The build stops with an error on
infeasible nesting or capacity. Reported per size: group count, repeated group memberships, distinct roots, class coverage per cell.
Serialized and hashed: model-facing rows (`{id,fen,budget,label}` sorted by id), reporting metadata (fit metadata rows), example streams.
Metadata is never a model input.

## 3. Models and initialization

D2-A: the unchanged E1 one-pass architecture (arm A; 1,876,417 parameters). D2-M: the unchanged D1 MLP (842 → 128 → 64 → 1; 116,225).
Every fit starts from its UNTRAINED initialization (A: E1 canonical init; M: D1 MLP init), each loaded tensor hash verified against the
expected hash, with fresh optimizer states and independent storage. No new initialization seeds.

## 4. Fits (six, serial)

Each fit: 2,400 updates, microbatch 2 × 8 accumulation (effective 16), stable BCE, microbatch loss /8, global accumulated-gradient
clip at 1.0 applied once (factor min(1, 1/(norm+1e-6)), only when norm+1e-6 > 1), AdamW (β 0.9/0.999, ε 1e-8), two instances for the
groups (decay wd 1e-4 on rank ≥ 2 tensors; wd 0 on biases/normalization). **Schedule (update index u, 0-based):** exactly the D1 rule
for u = 0…1,999 (u < 20: 5e-4·(u+1)/20; u ≥ 20: 5e-5 + ½(5e-4−5e-5)(1+cos(π(u−20)/1979)), reaching 5e-5 at u = 1,999); LR held at 5e-5 for
u = 2,000…2,399. Example stream per size: epoch e = Fisher–Yates permutation of the subset rows from stream `d2_train_order/<N>/e`, epochs concatenated
to 2,400 × 16 samples; identical for A and M at that size; update u uses samples [16u,16u+16), microbatch m uses [2m,2m+2).
No early stopping, best checkpoint, restart, extra updates or adaptive sizes. All six fits run even if an earlier size fails
scientifically; engineering failures stop dependent execution; timeouts are INCOMPLETE.

## 5. Measurements (fitting subset only; inference mode)

Snapshots at updates 0, 50, 100, 200, 400, 600, 800, 1000, 1600, 2000, 2400: accuracy, balanced accuracy, BCE, AUROC, confusion,
per-cell results, logit spread and class-conditional margins; per update gradient norm (pre-clip), clipping, LR; parameter movement by
component; actual exposure distribution (recorded and checked against the frozen stream); update and inference latency; host and sampled
device memory. For A only: pooled encoder/workspace variation with the D1 diagnostic definition ‖std over examples‖₂/‖mean over examples‖₂ of
the mean-pooled encoder output and mean-pooled final workspace on the subset rows, at every snapshot. Comparisons: equal-update (including 600)
and equal-average-exposure (12.5 exposures/example: N64 → update 50, N256 → 200, N768 → 600; 50 exposures: N64 → 200, N256 → 800,
N768 → 2,400). The learning rates at equal-exposure checkpoints differ: reported as a confound; no pure causal attribution. Predeclared endpoint
descriptions: **strong fit** = balanced accuracy ≥ 95% and BCE ≤ 0.05; **exact memorization** = accuracy 100% and BCE ≤ 0.05. These are fitting
diagnostics, not generalization gates; partial improvement is reported as such.

## 6. Qualification and limits

Reuse the qualified D1 machinery unchanged (equations untouched); additionally verify scheduler boundary values, subset/order indexing,
nesting/quotas, absence of labels/ids/metadata from inputs, role restrictions and hash enforcement (tests), then a disposable CUDA qualification
(weights never initialize a fit). Process-scoped CUDA environment as before. Limits: sampled device memory ≤ 3,072 MiB; host ≤ 8 GiB; ≤ 2 h total
native execution (excl. compilation) with a global remaining-budget check; qualification ≤ 30 min, each A fit ≤ 25 min, each M fit ≤ 5 min;
bounded launcher with native exit codes, process-tree cleanup and orphan check; sampling limitations reported.

## 7. Independent verification and interpretation

`v69-d2 aggregate` recomputes all metrics in f64 from serialized predictions and subset metadata, requires complete ids, snapshot counts, recorded
exposures equal to the frozen stream, matching artifact hashes, and agreement with in-process summaries; refuses incomplete runs. After the campaign, E1/D1/gen-001
preservation is re-verified (append-only end receipt). Interpretation: both strong-fit 768 → larger-set trainability established under D2 (propose a separately
frozen generalization experiment); M fits and A fails → prioritize current-architecture/optimization-path investigation (M a simple control, not a proven
chess model); A fits and M fails → retain A, recurrence still has no demonstrated benefit; degradation with size for both → report the bracket, exposure
comparisons, dynamics, do not enlarge model or data; loss still falling → more budget is a hypothesis only; input dependence contracting toward the prior →
"collapse-like behaviour", cause not claimed. Stop after D2; no follow-up is executed.
