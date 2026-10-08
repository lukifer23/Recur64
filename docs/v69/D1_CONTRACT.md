# Recur64 V69 — D1 contract: bounded diagnostic of learning failure

Frozen before any D1 fit. Machine-readable parameters: `d1_config.json` (hash-enforced at launch).
D1 follows E1 (`RESULTS_E1_THREE_ARM.md`), which failed every gate: all arms predicted the class prior
and lost their input dependence. D1 asks only: *can the current model, or a plain direct-board MLP, even
memorize a tiny fixed fitting panel under the same recipe, and how much of the label is explained by
shallow features?* It is **not** a rerun, extension or retune of E1 and creates no new validation evidence.

## 0. Status, boundaries, precision statement

- Retrospective diagnostic. gen-001 validation is **already-exposed development evidence**; nothing in
  D1 is fresh confirmation. No replacement data, no new master-seed draw, no sealed-test access, no model
  evaluated on validation or sealed test, no search/self-play/policy/hierarchy comparison, no extra
  seeds, no automatic rescue, no E1 checkpoint or optimizer state is loaded into any D1 model.
- **Precision (adopted from the E1 qualification finding):** CUDA backend, f32 storage and accumulation,
  matmul inputs possibly TF32 selected by Burn/cubek autotune. D1 does **not** claim strict FP32. Backend,
  kernels and precision are not changed during D1. No CPU substitution for either neural model.
- Namespace: all D1 outputs under `artifacts/v69/d1/` (immutable once written; reruns refused). E1 files are
  never written. D1 outputs are separately identified; E1 predictions are not touched.

## 1. Integrity and provenance (enforced, not merely recorded)

1. `d1/e1_supplementary_manifest.json` hashes the actual E1 model, optimizer and metadata files (and E1
   predictions, reports, specs, init, intervention, qualification and audit files). It is a **verification
   taken at D1 start**, not a historical launch-time receipt. `v69-d1 verify-e1` re-hashes every entry and the
   E1 frozen-dependency hashes (`spec/frozen.json`); any mismatch stops D1.
2. `d1/frozen_d1.json` lists expected SHA-256 of every input per role group (learner, baseline, aggregator):
   D1 contract + config, panel rows, example stream, canonical init, MLP init, E1 frozen file, E1 manifest,
   E1 final checkpoint files, dataset manifest, fitting rows (and metadata for the aggregator). Each launch
   re-hashes the group through the role-restricted `Access` and **aborts on mismatch**.
3. Each D1 launch writes `provenance.json` binding: executable source identity (git head, dirty count, source
   digest), contract/config hashes, input/panel/order identities, initialization hash, completed updates, and the
   SHA-256 of the actual final checkpoint files and prediction files. The aggregator re-checks those hashes.
4. Role access: learner = seed + `data/fit.jsonl` + D1 inputs (never metadata, validation, sealed, pool);
   baseline fitter = fit and val **rows** only; panel selector (data side) = fit rows + fit metadata;
   aggregator = fit/val metadata + predictions. Model code never reads metadata.

## 2. Panel (32 examples, gen-001 fitting partition only)

Selection (`d1::select_panel`, host side, never uses any model output): candidates ordered by SHA-256-keyed
hash of the example id under the domain-separated label `d1/panel`. Phase 1: for each family (KQQvK, KQRvK,
KRRvK) × budget (1, 2) × class (neg, pos) take the first 2 candidates whose connected `group_id` is unused
(24). Phase 2: the six family × budget strata are ordered by keyed hash (`d1/panel/stratum`); in the first four,
add one positive then one negative by the same rule (8). Total 32 = 16 positive + 16 negative, **32 distinct
group_ids**; the selection stops with an error if any cell cannot supply the examples (no silent change).
Serialized: `panel_rows.jsonl` (model-facing `{id,fen,budget,label}` only, sorted by id), `panel_meta.jsonl`
(metadata for reporting), `panel_selection.json`.
Example stream: epoch e = Fisher–Yates permutation of the 32 panel rows from stream `d1_train_order/e`,
epochs concatenated to 2,000 × 16 samples (every example exactly 1,000 times); update u uses samples
[16u,16u+16), microbatch m uses [2m,2m+2). Serialized in `train_order.json`; identical for both models.

## 3. Models

- **D1-A**: the unchanged E1 one-pass architecture (arm A: one fast + one slow update, final readout; 1,876,417
  parameters). Initialized from the E1 canonical **untrained** init file (hash-enforced); fresh optimizers.
- **D1-M**: direct-board MLP, no attention/pooling/normalization/dropout. Input 842 = 64 squares × 13 attacker-
  relative piece one-hots (index `square*13 + code`, square order and codes exactly as `features::featurize`,
  rank-flip for Black attackers) + the 8 rule scalars + 2-way budget one-hot (budget 1 → [1,0], 2 → [0,1]).
  Layers 842→128→64→1, GELU (erf) between hidden layers, biases on all Linear layers: 842·128+128 = 107,904;
  128·64+64 = 8,256; 64+1 = 65; **116,225 parameters**. Initialization once from the domain-separated stream
  `d1_mlp_init/0`: Linear weights ~ N(0, 2/(fan_in+fan_out)), biases 0; written to `d1/mlp_init.bin` (hash-
  enforced). No seed search. The comparison changes several properties at once (pooling, depth, normalization,
  residual scaling, size): it tests practical trainability, not the isolated causal effect of pooling.

## 4. Fixed fits (identical mechanics for both models)

2,000 updates, microbatch 2 × 8 accumulation (effective 16), loss = stable BCE-with-logits, microbatch loss
divided by 8, accumulated gradient clipped **once** by global L2 norm to 1.0 (factor min(1, 1/(norm+1e-6)),
applied only when norm+1e-6 > 1), AdamW (β 0.9/0.999, ε 1e-8) with two instances for parameter groups:
decay group wd 1e-4 (rank ≥ 2 tensors), no-decay group wd 0 (biases, normalization). **Scheduler indexing:**
update index u = 0…1999; u < 20: lr = 5e-4·(u+1)/20; u ≥ 20: lr = 5e-5 + ½(5e-4−5e-5)(1+cos(π(u−20)/1979))
(update 1999 uses exactly 5e-5). The horizon (2,000) is a D1 diagnostic intervention, not an E1 continuation.
Serial runs (D1-A then D1-M); no early stopping, no best checkpoint, no extra updates, no restart; an
interrupted/resource-limited run is INCOMPLETE and is not a scientific failure.
Measurements on the 32-panel only, at updates 0 (initialization), 100, 500, 1,000, 2,000 (inference mode,
no gradient): logits → accuracy, balanced accuracy, BCE, AUROC, logit spread and class-conditional
distributions. Also recorded: per-update gradient norm (pre-clip), clipping, LR, loss, latency; parameter
movement by component (‖θ_t − θ_0‖₂ and relative) at each measurement; exposure counts; device/host memory.
**Final memorization criterion: 32/32 correct and BCE ≤ 0.05 at update 2,000.** These are fitting
diagnostics, not model-selection evaluations.

## 5. Qualification (before the fits; disposable weights, never used to initialize a fit)

CUDA (RTX 2050), documented process-scoped `CUDA_PATH`/`PATH`. D1-M: forward vs independent f64 reference,
gradient vs exact f64 finite differences, stable BCE value/gradient, input/label separation, 8×2 accumulation vs
batch-16, single clip per update and clip behaviour, parameter groups, checkpoint + both optimizers' restoration
(with fresh-optimizer negative control), state independence. D1-A: forward at the canonical init equals the E1
one-pass forward (same code path) and the f64 reference within documented tolerances (2e-3 absolute, the E1
TF32-class tolerance), plus the same machinery checks through the D1 trainer. Failures are preserved; a failed
qualification stops D1 before fitting.

## 6. Resource limits

Sampled device memory ≤ 3,072 MiB; host ≤ 8 GiB; ≤ 2 h total native diagnostic execution (compilation
excluded). Every owned run is under `run_limited.ps1` with an explicit wall cap (qualification 30 min; each fit
50 min; baseline 10 min), exit-code capture and process-tree cleanup with orphan check.

## 7. Shallow feature baseline (host-side, explicit authorization; not a fallback for the neural fits)

Retrospective exploratory analysis. Fit once on all 768 gen-001 fitting rows, evaluate once on the 384
validation rows. Features (all from the child FEN with the defender to move; no exact-mate calls, no search, no
root metadata, no E1 predictions, no ids, no target-dependent or neural features). Base features (11):
1 number of attacker queens; 2 number of attacker rooks; 3 defender-king distance to the nearest board edge;
4 defender-king Chebyshev distance to the nearest corner; 5 Chebyshev distance between the kings; 6 minimum and
7 maximum Chebyshev distance from the two attacker major pieces to the defender king; 8 number of attacker
majors adjacent (Chebyshev ≤ 1) to the defender king; 9 defender currently in check (0/1); 10 number of legal
defender moves; 11 number of legal defender moves that capture an attacker major piece. Plus the budget
indicator (budget = 2) and its interaction with each base feature (11): **23 features**. Each feature
standardized with the fitting-set mean and population standard deviation (std ≤ 1e-6 ⇒ 1). Objective =
mean BCE + (λ/2)·Σ w², λ = 0.01, intercept excluded; gradient (1/n) Xᵀ(σ(z) − y) + λ w; full-batch gradient
descent from zero, 1,000 steps, learning rate 0.05. Threshold: logit 0. No regularization/feature/threshold
search on validation. Reported: fit and validation BCE, balanced accuracy, AUROC, per-cell metrics, and
95% intervals from a connected-`group_id` bootstrap (2,000 resamples, stream `d1_bootstrap/<set>`). Strong
results indicate shallow predictive structure; weak results do **not** show that search is necessary.

## 8. Independent aggregation and decision rules

`v69-d1 aggregate` recomputes all metrics in f64 from serialized predictions and metadata, checks complete ids,
labels, counts and artifact hashes, and refuses incomplete runs. Decision (per the owner's rules; every outcome
reported, including partial success): both memorize → basic tiny-panel trainability established, larger-set
learning open; MLP memorizes and D1-A fails → prioritize current architecture/optimization-path diagnostics;
neither memorizes → prioritize shared input/target/loss/update mechanics and training dynamics (do not conclude the
task is unlearnable); D1-A memorizes and MLP fails → retain the current architecture as a candidate and investigate
why the control was insufficient. The winner is never chosen from confidence or prose.

## 9. Stop conditions

Stop and report on: any frozen-hash mismatch, failed qualification, resource-limit breach, non-finite loss or
gradient (INCOMPLETE), or completion of D1. Not executed in D1: any follow-up, new validation campaign,
architecture variant, extra seed, hierarchy comparison, sealed-test inference, policy, search or self-play.
