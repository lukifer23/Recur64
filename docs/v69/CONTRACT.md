# Recur64 V69 — frozen contracts (data, teacher, partitions, architecture, comparisons, evaluation)

Status of this document: **FROZEN before the real V69 generation run** (seed drawn and
recorded first; see §2). Items are labelled **PROPOSED** (decided here, not empirically
validated), **MEASURED** (observed in this pass; numbers live in `DATA_PHASE_RESULTS.md`)
or **NOT RUN**. Nothing below claims model learnability, reasoning, or chess strength.

## 0. Custody and exclusion

V69 is scientifically fresh. **No scientific data or artifact of any earlier run or
branch enters V69**: no earlier TRAIN/DEV/CONFIRM/eval positions, ProofTargets, labels,
generated examples, checkpoints, pretrained encoders, optimizer states, acquisition
packets, donor maps, cached oracle results, or outcome-based selection by earlier models.
Historical results are motivation only; they are not V69 measurements, references,
initializations or fixtures.

Reused (software only): `recur64-core` rules authority (cozy-chess 0.3.4 move generation,
`rules::classify`), Cargo workspace/lock/toolchain, burn 0.21.0 pin (not exercised by the
data crate). The exact-mate algorithm is **re-implemented from the definition** in
`crates/recur64-v69/src/oracle.rs`; the historical `MateSolver`/ProofTarget code was
**not** ported (it exists only on experiment branches and is coupled to their proof/trace
assets). Fixtures in tests are hand-built positions or come from a TEST-ONLY seed
(`5eed…`) that is not the experiment seed.

Enforcement: `Custody::resolve` (`src/custody.rs`) is applied to every V69 input/output
path of `v69-data`. A path is refused (process error, nonzero exit) if it resolves outside
the V69 artifact root (`artifacts/v69/`, git-ignored), contains a historical component
(`runs`, `evidence`, `checkpoints`, `datasets`, `proof_targets`, `acquisition`, `donor`,
`oracle_cache`, `x1`, `x15`, `x2`, `h1`, `h3`, `r15`) or the root sits inside a historical
worktree (`Recur64`, `Recur64-v5`, `Recur64-v6`). Tested in `tests/semantics.rs`.

**Stated limitation.** The guarantee is *independent generation and zero historical asset
reuse*. It is **not** a certificate of zero accidental position overlap with earlier
datasets: that would require comparing against them, which this experiment forbids.

## 1. Fresh-position generator (PROPOSED, frozen)

- Families: KQQvK, KQRvK, KRRvK; attacker colour White or Black (drawn per attempt).
- Domain: pawnless, no castling rights, no en-passant, halfmove clock 0, fullmove 1, **no
  prior history**, attacker to move at every root. This restricted domain is *not*
  arbitrary game-history coverage; no claim is made beyond it.
- Algorithm: `sample_root(seed, family, index)` — a pure function. One SplitMix64 stream per
  (family, attempt index); attacker colour bit, then four squares (attacker king, two
  attacker pieces, defender king) each drawn independently and uniformly from 64.
- Rejections, all counted: overlapping squares; adjacent kings; defender king already in
  check with the attacker to move (illegal); any other board-validation failure
  (cozy-chess FEN validation is the legality authority).
- No neural model, old FEN, curriculum or cache participates.
- Roots are accepted only when the exact oracle proves minimal forced-mate depth
  **M ∈ {2, 3}** (attacker moves). M=1 roots are rejected; "no mate within 3" is rejected
  (exhaustively proven); resource-limited roots are UNKNOWN and discarded.
- Generation proceeds in deterministic rounds of `round_size` attempts per family;
  results are ordered by attempt index so thread count cannot change the data.

## 2. Seed and random streams (PROPOSED, frozen)

- Master seed: 32 bytes from the OS CSPRNG (`System.Security.Cryptography.RandomNumberGenerator`),
  written by `scripts/v69/new_seed.ps1` to `artifacts/v69/seed/master_seed.hex` (read-only)
  **before** any generation; refuses to overwrite; never redrawn.
  Fingerprint (first 16 hex of SHA-256 of the seed): **`5710d34e89d9dd35`**, recorded
  2026-10-08T21:51:33Z UTC.
- `stream(label, index)` = SplitMix64 seeded by the first 8 bytes of
  SHA-256("recur64-v69/stream/v1" ‖ seed ‖ len(label) ‖ label ‖ index_le).
- Labels: `generation/<family>` (index = attempt), `partition/<family>/<M>` (index = block),
  `partition/order` and `selection/{root,child}` (content-keyed hashes), `model_init`,
  `train_order`, `intervention` (derived and reserved; unused in this pass).
- No seed search. A single run executes; a failure to reach quotas is reported, not retried
  with a new seed.

## 3. Canonicalization and symmetry (PROPOSED, frozen)

Symmetry group of order 16 = 8 dihedral board symmetries × attacker-colour relabelling.
A position is encoded attacker-relative (each piece is `attacker`/`defender`, not
White/Black) plus a "attacker to move" bit; the canonical key is the lexicographic minimum
over the 8 dihedral images (65 bytes). Valid because the domain has no pawns, castling or
en-passant and clock is not part of identity. Roots and children are identified by this
**full 65-byte key** (ids are SHA-256 of the full key; never a hash of a prefix, and never
a Zobrist hash, for audit). Symmetry-equivalent positions are one sample.
Halfmove clock is excluded from identity (children carry clock 1 after one non-capturing
move; roots carry 0).

## 4. Target semantics (PROPOSED, frozen)

- Budget counts the **original attacker's own moves**; defender choices are universally
  quantified.
- Root: exact minimal forced-mate depth M ∈ {2,3}: mate within M is proven **and** absence
  of mate within every smaller depth is proven (iterative exhaustive queries).
- The complete legal root move set is enumerated. For each legal root move the child is
  applied (attacker identity preserved; defender to move). Terminal children are classified
  first by `recur64_core::rules::classify` (checkmate/stalemate/dead position) and are
  **excluded** from this task (counted). For each nonterminal child the teacher is queried
  directly with remaining budget n = M − 1: target = *positive* iff the attacker forces
  mate within n more attacker moves. A negative means "no forced mate within this budget",
  **not** eventual game loss. Labels are never inherited by deeper descendants.
- The correct-move set is the set of positive children; it must be non-empty (asserted).
- Rules authority: terminal and draw conditions via `rules::classify` with repetition count
  1 and no ply cap (no history in this domain; ≤ 6 plies from a zero-clock root cannot
  reach repetition or the 50-move rule).
- UNKNOWN (node budget exhausted) is never negative: it discards the root.

## 5. Oracle resource limits and safety (PROPOSED, frozen)

- ≤ 5,000,000 visited nodes per individual query (`mate_within_root` /
  `mate_within_child`), counted per query; exhaustion returns `Unknown` through `Result`
  propagation. Cache entries are written only when a subtree returns a completed exact
  value, so an aborted query stores nothing partial (tested: `node_limit_yields_unknown_and_never_poisons_the_cache`).
  The cache (exact `Board` keys, not hashes) is cleared at the start of every root; each
  root begins with an empty cache.
- ≤ 2 hours host wall time: internal deadline (an incomplete round is discarded whole)
  and an external hard limit in `scripts/v69/run_limited.ps1`.
- ≤ 8 GiB process memory: external watchdog (`PrivateMemorySize64`/working set, 500 ms
  poll) kills the whole process tree (`taskkill /T /F`); native exit code, reason and
  peak memory are captured to `<log>.exit` / `<log>.peak`.

## 6. Partitions, quotas, leakage control (PROPOSED, frozen)

- 12 cells = 3 families × 2 remaining budgets (n=1 from M2 roots, n=2 from M3 roots) × 2
  classes. Quotas per cell: fit 64, validation 32, sealed test 32 → 768 / 384 / 384 = 1536.
- Examples are only nonterminal immediate children of accepted M2/M3 roots.
- Groups = connected components of roots linked by sharing **any** canonical child position
  (all children, including unselected and terminal ones, across both budgets; full-key
  equality). A group never spans partitions.
- Assignment (label-blind, outcome-free, deterministic, made before any model exists): per
  stratum (family, depth of the group's smallest root key) groups are ordered by a keyed
  hash and dealt in blocks of four `[Fit, Fit, Val, Test]` permuted by the partition stream.
- Within each partition and cell, roots are visited in keyed order and at most **2
  positive and 2 negative children per root** are taken (`CAP_PER_ROOT_PER_CLASS`; a design
  choice that limits any parent's weight), skipping a canonical child already used at the
  same budget. Negatives are uniform over incorrect nonterminal moves (so many are
  trivially bad moves — a property of the task, not hidden).
- Capacity: rounds continue until every cell fills or a limit hits. Stopping uses counts
  only. If quotas are infeasible, the run stops, reports the capacity problem, and does
  not weaken grouping, substitute terminal children, reduce quotas or change the generator.
- Files: `data/{fit,val}.jsonl` and `sealed/test.jsonl` hold **only** `{id, fen, budget,
  label}` (model-visible fields). Parent/group/root-depth metadata lives in separate
  `*.meta.jsonl` files and must not be model inputs.
- **Sealed test:** `sealed/` files are set read-only and hash-pinned in
  `MANIFEST.sha256.json`. No model is run on them in this pass; release requires a frozen
  candidate and evaluation protocol. The audit reads their labels only to verify exactness.

## 7. Verification (PROPOSED, frozen)

`v69-data audit`: manifest hashes; id uniqueness/derivation; exact quotas and class
coverage; no group/root/canonical child across partitions; no terminal child; every child
is defender-to-move; recomputed canonical keys match; every contributing root is re-analysed
from an empty cache and must reproduce depth and every child label; an independently
structured exhaustive enumerator (`src/reference.rs`: no cache, no budget, min-max value
instead of budgeted predicate, no `rules` module) re-checks a deterministic per-cell sample
of child targets (and monotonicity of negatives) and the minimal depth of sampled M2 roots.
Unit tests (`tests/semantics.rs`) cross-check solver vs reference on newly generated
roots (both colours, both budgets, both classes), generator purity, canonical invariance
under all 16 symmetries, brute-force move-generation cross-check, stalemate/mate boundary,
budget-relative negatives, node-limit/no-poisoning, grouping, custody.

## 8. Feasibility acceptance (data phase)

Pass iff: all 36 (partition×cell) quotas met exactly; audit passes; limits respected
(2 h, 8 GiB, 5 M nodes/query); custody guard never tripped. Otherwise stop with measured
throughput, acceptance rates, bottleneck, remaining-cost estimate with uncertainty, and a
concrete revision proposal (no automatic extension).

## 9. V69 architecture (PROPOSED — specified, not built, not trained)

Standalone model: 64 square tokens, 8 workspace tokens, width 192, 6 heads (head dim 32),
FFN width 576, two initial board-encoder blocks.
- Board init: token = piece-code embedding (13) + learned square embedding (64) + a linear
  map of rule-state scalars (side to move, attacker-relative-to-mover indicator, halfmove
  clock/100, 4 castling flags, en-passant flag) + remaining-budget embedding; two
  pre-norm encoder blocks (self-attention + FFN).
- Workspace init: 8 learned slot embeddings + a linear map of the mean-pooled encoded board.
- **Fast shared block** (updates board): board self-attention → board attends to workspace →
  FFN. **Slow shared block** (updates workspace): workspace self-attention → workspace
  attends to board → FFN. Pre-norm residual sublayers, fixed residual scale 0.1.
  "Refresh access to the encoded board" is realised (PROPOSED form) as: the fast block's
  board self-attention keys/values are computed over the concatenation of the current board
  state and the fixed encoder output E (read-only, recomputed from E every fast update), so
  every fast update can re-read the original encoding. The exact form is not validated.
- Head: mean-pooled workspace → LayerNorm → MLP(192→192→1) → forced-mate logit.
- Inputs: current board, current rule state, remaining attacker-move budget,
  attacker-relative-to-side-to-move indicator. Excluded: history frames, root boards,
  ids, oracle outputs, root-depth metadata, B0 features, acquisition statistics.
  *Note:* in this dataset every example is defender-to-move, so the indicator is constant;
  it is retained for contract fidelity and will carry no information here.
- Recurrent state resets per example; full BPTT through the short unroll; no adaptive
  halting, no detached carry, no equilibrium-gradient approximation, no cross-example memory.
- **Parameter estimate (PROPOSED arithmetic, not measured):** attention 148,224; FFN 221,952;
  encoder block 370,944 ×2 = 741,888; fast block 519,936; slow block 519,936 (shared,
  counted once each); input embeddings/projections ≈ 17,664; workspace init ≈ 38,976;
  head ≈ 37,633; final norms 768 → **≈ 1.877 M parameters**. This is **below** the stated
  "≈ 2–3 M" target and under the 4 M ceiling; the specified dimensions cannot reach 2 M
  without changing them (e.g. a third encoder block adds ≈ 0.37 M). Flagged for owner
  decision; no dimension was changed here. Feasibility on 4 GB needs measured CUDA
  qualification (NOT RUN).

## 10. Prospective comparisons (PROPOSED — frozen, NOT RUN)

All arms share bit-identical initial parameter tensors (from the `model_init` stream) and
independent fresh optimizers; training order from the `train_order` stream (identical
across arms).
- **A:** 1 fast + 1 slow update; final readout. **B:** 4 cycles of (2 fast, 1 slow);
  readouts at cycles 1, 2, 4. **C:** 6 alternating fast/slow pairs; readouts at pairs 2, 4, 6.
- Loss: mean BCE over the prescribed readouts; final-readout BCE also reported separately.
- B and C have equal total block-call counts (12 each) but not guaranteed equal FLOPs or
  latency; measure before claiming compute parity.
- First fits per arm: 600 updates, physical microbatch 2, effective batch 16, AdamW, peak LR
  5e-4, 20-update warmup, cosine to 5e-5, betas 0.9/0.999, eps 1e-8, weight decay 1e-4
  excluding biases and normalization, global gradient clip 1.0, 2 h wall limit per arm,
  fixed final endpoint (no best-checkpoint selection). These are proposed, not validated.

## 11. Evaluation contract (PROPOSED — NOT RUN)

Validation is for the predeclared feasibility decision; sealed test stays inaccessible to
fitting and model selection (no model touches it in this pass). Gates: fitting balanced
accuracy ≥ 95%; validation balanced accuracy ≥ 75%; validation final-readout BCE ≤ 0.55;
≥ 15-point drop on recipient labels under a fixed, target-independent
within-(family, budget) board derangement (from the `intervention` stream); all engineering
and custody gates pass. Also report donor-label accuracy, board erasure (as an
out-of-distribution diagnostic), per-cell metrics, Brier score, parent-group uncertainty
(group bootstrap), actual latency and memory. These gates establish preliminary task
learnability only — not reasoning, generalization beyond the domain, or chess strength.

## 12. Roadmap (context only)

CUDA graph qualification → bounded three-arm fit → fresh-stream replications and
supervision/compute confound tests → direct legal-move policy on additional fresh roots →
held-out depth/family evaluations → optional bounded search at equal inference cost.
If simpler recurrence wins, drop the hierarchy; if one-pass suffices, keep it; if all fail,
diagnose task/representation before adding complexity. Historical B0 is not a control.
