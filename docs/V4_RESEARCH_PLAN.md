# V4 — `evidence_belief_v4`: learned active hypothesis testing

**Status: PRE-REGISTERED (2026-10-02). Frozen before any V4 model code, V4 data generation, or V4 measurement exists.**
Branch `experiment/workstation-v4-evidence-belief`, from accepted source HEAD `40916509936e83468e2fe58c455a33a246123545`
(V3.5, closed `PARTIAL - CONTENT`). Scope of this document: V4-P0/P1 (architecture, mechanism qualification on TRAIN only).
Companion: `V4_NOVELTY_REVIEW.md`. Ledgers: `V4_EXPERIMENTS.md`, `DECISIONS.md` (V4-D#).

V4 is a new architecture line. It is **not V3.6** and not another rescue of `active_search_v3`. No V3.x training continues
(no second DAgger iteration, no LR tweak, no extra V3.5 updates). The V3/V3.5 branches are not altered.

## 1. Causal chain V4 responds to

| # | Finding | Source |
|---|---|---|
| 1 | Recurrence over unchanged information gave no benefit. | `WORKSTATION_V25_SUMMARY.md`; V3 plan thesis |
| 2 | Raw exact future-state information helps: ALL-INFO − B0 on KQRvK M3 = **+0.2351**, CI [0.2044, 0.2671] (Gate I PASS). | `V3_P6_RESULTS.md` |
| 3 | V3 teacher forcing leaked the query pattern: teacher-forced top-1 ~0.99 came from the path alone; ACTIVE did not use content. | `V3_P5_RESULTS.md` (P5.2 ablation) |
| 4 | V3.5 on-policy relabelling removed enough of that confound for content to be measurable: Content-Use **+0.0369** nats, CI [+0.0185, +0.0553], every seed positive. | `V35_RESULTS.md` |
| 5 | ACTIVE_B8 − B0 = **−0.0222** (top-1), CI [−0.0400, −0.0049]. | `V35_RESULTS.md` |
| 6 | ACTIVE_B8 − FIXED_B8 = **−0.0227**, CI [−0.0396, −0.0058]: the learned selector is slightly worse than breadth-first. | `V35_RESULTS.md` |
| 7 | B0 improved (M3 top-1 0.457 → 0.539) while the query-budget arm regressed relative to B0: the base and query objectives share parameters. | `V35_RESULTS.md` |
| 8 | Selector targeting is still weak: KQRvK M3 proof-admissible 0.267-0.290, off-target 0.50-0.52, proofs complete 19-32 / 750 vs ideal 324 / 750. | `V35_RESULTS.md` |

Reading: information is sufficient (2), content is usable (4), but (a) the integration path is entangled with the base
policy (7), (b) the selector is trained to hit a *proof set* rather than to improve the *decision* (8), and (c) query
metadata is a standing temptation to learn answers from routing (3). V4 targets exactly these three.

## 2. Research question
Can a neural system improve its root decision by actively gathering a small number of exact future-state observations,
when those observations are explicit evidence about competing root hypotheses and queries are chosen by predicted
decision value? Search is information acquisition, not imitation of MCTS/A*.

## 3. Architectural laws (mandatory, each backed by a test in §8)
- **A, base isolation.** Root-only path yields immutable `z0`. Evidence adds `delta_z`; `z_t = z0 + delta_z_t`. B0 *is* the base tower.
- **B, content causality.** Routing metadata may route evidence, never constitute it. Zero content ⇒ exactly-zero message ⇒ bitwise-identity update. Guaranteed by construction (bias-free, content-multiplied), not learned.
- **C, explicit evidence.** Every real query yields an inspectable `EvidenceMessage` (norm, destination branch, per-candidate delta, trust, cumulative contribution).
- **D, decision-aligned acquisition.** The selector predicts the marginal root-decision value of a query. ProofTrace is diagnostic only.

## 4. Architecture

```
root board ──► BaseTower ──► h_1..h_N (one hypothesis token per legal root action) ──► BaseReadout ──► z0 (immutable)
                                   │ (detached into evidence path in Stages B-D)
 StateQuery(parent,action) ─► child StatePacket (exact)
        │ CONTENT: root/parent/child obs, action content, check/terminal, parity
        ▼
   EvidenceEncoder (bias-free, f(0)=0) ──► EvidenceMessage m_t ──► EvidenceLedger {m_1..m_t} (set, no order embedding)
        ROUTING: root branch, parent link, depth, parity ─ multiplicative weights only ─┐
                                                                                         ▼
        h_i (detached) ──attend over ledger──► trust(c)·evidence_delta_i(c) = delta_z_i ──► z_t = z0 + delta_z_t
        QueryUtilityHead(H, z_t, ledger summary, FrontierEdgeView[no child fields]) ──► predicted U(e) ──► argmax ──► next query
```

### 4.1 Module contracts
| Module | May read | May not read | Output |
|---|---|---|---|
| `BaseTower` | root board, legal actions, CandidateFacts of the root | any query result | `h_i`, `z0` |
| `EvidenceEncoder` | root/parent/child content, action content, check/terminal, side-to-move | branch id, depth, node ids, query index | `m` (zero iff content zero) |
| `EvidenceLedger` | messages + relative structure (same-branch, depth, parity) | absolute query index, acquisition order | set of messages |
| `BeliefUpdate` | detached `h`, ledger | `z0` as writable, any child content directly | `delta_z`, trust, norms |
| `QueryUtilityHead` | `H`, `z_t`, ledger summary, `FrontierEdgeView` | the unseen child's content | scalar predicted `U` |

Content-causality construction: all content-path `Linear` layers have `bias = false`; activations satisfy f(0)=0
(SiLU/GELU/tanh); normalisation is scale-only; routing weights multiply values that are already content-derived;
`trust` multiplies the delta, so even a nonzero trust cannot create evidence from zero. The final readout is bias-free.
Sizing target 20-40M parameters, set by measurement (V4-B), not inflated for capacity.

### 4.2 Utility definition
`U(e | S) = CE_before − CE_after_e`, root correct-target CE under the current evidence set vs. after integrating the
real child of frontier edge `e`. Positive: helped. Negative: harmed. Computed on TRAIN only, in a detached fork
(`QueryManager::new(state.clone())`, separate `probe_queries` counter), then discarded. The inference head predicts `U`
before seeing the child. Selection is `argmax predicted U`: no PUCT, UCB, visit counts, or hand-coded backup.
Known limit (Callaway et al., BMPS): one-step myopic value can be zero or negative for *complementary* computations, so
D/E/F also report the multi-step diagnostic where feasible.

### 4.3 Hygiene decisions
- New V4 weights from scratch. Prior checkpoints are references only.
- `CandidateFactsV1` is audited before the utility head uses it. Any field describing the child position (e.g. gives-check) is excluded; only parent-observable fields are whitelisted.
- Evaluation-only `ZeroContent` / `ShuffledContent` ablations are unconstructible outside evaluation (same pattern as `query_content_ablation_v1`).

## 5. Staged training protocol
- **A, base.** Train `BaseTower` on `V4_TRAIN_FIT`; check B0 on `V4_TRAIN_DEV`.
- **B, evidence integration.** Base frozen; label-independent FIXED/RANDOM schedules; train Encoder/Ledger/Update.
- **C, counterfactual utility.** Evidence frozen; generate probe labels; train `QueryUtilityHead` (one loss chosen by bounded TRAIN studies: pairwise ranking vs clipped regression, optional sign auxiliary; no blind stacking).
- **D, on-policy integration.** belief → utility → query → evidence → belief. Evidence parameters trainable; base frozen; B0 bit-identical before/after. Joint refinement only under a later pre-registered contract.
- **Base-freeze rule.** Default: frozen. Confirmed or changed only by TRAIN-only drift/benefit measurement, recorded in the ledger before final evaluation. Never decided on V4_TUNE.

## 6. Data
- **TRAIN:** `P25_DATA_V1` TRAIN (44,332 positions) via `load_working_split`/`load_dataset(Expected::train())`.
- **TRAIN-dev partition (mechanism studies only, not V4_TUNE).** `canon` string of each position → `SHA-256("v4_train_dev_v1|" + canon)`, first 8 bytes big-endian as u64; `u64 mod 10 == 0` ⇒ `V4_TRAIN_DEV`, else `V4_TRAIN_FIT`. Split by canonical class, so no class straddles both. Digests of both lists are recorded in the V4-C evidence file. Rule frozen here, before it is applied.
- **`V3_TUNE_V1` → `HISTORICAL_REGRESSION_ONLY`.** It was repeatedly inspected and influenced V4 design. No V4 gating or tuning use.
- **`V4_TUNE_V1`** (preregistered rule):
  - families KQRvK and KRRvK, mate depth M1/M2/M3, target **1,000 unique positions per cell** (6,000 total);
  - generation seed **`0x7A40_0001`**, committed here before generation;
  - hard-disjoint by exact FEN **and** canonical class from every prior set: V2/V2.5 data, P25 TRAIN and retired/replacement splits, HOLDOUT_A/B/C, V3_TUNE_V1, all V3/V3.5 generated sets (exclusion inventory digest recorded);
  - deterministic regeneration must reproduce the digest; independent audit must pass;
  - **no reduction of count after seeing performance.** If 1,000/cell cannot be generated under this rule, stop and report;
  - generated, sealed (`sealed: true`, `evaluated: false`), **not evaluated** in P0/P1. V4 training and mechanism code paths refuse it.
- **HOLDOUT_C** remains sealed and unevaluated; its seal is re-verified (digest `4ab951c6…d87d5`) and logged.

## 7. TRAIN-only mechanism questions and pre-registered pass rules
All measured on `V4_TRAIN_DEV`; three seeds (5101-5103) where a trained model is involved; paired bootstrap, 20,000
resamples, fixed bootstrap seeds listed in V4-E ledger entries before each run. Thresholds below are frozen now.

| Q | Question | Pass rule |
|---|---|---|
| A | Non-trivial content-dependent update learned | CE_B0 − CE_B8 (normal, same fixed/random paths) CI wholly > 0, every seed > 0; delta_z non-degenerate (cumulative norm > 0, not constant across positions) |
| B | Normal content beats ablation on identical paths | CE_zero-content − CE_normal and CE_shuffled − CE_normal, both CI wholly > 0, every seed > 0 |
| C | Zero content returns exactly B0 | bitwise-equal logits (tolerance 0) on every dev position |
| D | A single query sometimes helps, sometimes harms | on probe labels, fraction U>0 and fraction U<0 each ≥ 0.05, and positive/negative mass both non-trivial (std(U) well above the label-noise floor measured by repeated identical probes, which must be exactly 0) |
| E | Head ranks realised utility above random | within-position Spearman mean > 0 and pairwise accuracy > 0.5, both CI wholly above null; calibration table reported |
| F | Best-predicted query beats random frontier edge | mean U(argmax predicted) − mean U(random probed edge), CI wholly > 0; also regret vs best probed edge and fraction picked with U>0 |
| G | 2/4/8 useful messages improve the *decision*, not just confidence | top-1 at B8 − B0 CI wholly > 0 **and** correct-mass improves monotonically over B0→B2→B4→B8; CE-only gains with flat top-1 are reported as confidence-only, not a pass |

Stop rules: **A, B or C failing ⇒ stop architecture development and report.** D-F showing no learnable signal ⇒ reconsider
the utility formulation before any final science. Results are reported as measured; thresholds are not edited after seeing them.

## 8. Invariant tests (hard correctness gates, CPU, before any serious training)
1 B0 independent of budget · 2 B0 logits exactly the base logits · 3 zeroed content ⇒ zero message · 4 zero message ⇒
identity update · 5 routing metadata with zero content cannot change policy · 6 permutation-invariant over acquisition
order · 7 real content changes evidence · 8 shuffled unrelated content gives a different result · 9 deltas finite and
bounded · 10 parameter count budget-independent · 11 exact StateQuery accounting · 12 states only through StateQuery ·
13 utility scoring cannot access unseen child content (by type) · 14 probe forks cannot contaminate the real trajectory ·
15 deleting all evidence reproduces B0 · 16 V3/V3.5/ALL-INFO/Probe identities refused by V4 and V4 refused by every
historical command.

## 9. Proposed final V4_TUNE gates (NOT APPLIED)
For the later V4-P2/P3 phase, to be re-preregistered then: Gate I-V4 `ACTIVE_B8 − B0` (CI wholly > 0); Gate II-V4
`ACTIVE_B8 − FIXED_B8` and `− RANDOM_B8` (CI wholly > 0); Content-Use (`CE_ablated − CE_normal` > 0); B0-integrity (bit-
identical B0 before/after active training); selector regret/positive-utility rate vs random. Seeds, thresholds, and
bootstrap seeds to be frozen before V4_TUNE is opened.

## 10. Prohibited in this pass
MCTS/PUCT/UCB/minimax backup/A*, Searchformer-style trace imitation, Stockfish/Leela/tablebase distillation, Docker,
Python training, custom autodiff/CUDA, jobs > 2 h without owner approval, any V4_TUNE or HOLDOUT_C evaluation, final
multi-seed science.

`V4 FINAL SCIENCE NOT RUN.` `HOLDOUT_C REMAINS SEALED AND UNEVALUATED.`
