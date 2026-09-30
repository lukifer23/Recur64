# Workstation V2.5 - P2.5 results

Plan (pre-registered): `docs/WORKSTATION_V25_P25_PLAN.md`. Starting SHA
`729c1e5183f2b36bc86d68288ef2efea51bd2d09`. P2 is frozen history and is not reinterpreted.
MEASURED = produced by a command in this repo; INFERRED = reasoning, not tested.

## Phase 1 - LF (`legacy_facts_v25`) built and gated
QUESTION: can the legacy head-v2 policy and the exact CandidateFacts be combined (the missing
cell of the 2x2)? MODEL: the unmodified `ProbeModel` (wrapped, so probe_v1 record layout and
hashes are untouched) + a candidate-local fact delta
`Linear(64->1)(GELU(Linear(8->64)(facts)))` added to the legacy policy logit before the masked
softmax; final layer small nonzero init (std 0.01); facts do not feed WDL.
MEASURED: 26,810,585 parameters = L 26,809,944 + 641. Engineering gates (CPU): finite forward;
policy sums to 1; padding exactly 0; terminal rows safe; changed facts change the policy; ZERO
facts reproduce the legacy policy (max diff < 1e-5); facts gradient finite and nonzero for both
facts layers on update 1; fresh entropy 0.998 x uniform, worst top-1 1.52 x uniform; WDL
logits exactly 0; exact checkpoint round trip; batched == sync inference; missing facts are
errors. Identity: distinct architecture id + `fact_delta_contract`; the full 3x3 cross-refusal
matrix (probe_v1 / candidate_v25 / legacy_facts_v25, every ordered pair, both loader/config
mismatch forms) refuses by architecture id on real checkpoint directories; the frozen P4.5 hash
test and the Probe-serialization tests are unchanged and green. CUDA (RTX 2000 Ada, FP32):
device guard passed, forward finite (p50 25.9 ms at batch 32), 64x4 training step finite (peak
2.6 GB), lifecycle growth 0 MiB. fmt clean, clippy 0, 299 workspace release tests pass.
Evidence: `docs/evidence/v25/p25/model-info-legacy-facts.json`,
`docs/evidence/v25/p25/factorial/lf-cuda-qual.json`.

## Phase 2 - heavy-family holdouts (generated and audited; NOT yet evaluated by any model)
Heavy families KQQvK / KQRvK / KRRvK x M1/M2/M3, 500 positions per cell per holdout, the same
exact solver, shortest-correct-move target, <= 15% correct fraction, facts-ambiguity rule,
canonical dedup and fresh_no_history contract as before. Exclusion is by CANONICAL CLASS of the
retired and the replacement P1/P2 TRAIN/TUNE/CONFIRM (six datasets, 24293 classes,
exclusion-manifest digest `29302432847909bf57bc37dd327ce12974abc2515313d5a9abd81c847f925661`) and of the other holdouts; the training extension (built later) excludes all of them.

| holdout | role | positions | seed | sha256 digest | chance top-1 M1 / M2 / M3 |
|---|---|---:|---|---|---|
| A | P2.5-F factorial | 4500 | 0x7A130001 | `384a528eb037b268d3ca214d5d9c90575d657c7a513a6f498afdfe76dd4e9b07` | 0.043 / 0.058 / 0.076 |
| B | P2.5-D data scale | 4500 | 0x7A130002 | `13c8e018beea71058dbbbf510c723ebbe85b3a057c7b0e4a845a1039e107df10` | 0.042 / 0.059 / 0.078 |
| C | P2.5-O horizon (only if triggered) | 4500 | 0x7A130003 | `4ab951c6edd8dd4f531bb87d2f4373895d1fddb70efdf052b24c09a1b71d87d5` | 0.042 / 0.059 / 0.076 |

MEASURED: 100% independent GameState audit (0 failures in A, B, C); holdouts mutually
disjoint and disjoint from every excluded split by canonical class and exact FEN; no
holdout-exposure log exists (no model has evaluated any holdout). Evidence:
`docs/evidence/v25/p25/holdouts/`.

### Per-cell pool accounting (the pre-registered pool-limit rule is now resolved)
| cell | eligible pool | excluded (earlier splits) | available | holdouts (A+B+C) | REMAINING for extension (target 4,000) |
|---|---:|---:|---:|---:|---:|
| KQQvK M1 | 95649 | 2390 | 93259 | 1500 | 91759 |
| KQQvK M2 | 174163 | 2394 | 171769 | 1500 | 170269 |
| KQQvK M3 | 4409 | 2078 | 2331 | 1500 | 831 |
| KQRvK M1 | 111273 | 2384 | 108889 | 1500 | 107389 |
| KQRvK M2 | 306595 | 2398 | 304197 | 1500 | 302697 |
| KQRvK M3 | 211215 | 2391 | 208824 | 1500 | 207324 |
| KRRvK M1 | 41612 | 2364 | 39248 | 1500 | 37748 |
| KRRvK M2 | 122082 | 2393 | 119689 | 1500 | 118189 |
| KRRvK M3 | 108086 | 2384 | 105702 | 1500 | 104202 |

MEASURED: eight of nine heavy cells have at least 37,748 classes left. **KQQvK M3 has only 831**
(pool 4,409; 2,078 already used by earlier splits; 1,500 to the holdouts). RULE APPLIED (fixed
before this measurement): holdouts take priority, the extension takes all that remains up to
4,000, nothing is relaxed. So P25_DATA_V1 will hold 1,000 + 831 = **1,831** unique KQQvK M3
TRAIN positions, not 5,000; the other eight heavy cells get 5,000. `cell_balanced_v1` keeps
per-cell EXPOSURE equal, so KQQvK M3 will be oversampled (~3.7 local epochs at 400 updates
vs ~1.4 for a 5,000 cell). INFERRED: this slightly dilutes the "5k vs 1k" contrast in exactly one
of nine heavy cells; it is reported, not hidden.
