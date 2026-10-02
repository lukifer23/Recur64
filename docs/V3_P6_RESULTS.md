# V3 P6 results: ALL-INFO information-sufficiency control and Gate I

**Status: P6 COMPLETE. GATE I PASS - RAW FUTURE-STATE INFORMATION IS SUFFICIENT FOR THIS MODEL FAMILY ON V3_TUNE_V1.**
**DAGGER NOT RUN. P7 NOT RUN. HOLDOUT_C NOT EVALUATED.**

Pre-registered in `docs/V3_P6_PLAN.md` (V3-D22 to V3-D25) before any ALL-INFO model was evaluated on TUNE. Everything below
is TUNE (V3_TUNE_V1) evidence; HOLDOUT_C was never loaded. Gate I was applied exactly once
(`docs/evidence/v3/v3-p6-gate1.json`).

## 1. Gate I

Rule (frozen, unchanged from the research plan): ALL-INFO - B0 >= +0.20 top-1 on KQRvK M3 (n = 750) AND the paired 95%
position-level bootstrap CI wholly above zero. Estimator: `d[i,s] = 1(AllInfo_s correct on i) - 1(B0_s correct on i)`,
`d_i = mean_s d[i,s]`, `Delta = mean_i d_i`; 20,000 resamples keeping the three seed pairs together; SplitMix64 seed
`0x7A160001`; percentile ranks 499 and 19499.

| quantity | value |
|---|---|
| **Delta (three-seed mean paired delta)** | **+0.2351** |
| paired bootstrap 95% CI | [0.2044, 0.2671] |
| threshold | +0.20 (inclusive); CI lower bound must be > 0 |
| per-seed deltas (5101 / 5102 / 5103) | +0.2560 / +0.2227 / +0.2267 (range +0.2227 to +0.2560) |
| result | **PASS** |

All three seeds individually show a positive delta above +0.20 (there is no "all seeds positive" condition in the frozen
rule; it is reported as a stability diagnostic), and the CI lower bound (0.2044) is itself above the +0.20
threshold. The margin over the threshold is modest (+0.035), so the conclusion is "clears the bar", not "far beyond it".

### KQRvK M3 top-1 per seed

| seed | B0 (ACTIVE, zero queries) | ALL-INFO | delta |
|---|---|---|---|
| 5101 | 0.4493 | 0.7053 | +0.2560 |
| 5102 | 0.4893 | 0.7120 | +0.2227 |
| 5103 | 0.4320 | 0.6587 | +0.2267 |
| mean | 0.4569 | 0.6920 | +0.2351 |

ALL-INFO absolute KQRvK M3 top-1 is **0.692** against the non-gating reference
value 0.75: below it, and irrelevant to pass/fail by the frozen rule. B0 seeds 5101 and 5102 are the accepted P5 final
checkpoints (re-evaluated bit-identically to the committed evidence); seed 5103 is the pre-registered
`p6_baseline_replication_v1` (V3-D22).

## 2. All cells (mean over the three paired seeds)

Per-position results were saved, so every cell is a mean of per-position values; `n = 750` per cell.

| cell | B0 top-1 | ALL-INFO top-1 | B0 correct mass | ALL-INFO mass | B0 CE | ALL-INFO CE | B0 entropy | ALL-INFO entropy |
|---|---|---|---|---|---|---|---|---|
| KQRvK M1 | 1.000 | 1.000 | 0.999 | 1.000 | 0.369 | 0.368 | 0.374 | 0.367 |
| KQRvK M2 | 0.699 | 0.806 | 0.512 | 0.640 | 1.727 | 1.232 | 1.766 | 1.471 |
| **KQRvK M3** | 0.457 | 0.692 | 0.343 | 0.528 | 2.437 | 1.937 | 2.179 | 1.668 |
| KRRvK M1 | 1.000 | 1.000 | 0.999 | 1.000 | 0.093 | 0.091 | 0.100 | 0.091 |
| KRRvK M2 | 0.716 | 0.890 | 0.575 | 0.771 | 1.542 | 0.852 | 1.430 | 1.028 |
| KRRvK M3 | 0.669 | 0.811 | 0.499 | 0.713 | 1.971 | 1.377 | 1.726 | 1.147 |
| pooled (4,500) | 0.757 | 0.867 | 0.654 | 0.775 | 1.357 | 0.976 | 1.262 | 0.962 |

Reading: ALL-INFO improves top-1 on every non-trivial cell and, unlike ACTIVE with queries (P5), it **also improves
correct mass and CE** rather than trading calibration for sharpness. The M1 cells are solved by both (top-1 1.0). The
largest gains are on the hard cells (KQRvK M2/M3, KRRvK M2/M3). ALL-INFO is still far from solving M3: about three in ten
KQRvK M3 positions remain wrong.

## 3. Secondary diagnostics (reported, never gating)

Gate I is explicitly not a pure causal information-only effect: the integrator, the optimisation and the training
distribution differ between the two models, and no statistical correction is applied.

| quantity | ACTIVE (B0 reference) | ALL-INFO |
|---|---|---|
| parameters | 30,853,790 | 30,842,524 (-11,266, 0.0365%) |
| future states supplied per position | 0 (B0) | median 140, mean 147 on TRAIN; TUNE mean 142.2 |
| TUNE pooled policy CE (mean of 3 seeds) | 1.357 | 0.976 |
| TUNE KQRvK M3 policy CE | 2.437 | 1.937 |
| final TRAIN policy loss, mean of the last 100 updates | not comparable (multi-budget + selector objective) | 0.854 / 0.830 / 0.867 |
| training wall per run | ~16 min | 94.4 / 91.8 / 93.6 min |
| training objective | root-policy CE + selector NLL over budgets {0,2,4,8} | root-policy CE only |

Per-seed run facts: all three ran fresh and uninterrupted to exactly 800 updates (`exit 0`, zero resumptions) at micro16 x
accum8; about 16,300 supplied states per update; at most 2559 states in one microbatch; peak VRAM 11761/11761/11761 MiB; GPU 91/93/93% busy on
average; tree building 44/31/30 s in total per run (negligible). The TUNE evaluation of each model was done exactly once, after
update 800.

The comparison is between a model that sees the whole tree and a model that sees none (B0). It says that this family *can*
exploit raw depth-2 future states on the hardest cell; it does not compare against any budgeted selection (that is Gate II/III,
not run).

## 4. P5.2: refined selector behaviour (re-evaluated selected P5 checkpoints)

`docs/evidence/v3/v3-p5.2-selector-diagnostics.json` (DERIVED / RE-EVALUATED; does not alter P5). The policy values reproduce
the committed P5 evidence exactly (max |diff| = 0.0), and the invariant "no proof-admissible edge exists after the proof is
complete" held for every query (0 violations). Queries are now split at proof completion. Entries `seed 5101 / 5102`.

| B | pre-completion queries | proof-admissible | refute-admissible | off-target | post-completion queries (filler) | positions never completing | KQRvK M3 pre-completion proof-adm. / off-target (5101, then 5102) | KQRvK M3 never completing |
|---|---|---|---|---|---|---|---|---|
| B2 | 7500 / 7500 | 0.671 / 0.679 | 0.110 / 0.115 | 0.218 / 0.206 | 1500 / 1500 | 3000/4500 / 3000/4500 | 0.373 / 0.447 / 0.388 / 0.419 | 750/750 / 750/750 |
| B4 | 12817 / 12812 | 0.571 / 0.584 | 0.129 / 0.127 | 0.300 / 0.288 | 5183 / 5188 | 2313/4500 / 2307/4500 | 0.305 / 0.521 / 0.327 / 0.491 | 750/750 / 750/750 |
| B8 | 21381 / 21312 | 0.390 / 0.404 | 0.089 / 0.086 | 0.521 / 0.510 | 14619 / 14688 | 2054/4500 / 2023/4500 | 0.210 / 0.690 / 0.218 / 0.682 | 722/750 / 711/750 |

What changed in the reading: the aggregate P5 off-target share at B8 (0.65) mixed in forced filler queries issued after the
proof was already complete (about 14.6k of 36k). Before completion the B8 off-target share is about 0.52, and the proof-
admissible share is about 0.40 (not 0.23). But on **KQRvK M3 the weakness is genuine**: before completion only about 21-22%
of B8 queries are proof-admissible and about 68-69% are off-target, and only 28-39 of 750 positions ever complete a proof.
Post-completion queries are never proof-admissible (0 in all cases), confirming the filler accounting.

First-completion distribution at B8 (pooled, seed 5101 / 5102): completion after 1 query for 1500 / 1500 positions (the M1
cells), after 3 queries for 683 / 688, after 5 for 214 / 222, with smaller counts at 4, 6, 7 and 8; completed after at most
8 queries: 2446 / 2477 of 4500.

## 5. P5.2: query-content ablation (evaluation-only; never part of any gate)

`query_content_ablation_v1` replays a recorded query path (from ACTIVE or from the TEACHER) with the state-derived neural
content removed: the queried child's pooled state, the state-derived descendant action embeddings, and the terminal/in-check
bits. **Preserved:** the replayed edges, the parent slot and root branch (the branch memory row and root candidate token the
planner updates), depth, parity and the remaining budget. **Not preserved:** the identity of replies deeper than the first
move (inside the model a reply is identified only through state-derived action embeddings). The normal path is unchanged
(the committed P5 policy still reproduces bit-for-bit with the ablation code compiled in), and every replay with normal content
reproduces its source run exactly (max |CE diff| = 0.0).

| seed | path | B | KQRvK M3 top-1: source = normal replay -> ablated | pooled top-1 normal -> ablated | pooled CE normal -> ablated | positions whose top-1 changed |
|---|---|---|---|---|---|---|
| 5101 | active | B2 | 0.491 -> 0.491 | 0.761 -> 0.761 | 1.574 -> 1.538 | 0 |
| 5101 | active | B4 | 0.491 -> 0.491 | 0.761 -> 0.761 | 1.614 -> 1.544 | 0 |
| 5101 | active | B8 | 0.491 -> 0.491 | 0.758 -> 0.760 | 1.565 -> 1.524 | 17 |
| 5101 | teacher | B2 | 0.996 -> 0.999 | 0.994 -> 0.994 | 0.962 -> 0.960 | 7 |
| 5101 | teacher | B4 | 0.995 -> 0.999 | 0.995 -> 0.994 | 0.960 -> 0.959 | 10 |
| 5101 | teacher | B8 | 0.992 -> 0.999 | 0.994 -> 0.994 | 0.956 -> 0.957 | 10 |
| 5102 | active | B2 | 0.524 -> 0.524 | 0.773 -> 0.773 | 1.550 -> 1.497 | 1 |
| 5102 | active | B4 | 0.524 -> 0.524 | 0.773 -> 0.772 | 1.568 -> 1.433 | 24 |
| 5102 | active | B8 | 0.525 -> 0.524 | 0.771 -> 0.771 | 1.522 -> 1.431 | 27 |
| 5102 | teacher | B2 | 0.995 -> 0.996 | 0.993 -> 0.989 | 0.946 -> 0.943 | 21 |
| 5102 | teacher | B4 | 0.991 -> 0.975 | 0.993 -> 0.978 | 0.947 -> 0.957 | 69 |
| 5102 | teacher | B8 | 0.980 -> 0.975 | 0.991 -> 0.975 | 0.943 -> 0.956 | 73 |

Findings:

- **ACTIVE paths:** removing the state content leaves KQRvK M3 top-1 unchanged (0.491 -> 0.491 and 0.524 -> 0.524 at every
  budget) and does not change pooled top-1, while pooled CE *improves* (for example 1.568 -> 1.433). The trained ACTIVE policy
  is not extracting useful information from the contents of its queried states; the content it receives is, if anything, a
  source of overconfidence (consistent with the P5 observation that queries sharpen the policy and worsen CE).
- **TEACHER paths:** removing the state content leaves the teacher-forced accuracy essentially intact (KQRvK M3 top-1 0.992 ->
  0.999, 0.980 -> 0.975 at B8; pooled top-1 0.994 -> 0.994 and 0.991 -> 0.975). The P5 teacher-forced result (top-1 about 0.99)
  therefore **does not depend on the queried state contents; it is explained by which edges the teacher chose to query**,
  which reveals the answer. This is query-pattern leakage in the sense the P6 plan feared. Caveat: the ablation preserves
  which root move was queried and its branch, which is exactly the channel through which the leakage works, so this
  diagnostic shows that content is unnecessary, not how much of the leakage is first-move identity versus deeper structure.
- Taken with Gate I, the information picture is now consistent: a model that is given the *contents* of many states but no
  answer signal reaches M3 top-1 0.69; a model that is given only a teacher's *path* reaches 0.98-1.00 without reading any
  content. The P5 teacher number is not evidence that queried contents are usable at that level.

## 6. What Gate I does and does not establish

Establishes (narrowly): for this model family, answer-free raw exact depth-2 future-state information is sufficient to improve
KQRvK M3 top-1 by at least +0.20 over the zero-query baseline on V3_TUNE_V1.

Does **not** establish: that ACTIVE learns selective search; that ACTIVE meets Gate II or beats FIXED (Gate III) at the required
margins; anything about CONFIRM or HOLDOUT_C; or that the effect is purely causal information (the models differ in architecture,
optimisation and training distribution). The ACTIVE model's budget-8 queries cover at most 8 of roughly 190 states of an
M3 tree, and P5.2 shows its queried contents are not being used productively, so the ACTIVE B8 result (+0.04 over B0 in P5) says
nothing against the value of information; it says the selection/integration is the open problem.

## 7. DAgger-style rescue: does the evidence now justify considering it? (observation only; not run)

Yes, it now justifies *considering* the one pre-registered rescue, with these observations stated plainly:

- Gate I passes: the missing ingredient for ACTIVE is not that raw state information is useless. Information is sufficient; the
  P5 ACTIVE shortfall is a selection/integration shortfall (pre-completion proof-admissible share 0.21-0.22 on M3, 28-39/750
  proofs completed at B8 against an ideal ceiling of 43%).
- The selector is trained only on the teacher's prefix distribution and never on its own state-visitation, which is exactly what
  scheduled sampling / DAgger-style correction addresses.
- Two cautions the rescue design must face: (1) the teacher's apparent success was query-pattern leakage, so ANY training that
  lets the policy read a teacher-chosen path will reproduce that confound; (2) ACTIVE does not currently use queried contents
  productively (ablation), so a better selector alone may not help unless the integrator learns to use contents. The rescue's
  success criteria should therefore be defined on Gate II/III-style TUNE comparisons, not on teacher-forced accuracy.
- Not authorised or implemented here. It needs an explicit owner decision.

## 8. Integrity and process notes

- Contract digests: P5 `105ac313...009d2` unchanged; P6 `4d95dda0...6b110`; selected P5 recipe `a069ba9d...db33` unchanged.
- The seed-5103 baseline replication (V3-D22) was trained and its TUNE evaluations were visible before the P6 plan was frozen;
  they are ACTIVE/B0 numbers and nothing about ALL-INFO depended on them (see plan section 12).
- Same-seed initial weights are NOT identical between ACTIVE and ALL-INFO (lazy order-dependent initialisation); paired seeds
  label the comparison only.
- Test-isolation finding: the backend RNG is process-global, so tests that seed it must not interleave; model-building tests
  now serialise on a lock (test-only change; production training is single-model per process).
- Short CPU-only test runs overlapped the first ALL-INFO run for a few minutes; the live pace was unchanged (7-8 s/update) and
  the GPU workload was undisturbed. No competing GPU or training processes ran at any time.
- HOLDOUT_C: never loaded, no custody entry changed, unevaluated.
- Throughput: the ALL-INFO workload is GPU-bound (about 91% busy on average, 7 s/update); the P5 workload was CPU-bound. A
  dedicated throughput / duplication / overhead pass is recorded as the next priority engineering task (not started); it must
  not change the experiment or its scope.

**DAGGER NOT RUN. P7 NOT RUN. HOLDOUT_C NOT EVALUATED.**

