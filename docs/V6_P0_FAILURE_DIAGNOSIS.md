# V6 P0 frozen failure diagnosis

**Decision: propose one objective/content-contrast experiment, pending owner review.
Do not launch it.** Original P0 remains BOTH_READERS_FAIL_LEARNABILITY, execution
valid. V5 stays CLOSED / NO_SIGNAL. No weights were trained or retuned here.

## Verified sources and independent reproduction — MEASURED

Starting publication b3a2f6f089dd1427119bc7d49537f0def1b7b0bc matched local/remote;
worktree clean, no active scientific process. Producer source:
42f47b852451d59645dcc50120e4a426b034f493. Diagnostic source:
d34c9484b70800c3cfef86618172a5aecc42653d. Only two Rust example files changed;
production equations, losses, acquisition, training and original reports did not.
Consumer source is distinct from the frozen weight producer.

Config f48dcf28ce178581a99099d088fa4c975ab684c1c81c351896238a7ffc4b8335;
plan e7e09f8d853dc69f983c33e61e0186205fdfee21fc650edaafdcc1b7c1c5dee9;
TRAIN ProofTargets d7918b7a138b9aab17de24842dd5bf511cb0b95aeccbcbf07cb48921a81d86b9.
All available published model, final optimizer, metadata and endpoint hashes
matched. Update0 has no historical state.json; its optimizer bytes were
independently inventoried, not falsely compared to nonexistent metadata.
[Exact artifact identities](evidence/v6-p0-diagnostics/artifact-verification.json).

All four original endpoints at0/200 in both arms reproduced every serialized
field exactly on actual RTX2050 CUDA FP32 physical microbatch2, native exit0.
No CPU substitution. Native reproduction times: principal24.128/18.665s,
comparator16.120/16.518s. Independent root gates reproduce principal12 corrected,
5 harmed,0 paired shuffle reversals; comparator4,4,0. Correction is BOTH policies;
harm is EITHER. Policy rows are not independent roots.
[Reproduction and custody receipts](evidence/v6-p0-diagnostics/reproduction.json).

Scoped formatting/Clippy with warnings denied passed; three focused release
tests passed, including preservation of intended channels and production/mirror
equality. Serial pinned CUDA build passed. Historical P0 full workspace614 tests
and CPU/CUDA training qualifications were inspected, not rerun or relabeled as
fresh. Actual CUDA diagnostic forward reproduction/parity is newly measured.

## Frozen intervention execution — MEASURED

The manifest preceded controls. The history amendment preceded all new model
controls and was justified by ObservationV1 source inspection: the returned
tensor contains8 history frames, with the root at frame acquired depth.
Both arms0/200, unchanged96 roots and192 packets,15 conditions:11,520
root-policy-condition records,4,992 exact production-forward parity batches.
Candidate ActionIds, labels for offline alignment only, baseline/final logits,
factual/null/raw/centered scores, iteration corrections, selected actions, support,
flags, depths, legal/observed/unknown replies and donor mappings were serialized.
Ties select the first legal ActionId-order index at the maximum FP32 logit.

All8 invocations exited0; longest52.472s, below30minutes. Inference graph-free;
optimizer steps0. No DEV/CONFIRM model invocation. Full original all-null and
turn-blind controls were retained. Three additional fixed donor seeds were used
without outcome selection. The mirror was checked against production outputs for
every applicable condition; carrier overrides are separate post-encoder interventions.

Raw candidate-level matrices remain ignored under
runs/v6-p0/failure-diagnostics-d34c948/{arm}-{000,200}/diagnostics.json.
Committed summaries bind their hashes. They include cell/policy/B0 strata, support,
terminal and defender-coverage strata, individual helpful/harmful root masks.
[Full definitions](V6_P0_DIAGNOSTIC_PLAN.md),
[manifest](evidence/v6-p0-diagnostics/intervention-manifest.json),
[validation](evidence/v6-p0-diagnostics/validation.json).

## Payload robustness and exceptions — MEASURED

Counts below use192 root-policy rows for correctness/loss and ROOTS for survival
of previously helpful corrections under BOTH policies.

| Condition at200 | Principal correct rows / loss / helpful roots surviving | One-pass correct rows / loss / helpful roots surviving |
|---|---|---|
| Real | 121 /1.067126 /12 | 111 /1.188632 /4 |
| Original shuffle E002 | 121 /1.073096 /12 | 109 /1.192330 /4 |
| Fixed shuffle E102 | 120 /1.073739 /11 | 108 /1.191792 /3 |
| Fixed shuffle E202 | 121 /1.069792 /11 | 108 /1.191558 /3 |
| Fixed shuffle E302 | 123 /1.071942 /12 | 111 /1.192131 /4 |
| Observation+flags zero | 122 /1.090823 /11 | 109 /1.194349 /4 |
| Flags zero only | 122 /1.070763 /12 | 109 /1.190926 /4 |
| Successor frames zero, root history retained | 121 /1.067075 /12 | 110 /1.230784 /4 |
| Root history zero, successor frames retained | 123 /1.087736 /11 | 111 /1.192812 /4 |
| Fixed alternating0.1 carrier, flags zero | 96 /1.256288 /0 | 96 /1.255771 /0 |
| Root-candidate context zero | 98 /1.251327 /0 | 96 /1.255272 /0 |
| Turn blind | 92 /1.863524 /1 | 72 /1.998776 /1 |
| Original all-null | 96 /1.258191 /0 | 96 /1.258191 /0 |

At update0 neither reader corrects or harms any paired root; original conditions
retain96 correct rows, and initial corrections do not alter baseline correctness.
Complete update0 controls are retained separately, not used for tuning.

Principal: all four shuffles lose ZERO helpful roots under BOTH policies. E102
and E202 each lose one benefit under one policy. Comparator: E102 loses one
helpful root under both policies; other mappings lose at most one under either.
Original harmful roots survive all four shuffles:5 principal,4 comparator.
Board+flag removal retains4/5 principal harms and4/4 comparator harms.
Per-root IDs/actions/control masks:
[principal](evidence/v6-p0-diagnostics/principal-root-transitions.json),
[comparator](evidence/v6-p0-diagnostics/one-pass-root-transitions.json).

The principal changes ZERO selected actions when successor frames are removed.
The comparator changes24 actions and its loss rises0.042152, while all4 original
helpful paired roots survive. Thus there is some forward content sensitivity;
it is not demonstrated necessity for most useful corrections. Partial-frame
masking is a distribution shift, so that loss increase is not proof of semantic use.

## Learned generic carrier and root/structure effects

**MEASURED:** when all119 observation channels and all9 flags are zero, valid
encoder slots have identical inputs, and their post-slot anchor RMS is identical
across all measured valid nodes:1.630414829 principal,1.737700757 comparator.
Learned biases, square embeddings and slot queries remain. These content-free
encoder outputs preserve11/12 and4/4 helpful paired roots. Full-vector equality
was not separately serialized; identical inputs/source and identical measured RMS
support the constant-anchor interpretation.

The predetermined alternating0.1 carrier has RMS0.100000001 and preserves none.
It differs in magnitude AND direction from learned encoder outputs. Its collapse
is not evidence that board semantics are necessary, nor that any nonzero vector
is sufficient. Carrier-with-factual-terminal stopping produces the same aggregate
result here. Current rule fields and board-derived counts remain explicitly
retained in partial-frame controls; board+flags zero removes all observation
rule/history channels, but recipient structural counts still remain.

**MEASURED:** removing root-candidate context abolishes all original helpful roots.
Turn-blind intervention is destructive too. **INFERRED:** together with positive
sufficiency of observation+flag removal, this supports root/structure-conditioned
correction gated by a particular learned nonzero anchor. Destructive controls
alone would not establish this, and their distribution shifts remain limitations.
This is how most current corrections can operate, not a causal reconstruction
of which gradients produced the training trajectory.

## Why broad corrections are large — MEASURED / INFERRED

Principal narrow/broad mean correction ranges:1.375098/13.146230;
one-pass0.938942/9.975854. Broad adds192 queried candidates outside B0's top2;
NONE is minimum-mate-correct on this fixed panel. These are not necessarily losing
actions: labels identify the minimum-mate correct set, not generic winning status.

| Arm / acquired candidates | Count / correct labels | Mean factual / null / raw score |
|---|---|---|
| Principal narrow top2 | 192 /91 | 0.170751 /-0.001079 /0.171830 |
| Principal broad top2 | 192 /91 | 0.169588 /0.995361 /-0.825773 |
| Principal broad rank3+ | 192 /0 | 8.085406 /18.435611 /-10.350204 |
| Comparator narrow top2 | 192 /91 | -1.163604 /-1.398677 /0.235073 |
| Comparator broad top2 | 192 /91 | -1.163652 /-0.816467 /-0.347185 |
| Comparator broad rank3+ | 192 /0 | -0.453521 /6.574415 /-7.027936 |

Principal broad rank3+ raw scores are negative96.35% of the time; comparator94.79%.
Null candidate scores are EXACTLY unchanged by board/flag/history removal and
all four shuffles. Those treatments change factual scores. Larger raw range is
therefore not a larger useful-positive readout: it chiefly accompanies strong
negative differences on the extra acquired, exclusively negative-label candidates.
Owner context and acquisition structure affect both factual/null scoring. Acquisition
rank, topology, depth and support co-vary; this observational comparison does not
isolate one structural feature as a cause.

**INFERRED:** available-class auxiliary BCE rewards large negative corrections
on these additional candidates without requiring specificity to their returned
boards. A root/structure prior can satisfy much of that pressure. There is no
real-minus-shuffle constraint in the original objective. Policy and auxiliary
improvement thus need not certify useful content dependence.
[Candidate-direction/null audit](evidence/v6-p0-diagnostics/candidate-direction-audit.json).

## Support and unsupported correction — MEASURED

Of48 B0-wrong roots,25 have an observed correct branch and23 do not, under BOTH
policies. Broad supplies no additional positive branch on this panel. Defender
replies remain partial; total terminal evidence9 nodes across8 roots. Support,
terminal and reply-coverage breakdowns are committed; no complete coverage or
partial-proof certificate was invented.

Each arm corrects one paired root with NO observed correct branch: KRR M3 ID
`V5_HP_TRAIN_V2-KRRvK-m3-....................R...........R..........K..........k.........`.
Selected ActionId6510 has raw delta0. Principal suppresses B0 ActionId10430 by
-1.869919 narrow /-2.671676 broad; comparator by-1.951802/-2.525273. Centering gives
the unobserved selected action a relative positive shift. This is suppression of
queried alternatives, not assessment or proof of that unseen branch. A second
KQR M2 corrected root selects unobserved ActionId9405 under broad despite another
correct branch being observed; the same distinction applies.

No observed-correct-support rows have substantially higher real loss: principal
narrow/broad2.858575/2.497472 versus0.513357/0.605808 with support; comparator
3.052224/2.790276 versus0.626249/0.659228. This association is confounded with B0
error difficulty, not a causal theorem that more queries would fix learning.
[Fixed support](evidence/v6-p0-diagnostics/fixed96-support.json),
[principal control strata](evidence/v6-p0-diagnostics/principal-200-control-strata.json),
[comparator control strata](evidence/v6-p0-diagnostics/one-pass-200-control-strata.json).

## Ranked explanations and decision

1. **Strongest supported:** learned generic returned anchor enables mostly
   root/structure corrections. Evidence: observation+flag removal preserves most
   helpful roots and harms; principal successor removal changes no actions;
   repeated shuffles scarcely affect gains; null scores remain exact. Limitation:
   some exceptions and comparator content sensitivity remain; arbitrary carrier
   failure prevents a claim that all nonzero carriers work.
2. **Objective alignment shortcut:** raw correction BCE admits this solution,
   especially192/192 negative-label broad exploratory candidates. This is an
   inference consistent with measured candidate directions, not a proven gradient
   history. A prospective content-differential objective tests whether changing
   this pressure earns useful successor-specific correction. Failure would weaken
   its adequacy as the next remedy, not prove the original shortcut absent.
3. **Acquisition support limitation:**23/48 errors lack an observed correct branch;
   broad adds no positives and incomplete replies cannot certify min-mate actions.
   It remains a real bottleneck. Against it as the sole cause: removing all boards
   preserves most improvements even where positive support exists. Whether observed
   nonterminal boards contain sufficiently learnable signals remains unresolved.
4. **Harness defect:** none demonstrated. Exact original reproduction, source
   scope, structure tests and CUDA trace parity argue against it. No architecture
   rescue or assertion relaxation is justified.

Recommend a single objective/content-contrast probe with unchanged acquisition,
interface and frozen baseline, using matched one-pass control/treatment. It directly
tests the missing constraint without hiding support limits or claiming recurrence.
The next prospective contract includes a successor-specific validation control to
reject simple root-history mismatch detection. [Complete pending prompt and preregistration](V6_P0_NEXT_OBJECTIVE_EXPERIMENT.md).
NOT IMPLEMENTED OR RUN. Original failed gates remain unchanged.

## Preservation and NOT RUN

Post-run independently rehashed131 original P0 files and614 historical V5 files:
zero mismatches. All TRAIN/DEV/CONFIRM raw bytes match committed raw SHA bindings;
full Rust role/custody/disjointness checks passed each invocation. CONFIRM sealed,
evaluated=false. V5 worktree untouched. No new DEV model invocation.

No training, optimizer steps, extra updates, new training seeds, checkpoint selection,
800-update campaign, controller, self-play, V5 rescue, or sealed-set model use.
Current diagnostic weights remain inspection-only; never initialize future science.
