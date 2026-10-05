# V6 proposal: supervised branch-local adversarial evidence backup

Status: **PROPOSED, NOT IMPLEMENTED, NOT TRAINED**. Owner acceptance of this
package does not authorize model execution. [Diagnosis](V6_DIAGNOSTICS.md) and
[next implementation ticket](V6_IMPLEMENTATION_PROMPT.md) define the next boundary.
New scientific identities/digests will be measured and preregistered after P0
implementation validation, before any pilot. No guessed hashes appear here.

## Principal mechanism and evidence

Recommend V6, not a V5.5 repair. Principal change: make each root candidate own an
explicit partial adversarial branch, and directly teach its paired evidence
correction whether that candidate belongs to the exact correct set. Replace global
E/H mixing with shared, turn-aware local backup. This tests **decision-aligned
evidence learning**, rather than adding recurrent iterations or width to V5.

V5's competent-base wrong-root correction range was usually too small to change
actions, its payload contrasts were tiny, and global integration had no task
benefit. Information coverage on those errors was also sparse. Accordingly,
acquisition becomes a fixed branch/reply policy, identical for every comparison.
This is an explicitly new reader-plus-information experiment. Improvement over
V5 would not, by itself, isolate the contribution of the acquisition change.
Within V6, comparisons hold acquired information fixed and test the integration
mechanism. No learned query selector is introduced.

## Input contract: no labels or unseen descendants

Identity: `v6_branch_evidence_interface_v1`. Every legal root action in authoritative
ActionId order owns a branch. Observed graph nodes carry root-relative board/rule
features, four returned-state tokens, parent index, owner action, path, depth,
attacker/defender turn, observed child edges, legal-child count, terminal status,
and observed/unqueried child counts. Terminal status and legal counts are derived
only by existing rules at an acquired state. The root board and legal actions are
available context. Branches with no acquired root edge have an explicit unknown
observation, not a zero-valued victory. Ownership/turn/masks are typed fields.

Forbidden deployed inputs: ProofTargets, correct indices, mate-depth class,
family/depth sampling labels, MateSolver values, unseen boards, final outcomes,
dataset position IDs and exclusion identities. IDs are operational provenance,
not embeddings. Family/depth are sampling/reporting metadata only. TRAIN labels
are loaded by the loss module after the input packet has been constructed; the
packet schema cannot serialize them. Changing target labels must leave packet
bytes and predictions unchanged. No engine/tablebase/learned label source.

All candidate branches are hard-separated. A candidate can read its own geometry,
frozen baseline candidate/root context and its own acquired nodes. It cannot read
another candidate's returned evidence or mutable hidden state. Final centering
across legal candidates remains the only shared correction operation. Shuffle
replaces complete four-slot returned payloads; owner, edges, turns, legal counts,
terminal structural features and all query accounting remain recipient structure.
Terminal features may themselves help; structure-only controls must measure this.

## Attacker/defender updates and unknown information

Retain V5's root-relative returned-state encoder geometry: width256, eight heads,
FFN768, two encoder blocks, four256-dimensional returned tokens. Weights are fresh
for V6. Frozen root context is projected into a256-dimensional owner anchor.
Pool the four slots with a learned fixed-slot linear projection. Let x_v be the
payload anchor, s_v a projection of observed structure, and h_v^0=x_v+s_v+owner_v.
Use the same encoder/interface for all three arms below.

At each local layer/iteration, compute a child score w·h_child and aggregate
children with softmax(score) on attacker turns and softmax(−score) on defender
turns (temperature1). Include a learned unknown-child token of the same turn role
when unqueried children remain; its log multiplicity is log(unqueried_count), so
one observed reply cannot masquerade as all replies. Nodes with zero observed
children retain the unknown token unless terminal. Terminal nodes retain their
observed state/terminal embedding; no solver target is inserted into h.

For terminal nodes set update=0. For other nodes, a shared MLP computes
u_v=MLP([h_v, pooled_child_v, x_v, s_v]); h_v←h_v+0.1·u_v. MLP dimensions are
1024→768→256, GELU. All nodes update simultaneously from the preceding iteration;
no completion-order dependence. Root-candidate virtual nodes use the attacker
role, their own acquired root edge or unknown token, and the same update rule.
No parent-state propagation or cross-branch attention is added in this version.
The branch head is256→256→1 with GELU, shared across candidates/iterations.

This is a learned attacker/defender **belief backup**, not an exact proof solver.
Softmax/min-like aggregation over unknowns cannot certify a win. An independent
three-valued partial-graph proof diagnostic stays outside the model and obeys
the conservative rules in V6_DIAGNOSTICS. A winning branch is not necessarily a
minimum-mate correct action; no incorrect root action is labelled “cannot win.”

Run paired factual/null streams with identical structure/weights. Null zeros the
returned payload anchor after the encoder; all graph structure stays fixed.
δ_a^r=head(h_F,a^r)−head(h_N,a^r); logits=B0+center_legal(δ^r). B0's complete
root/baseline parameters and weights remain immutable, including its context.
No clipping, B0-preserving gate, baseline gradient or selective correction bypass.
Both streams are differentiable through the reader. A fully null factual packet
must produce exact zero paired correction.

## TRAIN objective and credit boundary

Targets are the existing native V2 exact **minimum-mate correct-root action set**.
No new solver semantics or generated labels are needed. Final policy objective
is unchanged correct-set log loss. Add auxiliary weight0.5 times class-balanced
binary cross entropy on each raw paired branch score δ_a (before centering/B0).
For each root, average positive-action BCE and negative-action BCE separately,
then average available classes. If all actions are correct, use positive BCE only;
empty correct sets refuse. Average auxiliary loss over executed iterations.

Loss = final correct-set loss + 0.5·mean_iteration(branch BCE). This explicitly
teaches evidence corrections rather than relying on loss already minimized by B0.
All arms use the same labels, weight, class balance and iteration/layer averaging.
Unobserved branches remain in the target/loss with unknown inputs; performance
must be reported separately for observed/unobserved correct branches. Learning
board-pattern inference from nonterminal states is permitted; feeding labels
or treating missing replies as proved wins is not.

## Frozen acquisition: two Q8 policies

Contract `v6_fixed_branch_reply_acquisition_v1`, graph seed0x7A60_E001. Rank legal
root actions by immutable B0 logits, ActionId as tie-break; no target inspection.
One query returns one child state through the existing authoritative rules/path.
No duplicate path; Q≤8, Q=8 unless the eligible frontier is exhausted. Maximum
path depth16 remains a safety bound. Acquisition occurs once, before choosing
model arm/iteration; all arms consume byte-identical persisted packets.

Policies: `four_branch_reply_v1` first queries up to four ranked root edges;
`two_branch_reply_v1` first queries up to two. Thereafter cycle branches in their
ranked order. Within a branch choose the shallowest observed node with an
unqueried legal edge (depth, then path tie-break); order its unqueried actions by
SHA256(contract || NUL || seedLE || rootID || nodePath || ActionIdLE), then ActionId.
Thus the first subsequent round explicitly observes an opponent reply per
nonterminal branch; later rounds expand replies before deeper attacker states.
If a branch exhausts, skip it. If all initially selected branches exhaust while
budget remains, query the next ranked unobserved root edge and add its branch.
Stop when Q8, depth16 frontier exhausted, or all branches exhausted. Record all
skips, actual Q, depth, branch/reply coverage and legal-generation costs.

This does not promise complete opponent coverage within Q8. Legal counts and
unknown multiplicity preserve that uncertainty. Coverage by baseline-right/wrong
and exact-correct-branch presence is mandatory reporting, not label-based selection.

## Three predeclared arms: information, recurrence and compute

1. **Principal shared local backup**, R1 and R4 using the same weights/initial state.
2. **Nonrecurrent same-information scorer**: per candidate pool *all* its observed
   nodes in a single turn-separated attention read, each with full path/depth,
   payload, structure and unknown multiplicity. Concatenate attacker pool,
   defender pool, owner anchor, branch structure; use the same1024→768→256 MLP
   once and same branch head. No iterative hidden-state updates. It must see
   deep nodes immediately; it is not a straw comparator limited to depth1.
3. **Nonrecurrent compute control**: same full-information branch pooling as arm2,
   followed by four independently parameterized MLP layers, residual scale0.1,
   same head. No shared recurrence. It spends extra compute on a feedforward
   branch decision and uses the same auxiliary at each layer.

All arms share root weights, input bytes, encoder geometry, targets, seed policy,
root episode IDs, graph cache and total root examples. Encoder/reader weights
train separately; equal initialization is required where tensor shapes coincide.
Arm1's R1/R4 contrast isolates more iterations of one model with identical
information, but also increases compute. Arm2 tests whether evidence can help
without recurrence; arm3 tests an alternative use of comparable reader compute.
Neither identical Q nor equal parameter count alone establishes a fair comparison.

## Budgets and training (prospective)

FP32, RTX2050, physical microbatch2. Frozen B0 hash remains
`2d1c770a43a6455148b774e9ddb552b6ca33efd9fdd5d37593cefe7c8ae0bb00` and fingerprint
`12b272a941e5b29589195a65c779a60509626d2108975e4793674ad5867d75c9`.
Total parameter caps including frozen root: arm1/arm2≤8M, arm3≤12M. These are caps,
not fabricated measured counts. Device-wide peak≤3.2GiB; synchronized arm3 reader
forward FLOPs/latency must each fall within0.8–1.25× arm1R4 on resident Q8 packets.
Measure encoder, reader, acquisition and end-to-end costs separately. If matching
fails, STOP before pilot; do not rename it compute-matched or silently resize.

Each arm: seed6301, fresh reader/optimizer; 800 fixed updates, warmup80, peak3e−4,
existing AdamW contract and warmup/cosine schedule, batch36, accumulation18,
all27,000 TRAIN positions, nine-cell balanced sampling. Arm1 conditions are
two policies×R1/R4, nine roots per condition/update. Pre-generate the condition
episode list; arm2/3 receive the identical36-root episode bag and packets per
update, including policy/iteration-exposure duplicates. They never discard the
R1-labelled exposure or get more unique information. No other objective, LR
screen, extra updates, best checkpoint or performance-based sampling.

Bound each training/evaluation invocation to45 minutes with deterministic resume;
training cap2hours per arm, complete seed campaign cap6hours excluding qualification.
Graph/audit jobs≤2hours, shards deterministic/resumable. Budget breach is an
engineering stop, not authorization to reduce counts or change model. Qualified
shape checks precede execution; no CPU fallback for CUDA claims.

## Engineering gate: representative competent-base learnability

Before any DEV model execution, use a new disposable TRAIN-only panel, fixed by
ID hash `v6_competent_base_drill_select_v1`, seed0x7A60_D101. Four cells KQR/KRR
M2/M3: select12 frozen-B0-wrong and12 frozen-B0-right per cell, total96. Determine
strata once from immutable B0, never a new reader. Exact quotas required; if a
cell lacks support, STOP for owner amendment, no substitution. This enriches
errors deliberately for an engineering test and is not population accuracy.

Drill arm1 only, reader seed6300, Q8/R4 both policies,200 updates, batch24,
microbatch2, peak1e−3, warmup20, same composite loss and frozen B0. Record raw
policy set loss separately from auxiliary. At fixed endpoints require all:

- Correct custody/source/checkpoints/FP32 layout, finite loss/gradients, baseline
  tensors exact, exact normal/profile parity and null correction zero.
- Mean correct-set loss drops≥20% (unless initial<0.05); ≥12/48 wrong roots become
  correct and ≤2/48 initially right roots become wrong, schedule-averaged rates.
- Shuffle minus real set loss≥0.05 on the fixed panel; at least six newly corrected
  roots lose correctness under shuffle in both policies. No self/turn/cell donor
  violation. No label-selected donors; use frozen widening hierarchy.
- Report candidate best-correct versus best-incorrect margins, correction-to-B0
  margin ratio and per-group payload sensitivity at both endpoints. These are
  fixed measurements; action/sensitivity thresholds above are actual gates.

If any gate fails, STOP before DEV and owner review. No new panel, LR retry,
extra steps or looser threshold. Drill weights never initialize scientific runs.
Future parameter-gradient instrumentation must pass a named-leaf discovery test
before backward: missing head/encoder/local-block names refuse. Report complete
named finite gradients, factual/null contribution probes and baseline absent
gradients; nonzero gradients alone are not the learnability gate.

## Primary population and exact statistical unit

Existing DEV is development evidence already consulted in designing V6. Primary
population is **ONLY V5_HP_DEV_V2 KQRvK M3, n=750**, one paired observation per
position. Both policies must be present exactly once per condition. Average the
two within-position contrasts first; resample750 positions, not1500 policy rows.
All4500 DEV summaries and other cells are secondary. Intended executable contract:

```text
primary = filter(role==DEV && family==KQRvK && mate_depth==3)
require unique_ID_count==750 && exact_primary_ID_digest==measured_binding
require each_ID has both policies and all frozen treatments/arms/endpoints
contrast[id] = mean_policy(metric_treatment[id]-metric_control[id])
bootstrap(sorted_IDs, contrast, SplitMix64, 20000, ranks=[499,19499])
```

Binding digest is measured from the committed DEV artifact at P0; no guessed
digest. Reversing row order must preserve report bytes. Duplicate/missing/extra
ID, wrong population or averaging unit refuses. This fixes the V5 primary750
versus pooled4500 discrepancy prospectively, without rewriting V5.

## Scientific gates and controls

Train all arms to800 before any DEV inspection. Evaluate immutable0/800 only,
all six750-position cells, both policies. Normal/shuffle/null at R1/R4 arm1;
normal/shuffle/null arm2/3; no-child-backup and turn-blind backup at arm1R4;
structure-only payload-null, plus frozen A/B branch composition at R1/R4. Bootstrap
uses20,000 SplitMix64 resamples, ranks499/19499; seeds0x7A60_0101..0106 respectively
for the six contrasts below. Policy sign checks are separate, no pooled hiding.

At update800 on primary750, all gates are conjunctive:

| Gate | Prospective requirement |
|---|---|
| Content usefulness | Shuffle loss − real arm1R4 loss≥0.01; CI lower>0 |
| More shared integration | arm1R4 − arm1R1 top1≥0.03; CI lower>0 |
| Practical B0 improvement | arm1R4 − B0 top1≥0.03; CI lower>0 |
| Loop/content interaction | Real(R4−R1) − shuffle(R4−R1) top1; CI lower>0 |
| Nonrecurrent comparator | arm1R4 − arm2 top1≥0.03; CI lower>0 |
| Comparable extra compute | arm1R4 − arm3 top1≥0.01; CI lower>0 |

Neither policy may have a negative mean for content, loop or B0 contrasts. All
engineering, accounting and B0 integrity gates must pass. Confidence intervals
are development uncertainty summaries, not multiplicity-adjusted final-confirmation
claims. Gate conjunction is fixed; no best-policy choice. Report every arm's
content usefulness even if principal fails. If arm2 helps but recurrence gates
fail, conclude evidence reading useful, recurrent thesis unsupported; do not
reclassify principal as a pass. No controller authorization follows any result.

Report all cells/policies: top1, mass, set loss, uniform-target CE, entropy,
action changes, margins, raw/centered correction, KL/B0, transitions, unknown/
coverage strata, auxiliary score alignment, branch/turn attention or backup
weights, stream/update RMS, actual Q/depth/replies and synchronized compute.
Metrics are diagnostics unless explicitly gated above. Do not infer reasoning
from nonlinear composition alone. B0 must agree position-wise with accepted B0
under the existing exact execution contract before interpreting corrections.

Control contract `v6_branch_controls_v1`: shuffle seed0x7A60_E002, other-root
donors within the same family/root-mate cell and same attacker/defender turn.
Use exact acquired depth if available; otherwise only minimum absolute depth
distance among same-turn donors, stable donor ID/depth/path/storage order and
recipient hash selection. No same-turn donor means STOP. Record tier/pool/depth
delta; never inspect labels/predictions/loss to choose a donor. The complete
750-position cell supplies DEV donors; the frozen24-per-cell drill pool supplies
TRAIN donors. Terminal/structural fields remain recipient fields even when a
donor board differs: this is deliberately a payload-only intervention, and the
structure-only null contrast is reported separately.

No-child-backup replaces pooled_child at every iteration with the turn-specific
unknown token, retaining observed counts/structure and owner/payload anchor.
Turn-blind-backup uses positive-score softmax on both attacker and defender
nodes; all other turn metadata/unknown token roles stay unchanged. These isolate
child integration and adversarial pooling sign, respectively; neither is a
retrained variant. For composition, sort each root's acquired nodes by
SHA256(`v6_branch_composition_v1` || NUL || seed0x7A60_E003LE || nodePath), then
path; first ceil(n/2) nodes are group A, remainder B. Neither/A/B/both zero the
excluded payload anchors while retaining every structural field. Apply the same
groups at R1/R4; report interaction descriptively, no new composition gate.

Margin is best-correct logit minus best-incorrect logit; when all actions are
correct report that margin as absent, not infinity. Baseline top-two margin and
correction range are separate fields. Wrong/right strata use immutable B0 only.

## Replication, confirmation and stop rules

Seed6301 is a development selection pilot. Only if all gates pass may the owner
separately authorize exact6302/6303 reader replication; same frozen Stage A.
Require all three seed means positive for content, loop and B0 contrasts and all
three satisfy practical≥0.03 loop/B0 gains; publish every seed, no seed shopping.
This is reader-seed replication conditional on one baseline, not replication of
baseline training. CUDA trajectory variance makes this distinction necessary.

Only after design and replication are fixed may the owner authorize one sealed
CONFIRM campaign. Proposed population/averaging/counts/gates are the same primary
KQR M3 n750; other CONFIRM cells secondary. Evaluate all three preselected final
models once, average contrasts over seeds within position before the same paired
bootstrap; require content/loop/B0 and compute/comparator gates above. Freeze
confirmation receipt and seeds before unsealing. No feedback to training/design
after confirmation. This proposal grants no access now.

Any source/data/layout/hash mismatch, baseline mutation, nonfinite value, CUDA
failure, checkpoint/resume failure, unresolved shuffle donor or budget breach
stops execution and preserves artifacts. No automatic performance retry or
post-negative changes to seed, LR, objective, counts, policy, loops or gates.
P0 failure: engineering review. Learnability failure: owner review before science.
Pilot negative: close this preregistered V6 mechanism test; no within-pass rescue.

## Lineage and artifact contracts

Proposed identities: `v6_branch_supervised_backup_v1`, `v6_reader_recipe_v1`,
`v6_fixed_branch_reply_acquisition_v1`, `v6_primary_population_v1`,
`v6_evidence_packet_v1`, `v6_checkpoint_v1`, `v6_qualification_v1`,
`v6_competent_base_drill_v1`, `v6_reader_evaluation_v1`, `v6_pilot_report_v1`.
Each binds source SHA, exact config/parameter inventory, frozen B0/model/producer,
data-v2 role/content/ID/set digests, graph schema/content/structure/source, policy,
arm, seed, update, optimizer/sampler/counters and runtime contract as applicable.
Evaluation binds primary predicate/IDdigest/count/unit, controls/donor metadata,
paired policy coverage, report producer distinct from model producer, and raw
artifact hashes. No generic old-source bypass: exact frozen B0 import gets its
own phase capability, never authorizes old V5 training/resume.

Scientific hashes exclude clocks/paths/user/host; operational metadata stays
outside. Large weights/cache/records remain ignored runs/v6/; compact manifests,
qualification/drill/summary/pilot evidence are committed. Raw-byte custody is
required; a manifest is not a dataset. TRAIN/DEV roles refuse each other; CONFIRM
is custody-only until a future versioned authorization. Graph reuse is strict
source/config-bound. New acquisition retains authoritative action/history rules.

## Rejected alternatives and boundaries

- More V5 LR/updates/loops/capacity: no evidence for rescuing the frozen negative
  result; would evade the missing decision-aligned content mechanism.
- Unfreezing B0: confounds useful evidence correction with relearning competent
  root policy. Keep its exact weights.
- Terminal-only solver reader: Q8 rarely proves error branches; misses possible
  nonterminal-state learning and confuses winning with minimum-mate correctness.
- Dead-encoder/collapse repair: states and corrections are active; gradient
  utility is unmeasured. No narrow production defect supports V5.5.
- Learned controller/self-play: premature before fixed-information reading helps.
- Random-base memorization drill: not representative of incremental learning
  over Stage A. Replace with fixed wrong/right competent-base gates.

The next action is P0 implementation/qualification of this proposed mechanism
under a new owner prompt. **This pass implements and trains none of it.**
