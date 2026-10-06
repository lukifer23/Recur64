# Prospective objective/content-contrast probe

**PENDING OWNER REVIEW. NOT IMPLEMENTED, QUALIFIED OR RUN.** This document is a
proposal, not authorization. The completed P0 remains
BOTH_READERS_FAIL_LEARNABILITY; V5 remains CLOSED / NO_SIGNAL. No existing gate
is weakened or retrospectively replaced.

## One change and why

Use the existing **one-pass** reader only. Compare fresh, identically initialized
original-objective control against a content-differential auxiliary objective.
Architecture, competent frozen base, acquisition, packets, labels, policy loss,
episode bags and optimizer remain unchanged. This tests objective alignment,
not recurrence, acquisition redesign or held-out improvement.

The diagnosis found that board+flag removal preserves four of four comparator
helpful paired roots; broad's 192 additional acquired candidates have zero
minimum-mate-positive labels. Raw correction BCE permits acquisition/root-context
priors. Replacing that BCE's argument is a narrow, falsifiable intervention.
Neither lack of observed support nor root-history mismatch shortcuts disappears
by definition; they are reported and tested below.

Rejected for this next probe: more queries/policy selection (changes information
and leaves the demonstrated shortcut untested); more recurrence/capacity (neither
failed arm earned it); adjusting LR/updates (no diagnosed optimizer defect);
retraining the frozen root (destroys attribution); a generic carrier model
redesign (the arbitrary carrier control was distribution-mismatched). If this
objective probe fails, do not automatically try these alternatives.

## Exact objective

For root i, legal action a and original observed packet G, let
`delta_G(a) = F_G(a) - N_G(a)` be the existing uncentered paired correction.
Let S(G) be the frozen original full-payload donor shuffle, seed `0x7A60_E002`.
The recipient structure and root-candidate context are identical. Its null stream
is identical to G's null stream. Therefore

```
d(a) = delta_G(a) - delta_S(G)(a) = F_G(a) - F_S(G)(a).
P = acquired branches intersect minimum-mate correct-action set
N = acquired branches minus minimum-mate correct-action set
A(z) = mean of available class means:
       mean_{a in P} BCEWithLogits(z(a), 1), when P nonempty;
       mean_{a in N} BCEWithLogits(z(a), 0), when N nonempty.
       Zero when both are empty.
control:   L = CorrectSetLoss(B0 + centered(delta_G)) + 0.5 * A(delta_G)
treatment: L = CorrectSetLoss(B0 + centered(delta_G)) + 0.5 * A(d)
```

This is the ONLY training change. Both factual and shuffled terms of d receive
gradients through the same reader weights; no detach that silently changes the
objective. Null cancellation must be tested numerically and analytically. Do not
add a shuffled policy loss, extra margin loss, supervised internal proof score,
auxiliary labels for unqueried branches or teacher descendants. P/N are
TRAIN-only root minimum-mate membership, not generic winning/losing labels.
Foreign payload is a counterfactual tensor control, not a proof about the root.

All legal actions remain in real policy loss. Eligible-class support is reported
per root/policy and cell. No eligible class is invented; acquisition never reads
labels. Targets exist only in objective/report alignment, never packet features.
Original all-null remains exact B0. No changes to paired correction/centering.

## Frozen inputs, panel and randomization

Use exactly the existing 96-root panel: KQR/KRR M2/M3, twelve B0-wrong and twelve
B0-right per cell. Reuse its committed ID/stratum and packet/episode receipts,
whose complete scientific plan digest is
`e7e09f8d853dc69f983c33e61e0186205fdfee21fc650edaafdcc1b7c1c5dee9`.
Never reselect support or replace a root. These roots have been repeatedly
exposed; every conclusion is learnability/memorization, not generalization.

TRAIN is V5_HP_TRAIN_V2, ProofTargets
`d7918b7a138b9aab17de24842dd5bf511cb0b95aeccbcbf07cb48921a81d86b9`.
Packets are the same 192 Q8 packets, policies ExploitTwo and broad four-root.
Verify every packet digest and report observed correct-branch support, unknown
counts and terminal evidence before fitting. No new acquisition or unseen replies.
Existing failed P0 checkpoints may not initialize either arm.

Exact Stage A import: producer `d11659eca0774e0064bed0ef64ead2b725886d93`,
update1200, model `2d1c770a43a6455148b774e9ddb552b6ca33efd9fdd5d37593cefe7c8ae0bb00`,
optimizer `cbef56e557f71a9205e34f65c782d9264cabd8fe960e5ab3915790a5f368f03d`,
recipe `6642579e1f2472bda955ca7ada5bb3b8a435634c023b665684da1e4676347e70`.
Root fingerprint `12b272a941e5b29589195a65c779a60509626d2108975e4793674ad5867d75c9`.
Normal complete-file integrity loader and exact predecessor capability run first.
No generic old-source permission and no V5 training/resume permission.

Seed6300 fresh reader initialization; save one canonical initial state, reload
independent exact replicas and independent fresh AdamW states. Shared parameter
bits must match. Deterministic episodes remain the existing formula: for update
u=0..199 and microbatch k=0..11, cell=k%4, local=(3u+floor(k/4))%12,
indices24cell+local and24cell+12+local, policy=(k+u+floor(u/4))%2.
Each root has25 exposures/policy, twelve examples/policy/update.

Freeze the original donor mappings before fitting: different root, same family
and root mate-depth cell, same root_to_move; exact acquired depth if possible,
else only minimum absolute-distance same-turn depths; original stable keys,
seed and recipient selection hash. Unresolved donors STOP. No correct-label,
prediction or loss-based mapping choice.

## Fixed controls: distinguish returned successor content from root matching

At both endpoints0/200 evaluate real, original full shuffle, all-null,
board+flags zero and successor-frames zero, plus the two following fixed
**successor-only shuffles**, seeds `0x7A60_E402` and `0x7A60_E502`.
Use original donor constraints/order/hash with the specified seed. For recipient
node depth dR and donor depth dD, replace only ObservationV1 history frames k<dR:
copy donor frame k when k<dD, otherwise use a zero14-channel frame including
validity=0. Retain recipient frames k>=dR (root/pre-root), observation features
112..118, flags and terminal stopping. Retain all ownership/turn/action/path,
depth, accounting, legal/observed/unknown counts, masks and topology. These are
tensor controls, not necessarily legal chess histories. No new donor search if
a mapping performs badly; freeze hashes before execution.

Why two maps: full shuffle can expose foreign root histories and reward
root-consistency detection. Successor-only controls leave root histories factual,
testing whether improvements depend on future-frame content beyond that cue.
They do not remove recipient flags or count-derived evidence. Distribution
shift and incomplete reply coverage still prevent a semantic-proof claim.
Report flag/terminal support explicitly; do not call these content-free boards.

## Parameters, compute and resource budget

Same one-pass architecture/config
`f48dcf28ce178581a99099d088fa4c975ab684c1c81c351896238a7ffc4b8335`:
6,703,152 total parameters including3,677,728 frozen root, no new tensors.
Verify inventory rather than copy this count blindly. FP32 RTX2050, physical2,
effective24/accumulation12, 200updates, warmup20, peakLR1e-3, original cosine.
AdamW beta1=.9, beta2=.999, epsilon=1e-5, weight_decay=1e-4,
gradient clip norm1.0. No LR screen or post-negative changes.

Each arm consumes4,800 root episodes. Treatment additionally computes4,800
shuffled episode views with gradient; reuse matched null computation only if
equations/gradients and profiling parity prove equivalence. No hidden views or
free compute. Report encoder, reader, backward, acquisition, end-to-end latency,
view counts and synchronized timings separately. Same information/parameters
does NOT make this compute matched. No recurrent superiority claim.

Device-wide peak <=3.2GiB, total parameters <=8M. Invocation <=45minutes,
deterministic full-state continuation, aggregate fit wall <=2hours/arm (<=4hours
pair). Endpoints and custody invocations <=30minutes each. A deadline reached
before200 is a recorded budget failure; no extra allocation or partial-result
model selection. If treatment cannot fit under this budget, STOP before fitting
and return measured resident-cost evidence; do not reduce batch/counts.

## Prospective engineering prerequisites

Commit contract/schemas/tests before fitting and bind one final scientific source.
Scoped fmt/Clippy, CLI refusal tests, focused/full release tests and serial pinned
CUDA build. Fresh CPU and actual RTX2050 FP32 MB2 qualification: original model
inventory, all-null/B0 exactness, absent root gradients, complete reader gradients,
independent replica/clone purity, exact normal/profile forward/gradient/AdamW
parameter+moment parity, fifty resident updates, full model/moment resume and
exact continuation. Numeric loss/gradient references cover available-class
averaging, empty classes, BCE at zero, d's two signed gradient paths and null
cancellation. Do not require nonexistent head bias.

Targets mutated/IDs relabeled must leave packets/predictions unchanged. Verify
intervention structure/channel preservation and exact original-objective
instrumentation agreement. Reverify TRAIN custody and DEV/CONFIRM custody-only,
zero overlaps and sealed/evaluated=false. No model on DEV/CONFIRM. Every native
exit and failed log persists. Frozen root digest must remain exact every resume.

## Frozen measurements and decision rules

Complete both fixed endpoints before comparing performance. Correctness uses
legal ActionId-order argmax with first-index tie break. Count roots, not policy
rows: corrected means initially B0-wrong and correct under BOTH policies; harmed
means initially B0-right and wrong under EITHER. Loss contrasts average two
policy values inside each root, then all96 roots with equal weight. Report
per-policy counts, every helpful/harmful ID, four cells and support strata.

Treatment must meet every original P0 requirement prospectively: finite training,
exact frozen B0/null; >=20% real policy set-loss reduction unless start<.05;
>=12/48 corrected; <=2/48 harmed; original-shuffle minus real mean set loss>=.05;
>=6 real-corrected roots lose correctness under original shuffle BOTH policies.

Additional prospective criteria: for EACH fixed successor-only seed, mean
shuffle-minus-real loss>=.05 and >=6 real-corrected roots lose correctness under
BOTH policies. To support this objective over its fresh matched control, require
treatment real mean set loss at least.05 lower, >=4 additional paired corrected
roots, and no more harmed roots (also <=2 absolute). Freeze all criteria together,
never retrospectively replace original failed P0 gates. No bootstrap significance
claim on this deliberately stratified, repeatedly used memorization panel; report
exact paired root deltas. Later held-out/bootstrap/replication design needs owner
authorization and a new prospective contract.

Report policy and auxiliary losses separately at0/200, margins, F/N/raw/centered
directions, support and correction range, training windows and exposure. A
performance failure does not stop the other fixed endpoint or invite retry.
Operational failure stops remaining execution. All success here means only
disposable TRAIN content-sensitive learnability, never held-out strength.

If all criteria pass, propose a separately authorized replication/development
experiment. If both objectives pass without a comparative advantage, report
learnability with no objective attribution. If treatment fails, close this probe:
no retuning, extra updates, alternative seeds/panels/acquisition, or automatic
model redesign. Independent CUDA trajectory variance limits single-pair causal
confidence. No outcome automatically licenses DEV,800updates or CONFIRM.

## Complete next implementation-agent prompt — pending owner approval

Execute ONLY after the owner explicitly authorizes this proposal. Work in
`C:\Users\Caitl\Desktop\Code Projects\Recur64-v6` on
`experiment/hp-v6-branch-backup`; fetch/inspect HEAD, worktrees, tracking, active
processes, AGENTS.md and dirty files. Base on the publication containing this
document; inspect every later commit and preserve unrelated edits. Do not edit
main or V5. Read V6_P0_FAILURE_DIAGNOSIS.md, this complete prospective contract,
original revised P0/results/receipts and diagnostic evidence. V5 remains
NO_SIGNAL, original V6 P0 remains failed and immutable.

Implement only original one-pass control and the above content-differential
auxiliary treatment, with separate versioned objective identity
`v6_content_differential_aux_v1` and launch plan
`v6_objective_content_probe_v1`. Keep production model equations/acquisition
unchanged; preserve exact Stage A import. Do not implement recurrence, controller
or future modules. Stage A/B/P0 disposable weights are never reused as new reader
initialization. Bind artifacts to the exact measured source/config/data and
input-plan digests, fixed panel/packet/donor maps and episode schedule.

Freeze implementation and prospective schemas before execution. Suggested
receipts: `v6_objective_probe_plan_v1`, `v6_objective_probe_endpoint_v1`,
`v6_objective_probe_decision_v1`. Include predecessor hashes, fresh init/optimizer
identity, objective formula/version, packet/control hashes, native exits, times,
memory, update/exposure and all original/additional root gates. Raw weights/
packets/matrices stay ignored. Do not reuse old P0 endpoint paths.

Run the engineering prerequisites above; any failure STOP and preserve evidence.
No CPU substitution for CUDA. Do not change toolkit/drivers/framework/precision.
After all pass, preregister both native invocations together under new clean
`runs/v6-objective-probe/seed-6300/{control,treatment}` directories. Never erase
unexpected contents. Execute control then treatment in bounded chunks with
validated continuation,200updates exactly, both endpoints before interpretation.
Run only specified TRAIN controls. Do not select checkpoint/mappings by metrics.

Publish exact source, inventory, custody, qualification and resident costs;
both endpoints, original/additional root gates, per-policy/cell/support metrics,
helpful/harmful IDs, training windows, frozen root proof and paired comparison.
Separate MEASURED/INFERRED/NOT RUN. Commit/push reviewed concrete checkpoints on
V6 branch only. STOP for owner review regardless of result. No DEV or CONFIRM
model invocation, scientific campaign, new seeds, controller, self-play or V5
rescue. CONFIRM remains sealed and unevaluated. This prompt is NOT executed in
the current diagnostic task.
