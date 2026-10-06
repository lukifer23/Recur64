# V6 successor-content differential objective probe — contract v2

**Frozen before implementation, qualification and any fitting.** Objective identity
`v6_content_differential_aux_v2_successor_only`; launch plan
`v6_objective_content_probe_v2`; receipts `v6_objective_probe_plan_v2`,
`v6_objective_probe_endpoint_v2`, `v6_objective_probe_decision_v2`.

This is a TRAIN-only, disposable, repeated-exposure learnability/memorization
probe. V5 remains CLOSED / NO_SIGNAL. Original V6 P0 remains
BOTH_READERS_FAIL_LEARNABILITY; its reports, gates, data and checkpoints are
unchanged. Frozen diagnostic weights are inspection-only. No DEV/CONFIRM model
invocation occurs; CONFIRM stays sealed and unevaluated.

## Recorded amendment to the historical proposal

[`V6_P0_NEXT_OBJECTIVE_EXPERIMENT.md`](V6_P0_NEXT_OBJECTIVE_EXPERIMENT.md)
(identity `v6_content_differential_aux_v1`, plan `v6_objective_content_probe_v1`)
is preserved as a historical proposal and is **not** the executed contract. It
defined the training contrast `d` with the *complete-payload* donor shuffle S(G).
The owner ticket replaced that with a **successor-only** shuffle. The two are not
interchangeable: a complete-payload contrast can be satisfied by detecting a
foreign root history (root/action consistency), while a successor-only contrast
leaves the recipient's own root/pre-root frames factual. This v2 contract
therefore carries new identities, and the complete-payload shuffle is retained
**only as an evaluation control**. Nothing here reinterprets v1 text.

## Question

Can a successor-content differential auxiliary objective produce useful,
content-sensitive corrections over competent frozen B0, compared with the
original objective? Only the existing one-pass reader is used. Architecture,
acquisition, packets, labels, real policy loss, optimizer, episodes and training
budget are unchanged.

## Successor-only intervention Ssucc(G)

For each acquired node with acquired depth `dR` (ObservationV1: 64 squares ×
119 channels; channels `frame*14 + c`, frames 0..7, then features 112..118):

* replace only history frames `k < min(dR, 8)`;
* donor frame `k` is copied if `k < dD` (donor acquired depth), else a zero
  14-channel frame (validity channel = 0);
* retain recipient frames `k >= dR` (root / pre-root), observation features
  112..118, all nine flags, terminal stopping, ownership, incoming action
  geometry, native path, turn, depth, legal/observed/unknown counts, topology,
  masks and query accounting. Only `payload.observation` frames change.

Donor mapping (frozen, label-free, before training; seed `0x7A60_E002`): original
rule — different root, same family/root-mate-depth cell, same policy, same
`root_to_move`; exact acquired depth if available else only minimum
absolute-depth-distance same-turn nodes; candidates sorted by
`(donor id, depth, path, storage id)`; recipient pick by the original
`SHA256(v6_ranked_two_hash_two_reply_v1, seed, root id, path, depth)` hash. The
mapping reads no correctness, prediction or loss. Unresolved donor ⇒ STOP.
Failed channel-preservation verification ⇒ STOP.

These are tensor controls, not guaranteed legal chess histories. Root/action
consistency detection remains a competing explanation. Passing this experiment
is **not** proof of semantic reasoning.

## Exact objectives

`delta_G(a) = F_G(a) − N_G(a)` (existing uncentered correction). With Ssucc
recipient structure and null input identical, `d(a) = delta_G(a) − delta_Ssucc(G)(a)`
(= `F_G − F_Ssucc(G)`). `d` is computed literally as the difference of two full
reader passes with the same weights; null cancellation is verified numerically
and analytically, not assumed.

`A(z)` = existing available-class auxiliary BCE: eligible observed
minimum-mate-correct candidates target 1; eligible observed candidates outside
that set target 0; class means averaged over *available* classes; empty support
contributes zero; unqueried branches are never supervised.

* Control: `L = CorrectSetLoss(B0 + centered(delta_G)) + 0.5 · A(delta_G)`
* Treatment: `L = CorrectSetLoss(B0 + centered(delta_G)) + 0.5 · A(d)`

Both factual and intervened terms receive gradients through the same reader
weights (no detach). No other loss, regularizer, architecture or acquisition
change. Minimum-mate membership is not generic winning/losing status; donor
payloads are not proof labels.

## Frozen run design

Existing 96-root panel (KQR/KRR M2/M3; 12 B0-wrong + 12 B0-right per cell),
192 Q8 packets (ExploitTwo, BroadRankedHash), plan digest
`e7e09f8d853dc69f983c33e61e0186205fdfee21fc650edaafdcc1b7c1c5dee9`, raw plan
SHA `3be158eb…7a49`, TRAIN ProofTargets `d7918b7a…6b9`. No root substitution,
no support-based reselection. Deterministic episodes: update `u∈[0,200)`,
microbatch `k∈[0,12)`: `cell=k%4`, `local=(3u+⌊k/4⌋)%12`, indices
`24·cell+local`, `24·cell+12+local`, `policy=(k+u+⌊u/4⌋)%2` (4,800 root
episodes/arm; 25 exposures/root/policy).

Fresh seed-6300 one-pass reader: ONE canonical initial state saved once; control
and treatment load independent bit-identical copies with independent fresh AdamW
states. Never initialized from P0/diagnostic weights. Exact Stage A root imported
through the existing integrity/predecessor loader (producer
`d11659eca0774e0064bed0ef64ead2b725886d93`, update 1200). 6,703,152 total
parameters (3,677,728 frozen root). Actual RTX 2050 CUDA FP32, physical
microbatch 2, effective batch 24, accumulation 12, exactly 200 updates, peak LR
1e-3, warmup 20, cosine, AdamW β=(.9,.999), ε=1e-5, wd=1e-4, clip 1.0.

Resource limits (preserved): ≤8M parameters; device-wide peak ≤3.2 GiB;
fitting invocations ≤45 min; ≤2 h aggregate fit/arm; endpoint invocations
≤30 min. If treatment cannot fit, STOP before fitting. Treatment's extra
gradient-bearing views (4,800 additional factual-encoder views and 4,800
additional null streams) are additional compute and are reported, not hidden;
**no compute-matching claim**.

## Fixed endpoints (both arms, updates 0 and 200)

Conditions, all on the 192 root-policy rows: `real`; `shuffle_complete_e002`
(original complete-payload shuffle); `all_null`; `board_flags_zero`;
`successor_frames_zero`; `successor_only_e002` (the frozen training mapping);
`successor_only_e402`; `successor_only_e502`. Donor mappings for seeds
`0x7A60_E002`, `0x7A60_E402`, `0x7A60_E502` are frozen before fitting and bound
by digest. No outcome-based remapping and no additional controls.

Serialized per row: candidate ActionIds, baseline/final logits,
factual/null/raw/centered scores, selected action, set loss, margin, support
counts, unknown counts, flags and donor identities. Argmax uses ActionId-order
first-index tie-break.

## Decision gates (all reported independently; none weakened)

Counting unit is a ROOT (96 roots, 48 B0-wrong, 48 B0-right). Corrected =
B0-wrong and correct under BOTH policies; harmed = B0-right and wrong under
EITHER. Policy rows are not independent roots. Loss contrasts average the two
policies within a root, then the 96 roots equally.

Original P0 gates, applied to the treatment (and reported for the control):
1. finite execution, exact frozen B0 and exact all-null;
2. real policy set-loss reduction ≥ 20% unless initial loss < 0.05;
3. ≥ 12/48 baseline-wrong roots corrected;
4. ≤ 2/48 baseline-right roots harmed;
5. complete-shuffle(E002) − real mean set loss ≥ 0.05;
6. ≥ 6 corrected roots wrong under complete shuffle under BOTH policies.

Additional successor-only gates for EACH **independent** evaluation seed
(`E402`, `E502`): shuffle − real mean set loss ≥ 0.05, and ≥ 6 real-corrected
roots wrong under successor-only shuffle under BOTH policies. The training
mapping (`successor_only_e002`) is evaluated and its values reported, but it is
not independent of training and is not a pass criterion.

Comparative gates, treatment vs fresh control: treatment real mean set loss ≥
0.05 lower; ≥ 4 additional paired corrected roots; no more harmed roots (and
the absolute ≤ 2 limit).

Outcome classes (fixed now): `OBJECTIVE_SUPPORTED_TRAIN_ONLY` (every absolute
and comparative gate passes); `TREATMENT_PASSES_NO_COMPARATIVE_ADVANTAGE`
(absolute pass, comparative fail — learnability with no objective attribution);
`TREATMENT_FAILS` (any absolute gate fails ⇒ probe closed, no retuning). Board
erasure and successor removal survival of helpful/harmful roots is reported even
when gates pass. Content sensitivity is distinguished from semantic utility.

A performance failure never authorizes retry and does not cancel the other fixed
arm. An engineering/integrity failure stops execution. No bootstrap significance
claim on this deliberately stratified, repeatedly exposed panel; exact paired
root deltas are reported. Single CUDA trajectory pair: independent trajectory
variance limits causal confidence.

## Claims

Repeated TRAIN exposure supports learnability/memorization only. No recurrent
superiority, generalization, held-out strength or proof-backup claim. No outcome
licenses DEV, 800 updates, CONFIRM, seeds 6301–6303, controller, self-play or
V5 rescue, and a negative result does not automatically create a V7 or authorize
variants.
