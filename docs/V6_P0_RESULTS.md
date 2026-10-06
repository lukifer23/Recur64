# Revised V6 P0 results

**MEASURED: ENGINEERING PASS; BOTH READERS FAIL LEARNABILITY.**

Both disposable arms completed exactly 200 updates on TRAIN only. No retuning,
extra updates, other seeds, DEV inference or scientific campaign occurred.
Stop for owner review.

## Source and corrected contracts

Scientific implementation: 42f47b852451d59645dcc50120e4a426b034f493.
Qualification/panel freeze publication: 15892e5. Isolated branch:
experiment/hp-v6-branch-backup, based on inspected closure publication 1b397f4.
V5 remains NO_SIGNAL. Its primary750 versus pooled4500 reconciliation, original
classifier and retrospective primary analysis are unchanged. No existing
production crate, V5 model/training source or historical checkpoint changed.

Full-information initialization exposes every acquired depth to R1 in BOTH
readers. Shared turn-aware refinement adds computation, not information.
Auxiliary BCE includes only observed branches and available positive/negative
classes; coefficient 0.5. Unknown replies remain unknown. Turn scores are latent
compatibility scores, not winning probabilities or proofs. All-null is exact B0
invariance; the separate turn-blind intervention holds payload fixed. Acquisition
hashes authoritative FEN, never label-bearing dataset IDs. See
[exact equations and contracts](V6_P0_REVISED_CONTRACT.md).

Principal total parameters: 7,687,216; comparator: 6,703,152, each including the
3,677,728-parameter frozen competent root. Same packets, root episode bags and
shared-shape initial tensors; NOT compute matched. The comparator sees deep
nodes, ownership, path, turn and unknown counts immediately.

## Qualification and custody — MEASURED

Full release workspace: 614 passed, two original ignored; native exit 0.
Scoped formatting, CUDA all-target Clippy and serial pinned CUDA build: native
exit 0. Actual CPU and RTX2050 CUDA FP32/microbatch2 qualification passed exact
normal/profile forward, discovered reader gradients, post-AdamW parameter/moment
parity, populated-state clone purity, 50 resident updates per arm, model/moment
restoration and exact continuation. Depth-five R1 dependency, intentionally
absent correction_out.bias, null cancellation and absent baseline gradients
passed. Gradient presence is engineering evidence, not useful-learning evidence.
Historical V5 GPU gradient probes remain NOT RUN.

CUDA device-wide sampled peak: 451 MiB, below 3.2 GiB. Mean instrumented resident
update: principal 0.122176478 s; comparator 0.092814912 s. These include validation
instrumentation; no production-throughput or compute-matched claim. Separate
synchronized acquisition, frozen-root, encoder, initialization, refinement,
readout, backward and AdamW phases are in [costs](evidence/v6-p0/costs.json).

Actual pre/post TRAIN, DEV and CONFIRM raw-byte custody PASS. All pairwise FEN
and canonical intersections zero. CONFIRM remains sealed=true, evaluated=false.
Custody does not invoke models. Independent post-run preservation checked 614
historical files / 12,097,788,866 bytes: zero mismatches. V5 worktree remains clean.

## Acquisition and eligible support — MEASURED

Unchanged 216-position TRAIN panel: 33 B0 errors, 183 B0-right. Narrow ExploitTwo
observed a correct branch for 11/33 errors; BroadRankedHash for 12/33. Both
observed correct branches for 183/183 right roots. Narrow observed 141/298
defender replies on errors; broad 131/418. These sums are over acquired parent
states, not independent roots. Unknown replies remain explicitly represented.
No policy adjustment followed this census. See [full cell/policy/stratum census](evidence/v6-p0/acquisition-coverage-42f47b8.json).

Fixed 96-position panel: 12 B0-wrong and 12 B0-right roots in each KQR/KRR M2/M3
cell. Exact IDs and 192 packet digests are committed in the
[binding](evidence/v6-p0/plan-binding-42f47b8.json). Twenty-three roots have no
observed correct branch under either policy. Across 192 root-policy rows:
182 eligible positive branches, 394 eligible negative branches, 46 rows without
an eligible positive. No support substitution or acquisition label access.

Both arms: seed6300, 200 updates, batch24/microbatch2, warmup20, peak LR1e-3,
existing AdamW/warmup-cosine. Exactly 4,800 root episodes/arm; 25 exposures per
root/policy, 1,200 examples/cell, 2,400/policy. Independent formula check passed.

## Fixed endpoint gates — MEASURED

Counting unit is a ROOT: corrected under BOTH policies; harmed under EITHER.
Shuffle specificity requires a corrected root to become wrong under BOTH
shuffled policies. These are TRAIN-panel learnability results, not held-out strength.

| Frozen gate | Required | Principal | One-pass |
|---|---:|---:|---:|
| Policy loss reduction | >=20% | 15.1694% FAIL | 5.5131% FAIL |
| Initially wrong roots corrected | >=12/48 | 12 PASS | 4 FAIL |
| Initially right roots harmed | <=2/48 | 5 FAIL | 4 FAIL |
| Shuffle minus real set loss | >=0.05 | 0.005970004 FAIL | 0.003698166 FAIL |
| Corrected roots become wrong under shuffle, both policies | >=6 | 0 FAIL | 0 FAIL |
| Finite / frozen B0 / exact null | all required | PASS | PASS |

| Metric, 192 paired-policy rows | Principal 0 -> 200 | One-pass 0 -> 200 |
|---|---:|---:|
| Mean policy loss | 1.257948255 -> 1.067125608 | 1.257985467 -> 1.188631733 |
| Mean eligible auxiliary BCE | 0.693166475 -> 0.482377783 | 0.693096422 -> 0.501379553 |
| Real correct rows | 96 -> 121 | 96 -> 111 |
| Shuffled correct rows | 96 -> 121 | 96 -> 109 |
| Mean correction range | 0.008335607 -> 7.260664305 | 0.006769534 -> 5.457398393 |
| Mean correct/incorrect margin | 0.622703766 -> 1.174473020 | 0.622793604 -> 1.048874859 |

Per policy narrow/broad: principal corrected 14/17, harmed 1/5; comparator
corrected 6/13, harmed 1/3. Policy rows are not independent roots. Equal real
and shuffled correctness counts do not establish identical chosen actions.
Independent endpoint audit reproduces root gates. Per-cell/policy/B0-stratum
losses, margins, eligible support, correction ranges and histories are committed
in each compact arm result. Correction direction is signed margin change from
update0 to200, not an unmeasured raw tensor direction.

Turn-blind intervention (payload unchanged): principal final loss 1.863523791 /
92 correct rows versus real 1.067125608 /121; comparator 1.998776230 /72 versus
real 1.188631733 /111. This establishes sensitivity to the specified turn-pooling
intervention on this panel, not a useful proof backup or held-out structural
benefit. All-null remained exact B0.

## Runtime and checkpoint receipts — MEASURED

Principal: one invocation, no resume, native exit0; complete process 260.477018 s.
Comparator: one invocation, no resume, native exit0; complete process 192.035758 s.
Both below45minutes and2hours/arm. Both fixed endpoints completed before outcome
comparison. No performance retry. Weights are disposable and never reused.

Principal sampled policy loss: first 1.173238377; first50 mean 1.264215076;
last50 mean 1.083421641; last 0.862006495. Auxiliary first50/last50 means
0.622395039/0.486386565.

Comparator sampled policy loss: first 1.173186287; first50 mean 1.255234254;
last50 mean 1.197255711; last 1.111381086. Auxiliary first50/last50 means
0.609994140/0.503200713. Sampled losses are separate from full-panel endpoints.

Exact update0/update200 model and optimizer SHA256s, endpoint hashes, native
execution and detailed aggregates are in [principal receipt](evidence/v6-p0/principal-result.json)
and [one-pass receipt](evidence/v6-p0/one-pass-result.json). Raw checkpoints,
packets and per-root endpoints remain ignored under runs/v6-p0/.

## Interpretation and stop — INFERRED / NOT RUN

**INFERRED:** this implementation changes decisions and correction magnitude
over competent B0, unlike frozen V5. Neither reader satisfies the stronger
content-sensitive learnability contract. Sparse observed correct-branch coverage
limits direct positive supervision; it does not prove nonterminal states useless.
The objective/integration learns some correction and auxiliary patterns, but
weak real-minus-shuffle loss and zero paired shuffle reversals do not demonstrate
useful payload reliance. Structural/root-context exploitation is a competing
explanation, not a proven cause. Nonzero gradients and improved auxiliary BCE
do not settle attribution.

Principal is better on this panel, but parameters/compute differ and neither
passes. No recurrent-superiority or held-out-strength claim follows. No
engineering failure was detected; performance failure must not be relabeled.

**DECISION: BOTH_READERS_FAIL_LEARNABILITY. Stop.** The held-out scientific
campaign has not earned authorization. No next scientific-agent execution prompt
is issued: it is not justified by these gates. Owner review must separate
information support, objective learnability and content attribution before any
separately preregistered change. This pass does not select or implement a rescue.

**NOT RUN:** DEV models; CONFIRM/V4_TUNE/HOLDOUT_C evaluation; scientific800-update
campaign; seeds6301/6302/6303; extra P0 updates/retuning; learned controller;
self-play; V5 GPU gradient probes/rescue.

V5 remains NO_SIGNAL. V5_HP_CONFIRM_V2 REMAINS SEALED AND UNEVALUATED.
