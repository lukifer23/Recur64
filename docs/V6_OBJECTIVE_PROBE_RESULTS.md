# V6 successor-content differential objective probe — results

**MEASURED: ENGINEERING PASS; OUTCOME `TREATMENT_FAILS`. Probe closed. Stop for owner review.**

Contract: [V6_OBJECTIVE_PROBE_CONTRACT.md](V6_OBJECTIVE_PROBE_CONTRACT.md)
(`v6_content_differential_aux_v2_successor_only`, frozen at `e551bdd` before any
code). The older full-payload proposal is historical. V5 remains CLOSED / NO_SIGNAL;
original V6 P0 remains BOTH_READERS_FAIL_LEARNABILITY and unchanged (all four P0
checkpoint/optimizer/metadata hashes of both arms re-verified identical to their
receipts after this run; V5 worktree clean at its original head).

Scientific source `3cf0aec0c1cb8d24af064a80a21f889344aca1d1`. A first implementation
commit `6faea06` was superseded *before any runtime qualification or fitting* to bind
endpoints/decision to the launch plan and checkpoints (the decision path previously
stamped the current source onto unchecked files). Its builds/lint, including the
two preserved wrong-feature CUDA failures (native exit 101, my `--no-default-features`
invocation error, not a code defect), are kept in
[operations/build-ledger-attempt1.json](evidence/v6-objective-probe/operations/).
Equations, objectives and gates were not changed by that commit.

## Question and answer

*Can a successor-content differential auxiliary objective produce useful,
content-sensitive corrections over competent frozen B0, compared with the original
objective?* **No, not in this fixed 200-update, single-trajectory-pair TRAIN probe.**
The differential term was never learned (below), the treatment failed 8 of 10
absolute gates, and none of the two comparative gates that measure benefit passed.

## Qualification and custody — MEASURED

Format check, CPU Clippy, CUDA all-target Clippy: exit 0. Full release workspace:
624 passed, 0 failed, 2 original ignored. CPU and serial pinned CUDA builds: exit 0.
Actual CPU (84.8 s) and RTX 2050 CUDA FP32 microbatch-2 (62.3 s) qualifications:
native exit 0, both source-bound to `3cf0aec`. They cover: closed-form f64 and
numeric BCE value/gradient references (both signed paths of `d = a − s` exact
negatives, empty support exactly zero, BCE at zero = ln 2, single available class
not averaged with an invented empty class); `d` equal to `F_G − F_S` within fp32;
exactly zero `d` and gradient cancellation under the identity intervention; all-null
exact B0; absent baseline gradients; full reader gradient inventory (no
`correction_out.bias`); independent replicas, clone purity (empty and populated
moments); exact normal/profile output, gradient, post-AdamW parameter and moment
parity per objective; the control equal to the original objective bit-for-bit;
fifty resident updates; model/moment restore and exact continuation; target
mutation leaving predictions and packets unchanged; donor maps independent of
correctness; successor views leaving structure/masks/flags tensors identical; and
93,585,408 intervention channels independently verified. 6,703,152 parameters
(3,677,728 frozen root). CUDA sampled device peak 931 MiB. On CUDA, the new
evaluator reproduced all 960 compared rows of the frozen P0 one-pass diagnostics
exactly (inspection only; those weights never initialized anything).
Pre/post custody pass; all overlaps zero; CONFIRM sealed, no model on DEV/CONFIRM.

Both arms: seed 6300, one invocation each, no resume, native exit 0, 200 updates,
4,800 root episodes (25 exposures/root/policy), fresh optimizers, identical
canonical initial reader (`d96aa862…b817`) and identical update-0 optimizer
(`11e3c6fc…3d40`); frozen B0 digest unchanged; all-null exactly B0.

## Fixed endpoints and gates — MEASURED (recomputed independently with `jq`)

| Gate | Required | Control | Treatment |
|---|---:|---:|---:|
| Finite, exact B0 / all-null | yes | PASS | PASS |
| Real policy-loss reduction | ≥20% | 5.50% FAIL | 2.21% FAIL |
| Corrected B0-wrong roots (both policies) | ≥12/48 | 4 FAIL | 4 FAIL |
| Harmed B0-right roots (either policy) | ≤2/48 | 4 FAIL | 1 PASS |
| Complete-shuffle − real set loss | ≥0.05 | 0.00354 FAIL | 0.00097 FAIL |
| Corrected roots wrong under complete shuffle, both policies | ≥6 | 0 FAIL | 0 FAIL |
| Successor-only E402 shuffle − real loss | ≥0.05 | −0.00024 FAIL | 0.00013 FAIL |
| Corrected roots wrong under E402, both policies | ≥6 | 0 FAIL | 0 FAIL |
| Successor-only E502 shuffle − real loss | ≥0.05 | −0.00025 FAIL | 0.00014 FAIL |
| Corrected roots wrong under E502, both policies | ≥6 | 0 FAIL | 0 FAIL |

Successor-only E002 (the training mapping, reported, not a pass criterion):
contrast control −0.00025 / treatment +0.00014; reversals 0 / 0.

| Comparative gate (treatment vs fresh control) | Required | Value |
|---|---:|---:|
| Real mean set loss lower by | ≥0.05 | −0.0414 (treatment worse) FAIL |
| Additional paired corrected roots | ≥4 | 0 (4 vs 4) FAIL |
| Harmed no more than control and ≤2 | yes | 1 vs 4 PASS |

Real mean set loss 1.25799 → control 1.18880, treatment 1.23025. Real correct
rows (of 192): 96 → 111 control / 104 treatment. Mean correction range 0.0068 →
5.42 control / 0.83 treatment. Mean auxiliary BCE on `delta_G`: 0.693 → 0.502
control / 0.861 treatment (the treatment does not optimize it). Policy rows are
not independent roots; per-policy corrected/harmed rows: control 6/1 and 13/3,
treatment 6/1 and 4/1.

**The differential objective was not learned.** Treatment A(d) was 0.69315 at
update 1 (first-50 mean 0.69313) and 0.69300 over the last 50 updates, i.e. ln 2
throughout; at update 200 mean |d| on eligible candidates (E402) is 0.00125
(max 0.0055), against mean |raw correction| 0.446 (control: |d| 0.00039, |raw|
2.86). The treatment's real correction instead shrank. The control reproduces the
original P0 one-pass endpoint within CUDA trajectory variation (4 corrected, 4
harmed, loss 1.18880 vs 1.18863).

## Root transitions, erasure/removal survival, support — MEASURED

Control corrected (4): KQR M2, KQR M3, KRR M3 ×2; control harmed (4): KQR M3 ×3,
KRR M2. Treatment corrected (4): KQR M2, KRR M2 ×2, KRR M3; treatment harmed (1):
KQR M3. Overlap: two corrected roots are shared (KQR M2, one KRR M3); each arm
has two corrected roots the other lacks (control: KQR M3, KRR M3; treatment: KRR M2 ×2);
the treatment's one harmed root is also harmed by the control. Exact IDs, paired deltas
in [decision.json](evidence/v6-objective-probe/decision.json); per-root, per-condition
correctness and selected actions in the two `*-analysis.json` files.

Survival of the real-corrected roots (corrected under both policies) / real-harmed
roots (still harmed): board+flags erased — control 4/4 and 4/4, treatment 4/4
and 1/1; successor frames removed — control 4/4 and 2/4, treatment 4/4 and 1/1.
All eight conditions give essentially identical treatment losses (1.2302–1.2312).

Support: of 25 B0-wrong roots with an observed correct branch under both policies,
control corrected 3 and treatment 4; of 23 with none, control corrected 1 (the same
unobserved-branch KRR M3 root P0 diagnosed) and treatment 0. Per cell
(corrected/harmed): control KQR M2 1/0, KQR M3 1/3, KRR M2 0/1, KRR M3 2/0;
treatment 1/0, 0/1, 2/0, 1/0.

## Compute — MEASURED

Control: 4,800 gradient-bearing factual views, 4,800 null streams; fit wall 177.5 s
in-process (199.5 s native incl. endpoints). Treatment: 9,600 factual views and
9,600 null streams (+4,800 extra gradient views; two full reader passes per
microbatch), fit wall 229.5 s (251.5 s native); mean synchronized update 0.845 s vs
1.105 s (1.31×). Resident qualification ratio 1.17× (0.0799 vs 0.0938 s). Peak
sampled device memory 291 / 323 MiB during fitting. No acquisition at fit time
(frozen packets). **Not compute matched.**

## Interpretation — INFERRED (competing explanations kept)

* The treatment did not get a usable gradient to a successor-dependent solution in
  200 updates: the reader would have to discriminate a real successor frame from a
  same-cell donor frame, and A(d) never left ln 2. Not tested: optimizer
  difficulty vs signal absence in the representation vs the objective's sign
  structure. One seed pair cannot separate them.
* Fewer harmed roots in the treatment coincide with much smaller corrections, not
  with demonstrated content use: its helpful and harmful roots survive every
  intervention, so they are not successor-content dependent either.
* The control's content insensitivity (positive complete-shuffle contrast of 0.0035,
  negative successor-only contrasts) is consistent with root/structure-conditioned
  correction, as the P0 diagnosis inferred; this run does not prove it.
* Passing would not have shown semantic reasoning (root/action consistency
  detection was a competing explanation); failing does not show that returned
  successor boards contain no learnable signal.

## Limitations

96-root stratified panel exposed 25× per policy: learnability/memorization only.
Single trajectory pair on CUDA (nondeterminism limits attribution); one seed.
Donor tensors are not legal histories. No bootstrap or significance claim.
Two arms differ in compute. 23/48 B0-errors have no observed correct branch.

## NOT RUN

DEV/CONFIRM/HOLDOUT models; extra updates, seeds, panels or acquisition; learning
rate or objective variants; recurrent/principal reader under this objective;
800-update campaign; controller; self-play; any V5 work. No outcome-based remap,
checkpoint selection or retry occurred.

## Evidence

[Receipts and hashes](evidence/v6-objective-probe/): qualification (CPU/CUDA),
plan binding, custody, arm results (history, compute, hashes), analyses,
[decision](evidence/v6-objective-probe/decision.json), independent `jq`
recomputation ([script](evidence/v6-objective-probe/operations/recompute.jq),
[output](evidence/v6-objective-probe/independent-recompute.jsonl)). Raw
checkpoints and endpoint matrices stay ignored under `runs/v6-objective-probe/`.
