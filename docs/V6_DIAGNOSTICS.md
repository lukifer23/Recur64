# V5 mechanism diagnosis for V6

Status: completed design investigation, not a V6 model experiment. V5 is closed
as **NO_SIGNAL**. [Closure and population reconciliation](V5_CLOSURE.md),
[frozen diagnostic plan](V6_DIAGNOSTIC_PLAN.md),
[complete numerical receipt](evidence/v6/train-diagnostics.json).

## Provenance and independently repeated checks

The reviewed publication was `3efce66f62c120df4bf599796c27f66d2ec194d6`.
Stage B was produced by `3db24a926815159d592e93b60a8ae51852abad13`; the
source-matched cached evaluator library is from
`7fb7461e94fa4819404913d9a51e91d4d154c167`. Documentation commits do not replace
either identity. Update 0 model SHA is
`2d1c770a43a6455148b774e9ddb552b6ca33efd9fdd5d37593cefe7c8ae0bb00`;
update 800 is `c7f6b10a0b60982199cd352c157a377115c6f00a46c699a6a9b3bb859e8bd273`.

MEASURED here: branch/remote/worktree status, absence of active scientific
processes at handoff, hashes of all 614 files in the preservation inventory,
raw split bindings, both serialized full DEV matrices and accepted B0 publication.
Fresh native data verification read actual TRAIN/DEV/CONFIRM bytes; all custody
checks passed, all pairwise FEN/canonical intersections were zero and CONFIRM was
sealed=true, evaluated=false. [Start receipt](evidence/v6/start-verification.json),
[custody](evidence/v6/current-custody.json).

Historical receipts inspected, NOT rerun: CPU/CUDA qualification, full release
tests, source-scope audit, engineering drill, training exposure and baseline
parameter preservation. Their original source-bound claims remain historical.
No production model, training, data, loss, optimizer or configuration was edited.
Cross-disjointness from unavailable workstation-only raw datasets remains NOT
VERIFIED; those artifacts are not used as V5/V6 evaluation inputs.

## Fixed panel and execution limits

The panel was committed/pushed before execution: 216 TRAIN positions, 24 per
native heavy-family/depth cell, chosen solely by frozen ID hash. Panel ID digest
`33831f8a6c80d27899714362a12dacba31fe7095dcc1601934e96927b99739b5`.
There were 183 baseline-right and 33 baseline-wrong positions. These are a small
diagnostic panel, not an estimate of full TRAIN or held-out performance.

One Q8 graph per root/schedule was acquired using frozen B0; identical graph
digests were reused at updates 0/800. Each successful forward invocation produced
2,592 records: 216 roots × two schedules × R1/R4 × real/shuffle/null. RTX 2050,
FP32, microbatch 2; no optimizer construction or steps. Before/after parameter
digests were exact. The normal checkpoint loader also runs its fixed baseline
integrity fingerprint; that is not a new DEV measurement. No new DEV inference.

The initial scratch forward utility incorrectly retained an autodiff graph and
failed with native exit 101 (CubeCL memory-page/resource error), before rows were
serialized. Resource retention is a likely explanation, not a proven backend
defect. The failure was preserved, then a documentation-only instrumentation
amendment was committed before one graph-free recovery. Recovery forward 0/800
both exited 0. The gradient utility then refused with native exit 1, `head visitor
absent`, in parameter-discovery preflight **before backward**. The amended hard
stop was respected: no further GPU probes. Gradients, factual-only/null-only
gradient attribution and gradient800 are **NOT RUN**. This limits credit-
assignment diagnosis; no nonzero-gradient claim is made. Raw utilities/logs are
ignored scratch artifacts; their hashes and exits are in the compact receipts.
[First failure](evidence/v6/diagnostic-failure.json).

## A — Acquired information is sparse on baseline errors

MEASURED coverage reconstructs only acquired states/edges. Legal-action counts at
observed nodes provide denominators; no unseen descendants or solver values are
queried. The terminal-only partial diagnostic has three values: WIN, NOT_WIN,
UNKNOWN. At attacker nodes, an observed WIN child suffices; NOT_WIN requires all
legal children observed and NOT_WIN. At defender nodes, an observed NOT_WIN child
suffices; WIN requires all legal children observed and WIN. Missing replies stay
UNKNOWN. Terminal wins are relative to the root attacker; draws/losses are NOT_WIN.

Root ProofTargets were used offline only to stratify/evaluate this diagnostic,
never as reader inputs or donor-selection information. A proven winning branch
can be slower than the exact minimum-mate correct set; the two labels differ.

| Population/schedule | Correct branch observed | Defender replies observed/legal | Terminal mates | Roots with partial winning branch |
|---|---:|---:|---:|---:|
| All 216, uniform | 56/216 | 247/4515 | 19 | 17 (15 match exact correct set) |
| All 216, ranked | 194/216 | 620/2290 | 124 | 74 (74 match exact correct set) |
| Wrong 33, uniform | 9/33 | 43/785 | 0 | 0 |
| Wrong 33, ranked | 11/33 | 99/466 | 1 | 1 |

On wrong roots, uniform covers 5.576 branches on average and 15.39% of root legal
actions, with mean maximum depth 2.515. Ranked covers 2 branches and 5.51%, depth
4.970. Ranked observes B0's selected branch on all 33 wrong roots, but a correct
branch on only 11. Complete defender coverage occurs at 1/219 uniform observed
defender nodes and 26/163 ranked nodes. The detailed receipt contains every cell,
schedule and wrong/right stratum, rather than pooling away this asymmetry.

INFERRED: coverage is a serious limiting factor, especially the concentration of
deep ranked evidence on already-correct B0 choices. It cannot explain everything:
one wrong root had a conservatively certified correct winning branch and still
did not change action. n=1 is illustrative, not a population causal conclusion.
Absence of terminal proof does **not** imply nonterminal board states are useless.
Falsifier: useful payload effects on a prospectively fixed reply-covered panel
would show that terminal sparsity was not the binding constraint there.

## B — Competent-base correction learning did not become useful

MEASURED: no action changed in any of the 5,184 successful forward records.
Baseline top-two margin medians were 2.277 on right roots and 0.495 on wrong roots.
New reader top-two margins were not serialized; we report correction ranges
relative to the baseline margin, not fabricated reader margins.

| Wrong roots, real R4 | Update 0 set loss | Update 800 set loss | Update 800 correction range | Range/B0 margin median |
|---|---:|---:|---:|---:|
| Uniform | 2.30185210 | 2.30455425 | 0.03899256 | 0.08802442 |
| Ranked | 2.30182515 | 2.30464453 | 0.04043498 | 0.08981026 |

At update 800, 32/33 wrong roots in each schedule have correction range below the
B0 top-two margin; therefore no such correction could change the argmax there.
The remaining root also did not change action: adequate range alone does not
establish useful direction. Right-root set loss slightly improves, uniform
0.19302966→0.19221053 and ranked 0.19303462→0.19218072. Wrong-root real-minus-null
loss is +0.00277489/+0.00286518, while real-minus-shuffle loss is only
+0.000006738/−0.000002036. Content effects are tiny and mixed in sign.

INFERRED: the frozen-base objective can reward mild calibration on easy/right
examples without teaching branch-specific corrections that overcome wrong action
margins. This is a hypothesis about the learned solution, not proof of zero
gradient or a broken optimizer. Direct TRAIN branch-correctness supervision is
proposed to test it. A competent-base drill that reliably fixes wrong roots and
loses that benefit under payload shuffle would falsify a general inability to
learn corrections through the proposed interface.

The historical 24-position drill memorized against a frozen **random** baseline.
That proves a reader can reduce a disposable panel's loss; it does not qualify
learning an incremental useful correction over trained Stage A. Its two historical
final losses (0.0266246 and 0.0977522), despite matching seed/positions/graphs/
initial loss/math, also show cross-process trajectory variance. This is not a
proven defect. Future positive single-seed results require independent replication.

## C — Routing/integration is active, but task alignment is weak

DETECTED from production equations: acquired nodes retain their root candidate
owner. Evidence initializes from returned anchor, owner hypothesis and structure.
Evidence reads evidence/all hypotheses/root context; hypotheses read all
hypotheses/all evidence/root context. A shared readout subtracts null from factual,
centers candidate corrections, and adds frozen B0. Owner relationships are present;
no narrow missing owner/mask implementation defect was detected.

MEASURED on wrong roots, update800 R4 hypothesis attention mass, averaged over
valid candidate queries:

| Schedule | Own-branch evidence | Other evidence | Candidate context | Root context |
|---|---:|---:|---:|---:|
| Uniform | 0.011893 | 0.419607 | 0.332391 | 0.236109 |
| Ranked | 0.011461 | 0.416141 | 0.334527 | 0.237872 |

Own-branch mass exceeds the uniform expectation 0.006672, so this is not proof of
missing routing. Most candidate branches are unqueried; this averaging denominator
must be retained. Global other-branch mixing may dilute decision-aligned evidence,
but attention is not a causal attribution measure. Evidence attention increased
after training and root attention decreased: “it only attends to root context”
does not fit these measurements.

First-loop factual-minus-null projected evidence pair distance falls from 0.11815
to 0.06404 (uniform wrong roots), and 0.19603→0.11038 (ranked). Returned states
remain distinguishable. Later trace differences are evolving stream states, not
raw encoder outputs. Uniform factual/null hypothesis RMS at loop4 grows from
0.06740/0.06514 to 0.14514/0.14952. Correction range also grows substantially.
Neither normalized RMS nor diffuse attention proves representational collapse.

## Ranked explanation and next discriminating test

1. **Weak decision-aligned learning over competent B0**, supported by zero action
   changes, wrong-root loss worsening and tiny payload contrasts. Gradient-path
   utility remains unmeasured. Test explicit branch supervision and wrong/right
   action/margin/payload gates before held-out science.
2. **Insufficient/wrongly concentrated acquired information**, supported by scarce
   correct branches and opponent replies on errors. Test fixed branch/reply
   coverage with identical packets for all readers, and report missing information.
3. **Global integration dilutes branch evidence**, plausible from equations and
   routing measurements, not proven collapse. Test hard branch-local adversarial
   aggregation against a nonrecurrent same-information scorer and compute control.

These explanations can coexist. The recommended V6 changes the evidence-learning
mechanism and records acquisition separately; it does not pretend one experiment
isolates every V5 failure cause. A V5.5 bug correction is not supported: the only
new defect detected was in scratch diagnostic instrumentation, not frozen V5.
No V6 implementation, model fitting, new DEV inference, R8/Q16, controller,
replication or sealed-set evaluation ran in this investigation.

## Per-cell graph coverage (all selected roots)

Each row is24 TRAIN roots; wrong/right details remain in the numerical receipt.

| Cell | Schedule | Mean branches | Mean maximum depth | Correct branch seen | Defender observed/legal replies | Partial WIN roots |
|---|---|---:|---:|---:|---:|---:|
| KQQvK M1 | uniform | 6.500 | 2.125 | 3/24 | 19/298 | 4 |
| KQQvK M1 | ranked | 3.542 | 4.583 | 24/24 | 60/139 | 24 |
| KQQvK M2 | uniform | 5.417 | 2.667 | 7/24 | 31/527 | 0 |
| KQQvK M2 | ranked | 2.000 | 5.000 | 21/24 | 72/251 | 0 |
| KQQvK M3 | uniform | 5.250 | 2.500 | 11/24 | 35/647 | 0 |
| KQQvK M3 | ranked | 2.000 | 5.000 | 23/24 | 72/396 | 0 |
| KQRvK M1 | uniform | 6.083 | 2.250 | 4/24 | 17/340 | 5 |
| KQRvK M1 | ranked | 3.583 | 4.667 | 24/24 | 62/140 | 24 |
| KQRvK M2 | uniform | 6.125 | 2.417 | 3/24 | 21/497 | 0 |
| KQRvK M2 | ranked | 2.000 | 4.833 | 21/24 | 71/236 | 1 |
| KQRvK M3 | uniform | 5.750 | 2.458 | 8/24 | 26/655 | 0 |
| KQRvK M3 | ranked | 2.000 | 4.958 | 15/24 | 72/324 | 0 |
| KRRvK M1 | uniform | 5.750 | 2.375 | 8/24 | 20/329 | 8 |
| KRRvK M1 | ranked | 3.208 | 4.958 | 24/24 | 67/156 | 24 |
| KRRvK M2 | uniform | 5.333 | 2.625 | 5/24 | 37/523 | 0 |
| KRRvK M2 | ranked | 2.000 | 5.000 | 23/24 | 72/281 | 1 |
| KRRvK M3 | uniform | 5.042 | 2.792 | 7/24 | 41/699 | 0 |
| KRRvK M3 | ranked | 2.000 | 5.000 | 19/24 | 72/367 | 0 |
