# V5 closure and statistical scope reconciliation

V5 is CLOSED as the completed frozen negative pilot **NO_SIGNAL**. This owner-
accepted result is not a request for another seed, more updates or architecture
rescue. Stage B remains exactly 800 updates; its model/checkpoint bytes stay fixed.

## Three different provenance identities

| Role | Identity |
|---|---|
| Stage-B model producer | `3db24a926815159d592e93b60a8ae51852abad13` |
| Repaired evaluation producer | `7fb7461e94fa4819404913d9a51e91d4d154c167` |
| Reviewed result publication HEAD | `3efce66f62c120df4bf599796c27f66d2ec194d6` |

Later documentation-only closure/design commits do not produce new scientific
model/evaluation source. Model update 800 remains
`c7f6b10a0b60982199cd352c157a377115c6f00a46c699a6a9b3bb859e8bd273`.
All-DEV top1 remains 0.7928888888888889, KQR M3 remains 0.576, KRR M3 remains 0.704.
No reader condition changed a DEV action. [Published result](V5_STAGE_B_RESULTS.md).

## Discrepancy: preregistered primary versus implemented pooled classifier

The V2 plan states **"bootstrap primary KQR M3 n=750"**. However, the unchanged
`classify_pilot` implementation collects/sorts/deduplicates every position in the
complete bundle and requires 4500, without filtering family/mate depth. It averages
the two acquisition-schedule contrasts within each position, then bootstraps all
4500 position-level observations. The published v3 report is that pooled result.
This is a real population-contract discrepancy, not merely a display convention.
The earlier publication made its all 4500 scope explicit, but that does not make
the implementation conform to the primary-population statement in the V2 plan.

Do not silently call the pooled report the preregistered primary test, change its
output, or claim a retrospective correction was the original classifier. V5's
engineering execution remains qualified; this statistical scope limitation is
separate and is now explicitly part of its scientific audit trail.

## Retrospective supplementary primary analysis

Only existing serialized update 800 KQRvK M3 records were used; **zero new DEV model
invocations**. n=750, IDs sorted, original schedule pairing/averaging, original
SplitMix64 seeds 0x7A50_0101..0104, 20,000 resamples and ranks 499/19499. The exact
existing `paired_bootstrap` library function was reused. This is labelled
RETROSPECTIVE SUPPLEMENTARY ANALYSIS, not a new published classifier.

| Contrast | Mean | Paired 95% interval |
|---|---:|---|
| Q8/R4 minus Q8/R1 top1 | 0 | [0,0] |
| Q8/R4 minus B0 top1 | 0 | [0,0] |
| Shuffle minus real Q8/R4 set loss | 0.000002565325624528057 | [-0.0000002226824736014932,0.000005292040717934673] |
| Real minus shuffled loop top1 benefit | 0 | [0,0] |

The supplementary shuffle mean is positive but tiny, its interval includes zero,
and it is far below 0.01. Thus this reconciliation supplies no basis to rescue V5
or replace NO_SIGNAL. [Exact receipt](evidence/v6/primary-retrospective.json).

## Evidence boundaries after closure

The V5 DEV split is now development evidence consulted in design selection. It
must not be presented as untouched final confirmation for V6. TRAIN-only new
diagnostics can investigate mechanism hypotheses; they cannot establish held-out
benefit. CONFIRM remains sealed/unevaluated and requires a future explicit phase
authorization, fixed design, replication and one-time measurement protocol.
V4_TUNE/HOLDOUT_C are not substitutes and remain unevaluated.

Future statistical code must encode the exact primary predicate, expected count,
averaging unit and paired schedule coverage as executable refusal boundaries.
All4500 summaries must be explicitly secondary. No controller/self-play before
useful fixed-information evidence reading is demonstrated.
