# Frozen V6 P0 failure investigation

Producer 42f47b852451d59645dcc50120e4a426b034f493; reviewed publication b3a2f6f.
Both original learnability gates FAILED; execution valid. V5 stays CLOSED/NO_SIGNAL.
This is diagnosis only: no gradients, optimizer steps, model fitting, retuning,
checkpoint selection, other panels or DEV/CONFIRM model invocation.

## Frozen execution and reproduction gate

Use unchanged 96 TRAIN roots, both policies, both arms at updates0/200, graph-free
FP32 CUDA physical microbatch2. Each invocation <=30minutes, fresh output dirs,
native exit/log captured. Independently verify custody and all available published
artifact hashes. Update0 has no historical state.json; record that absence rather
than claim a nonexistent metadata check. Inventory its optimizer bytes now.
Run original endpoint implementation and compare every serialized field exactly
before interpreting any new controls. Recompute root BOTH-correct/EITHER-harmed
gates separately from policy-row counts. Any reproduction/custody failure STOP.

## Predetermined interventions

All recipient paths, ownership, geometry, legal/observed/unknown counts, topology,
query accounting, masks and legal candidate order remain factual except owner_zero
explicitly removes root-candidate hypothesis context equally from both streams.
Labels are serialized for offline alignment only, never supplied to forward.

| Condition | Observation / flags / terminal stopping / post-encoder anchor |
|---|---|
| real | all factual |
| shuffle_original | original seed0x7A60E002 complete donor observation+flags, donor terminal flag controls stopping; recipient structure unchanged |
| shuffle_e102/e202/e302 | same deterministic matching, fixed seeds0x7A60E102/E202/E302; all mappings recorded, no selection by outcome |
| all_null | original all-null forward, zero anchors and terminal stopping in both streams; exact B0 |
| turn_blind | original pooling-sign intervention, all payload factual |
| board_zero | observation zero, flags and terminal stopping factual; encoder still runs |
| flags_zero | observation factual, all9 flags zero, stopping false |
| board_flags_zero | observation and flags zero, stopping false; encoder may still emit nonzero embeddings |
| carrier_flags_zero | encoder replaced after slot projection by fixed256-vector alternating +0.1/-0.1, identical every acquired node; stopping false |
| carrier_flags_factual | same carrier; terminal stopping factual; other flag encoder channels bypassed |
| owner_zero | payload/structure factual; frozen candidate-context tensor zero in BOTH streams; B0 logits unchanged |
| successor_frames_zero | zero observation history frames k < acquired node.depth (current/successor frames, including validity); retain root/pre-root history, current rule fields, flags and stopping |
| root_history_frames_zero | zero observation history frames k >= acquired node.depth; retain current/successor frames, current rule fields, flags and stopping |

The carrier is an embedding intervention, not a valid chess position. Board/flag
zeroing and owner zeroing are distribution shifts. Collapse alone is not proof
of semantic board use. Structure retains board-derived legal counts; shuffle does
not erase those. Unknown replies remain unknown, never fabricated as covered.
No further structural controls will be selected after observing these outcomes.

Pre-execution amendment: authoritative ObservationV1 contains eight history frames;
the original root board is at frame acquired depth. These two additional fixed
channels are necessary to separate successor-board dependence from root recall.
They are frozen before any new control execution, based on source inspection alone.
Legal counts and current rule fields still carry returned-state information, so
neither history intervention is total removal of all future-derived information.

## Instrumentation and analysis

A separate Rust example mirrors the frozen stream for factual/null decomposition.
Production files/equations remain untouched. Compare its output against actual
production forward on every condition; original endpoint is the independent first
gate. Serialize per-candidate logits, scores, corrections and iteration scores,
ActionIds, targets, selected actions, margins/loss, eligibility and acquired support.
Intermediate tensors are diagnostic outputs, never checkpoint changes.

Compare helpful/harmful roots under all four fixed shuffles and content/carrier
conditions. Stratify arm/update/policy/cell/B0 stratum/observed-correct support/
terminal/reply coverage. Examine factual-minus-null candidate direction, including
unobserved correct actions helped only by suppressing observed alternatives.
Minimum-mate membership is neither generic winning status nor proof certification.
No recurrent superiority or held-out claim from unequal arms/repeated TRAIN exposure.

Publish measured/inferred/not-run distinctions, ranked explanations, one recommended
prospective single-change experiment and complete pending-owner-review prompt.
Do not execute that proposal. Preserve original reports and all historical bytes.
