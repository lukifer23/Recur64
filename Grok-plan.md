You are the senior research engineer and independent experimental-design reviewer
for:

Recur64
https://github.com/lukifer23/Recur64

TARGET BRANCH:

experiment/hp-r15

CURRENT TARGET HEAD:

89d5ccba4c986dbe5148bbe0f0ffd7f5c212c7b1

IMPORTANT MERGE COMMIT:

fa66c3297332f083b3f6236bc45a8af6b0e7df5d

MAINLINE PHASE 3 SOURCE:

5ac291c6736b6037703522c49db62f2296c20bde

COMMON ORIGINAL HP ANCESTOR:

78be2052612236547f6b5232b417175c2ccdcfc9

============================================================
MODE
============================================================

PLAN MODE ONLY.

DO NOT MODIFY FILES YET.
DO NOT COMMIT.
DO NOT PUSH.
DO NOT START SELF-PLAY.
DO NOT START TRAINING.
DO NOT RUN LONG TESTS.
DO NOT RUN CUDA BENCHMARKS WHILE ANOTHER WORKLOAD MAY BE ACTIVE.
DO NOT BEGIN F15 OR R15 LEARNING.

Inspect the current merged branch thoroughly and produce the implementation
plan for the NEXT HP PHASE.

After presenting the plan:

STOP.

Wait for explicit approval before implementation.

============================================================
PHASE NAME
============================================================

HP H1:

EXPERIMENTAL HARNESS HARDENING + F15 QUALIFICATION

This phase occurs AFTER:

- HP hardware discovery
- F15/R15 ~15M model definition
- CUDA proof
- GPU benchmark proof
- real-concurrency proof
- merge/reconciliation of main Phase 3 infrastructure

and BEFORE:

- meaningful F15 training
- R15 recurrence training
- recurrence-vs-search claims
- any 24-hour run

============================================================
PRIMARY OBJECTIVE
============================================================

Make the HP experimental branch sufficiently rigorous that a subsequent F15
learning result can be trusted.

Then:

1. measure the actual RTX 2050 scheduling envelope,
2. select a defensible PUCT budget,
3. run a short real-learning F15 smoke,
4. run a bounded F15 qualification pilot if and only if the smoke is healthy,
5. produce an explicit GO / CONDITIONAL GO / NO-GO package for entering R15.

Do NOT enter R15 during this phase.

============================================================
CURRENT HARDWARE
============================================================

HP home machine:

CPU:
AMD Ryzen 5 7535HS
6 physical cores
12 logical threads

RAM:
~32 GB

GPU:
NVIDIA GeForce RTX 2050
4 GB VRAM
Ampere compute capability 8.6

OS:
Windows 11

CUDA:
user-space CUDA 12.9.1

Framework:
Burn 0.21.0

Rust:
1.97.1

Known direct-model measurements before the Phase 3 merge:

F15:
~15.154M params
warm inference plateau roughly ~620–640 examples/sec at batches 16–64
training batch 64 ~220 examples/sec

R15:
R1 ~617 ex/s
R2 ~413 ex/s
R4 ~249 ex/s

VRAM:
roughly ~1.95 GB F15
~2.11 GB R15 in measured workloads

temperature:
~62–63 C

GPU can hit 100% utilization in direct benchmarks.

Pre-merge real self-play concurrency measurement:

best observed approximately:
334.6 evaluations/sec
12 workers
batch p50 = 11

VERIFY all committed claims before relying on them.

============================================================
CURRENT MODEL FAMILY
============================================================

F15:

width = 512
heads = 8
head_dim = 64
FFN = 768

layout:
0 input
8 core
0 output

unique params:
15,154,120

R15:

width = 512
heads = 8
FFN = 768

layout:
2 input
4 shared recurrent core
2 output

unique params:
15,154,120

R15 executed blocks:

R1 = 8
R2 = 12
R4 = 20

============================================================
IMPORTANT SCIENTIFIC WARNING
============================================================

DO NOT assume:

F15 == R15 R1.

The current parity test proves:

- equal unique parameter count
- expected executed-block accounting
- R15 forward_r(R=1) equals R15's own explicit control path

It does NOT prove:

F15 and R15 R1 are functionally identical under a deterministic weight mapping.

Inspect:

crates/recur64-model/src/model.rs
crates/recur64-model/tests/f15_r15_parity.rs

Pay special attention to:

embed
input_blocks
inject_norm
alpha injection
core_blocks
output_blocks

Determine whether the placement of input injection means F15 and R15-R1 belong
to different function classes.

I currently suspect they do.

DO NOT force equivalence if it is not mathematically true.

Instead determine the cleanest experimental interpretation.

A likely interpretation is:

F15:
independent feed-forward architecture control

R15 R1:
recurrent-family baseline

R15 R2/R4:
additional internal-compute conditions

Under that interpretation the clean recurrence comparison is:

R15 R1
vs
R15 R2
vs
R15 R4

while F15 is a separate matched-parameter architecture control.

VERIFY this.

Design the future recurrence experiment accordingly.

============================================================
P0 — UNIFY SELF-PLAY COLLECTION
============================================================

There are currently multiple collection implementations.

Inspect at minimum:

crates/recur64-runtime/src/coordinator.rs
crates/recur64-runtime/src/pilot.rs
crates/recur64-runtime/src/sweep.rs

The HP branch added useful SelfPlayMetrics:

- terminations
- W/D/L
- truncations
- mean game length
- inference batching metrics
- peak_in_flight

But the Phase 3 pilot has its own `collect_games()` path.

Determine whether pilot collection currently bypasses these metrics.

If so:

PLAN TO ELIMINATE THE DUPLICATE SEMANTICS.

There should be one authoritative concurrent collection engine, with callers
able to specify:

- total games to collect
- maximum concurrent games
- deadline
- model snapshot
- search config

and receive:

- replay records
- inference metrics
- self-play data-health metrics

Do not maintain three subtly different definitions of self-play concurrency.

============================================================
P0 — SEPARATE TOTAL GAMES FROM CONCURRENCY
============================================================

Audit current semantics of:

active_games
cpu_workers

The merged Phase 3 implementation appears to make:

active_games = simultaneous game threads

AND:

active_games = total games collected in that invocation.

This coupling is undesirable on the HP.

We need to be able to express:

collect 64 games
with at most 12 concurrent games

for example.

Design explicit concepts such as:

games_per_cycle
concurrent_games

or equally clear names.

Do not create unnecessary knobs.

`cpu_workers` must either:

A. have a real, documented purpose

or:

B. be deprecated/removed from the relevant path.

Do NOT leave a configuration value that appears to limit worker count while the
runtime ignores it.

Backward compatibility should be handled explicitly.

============================================================
P0 — REPLAY REUSE MUST CONTROL TRAINING
============================================================

Main Phase 3 exposed:

replay reuse ~0.07–0.13
versus target 2.0.

The current HP branch has:

replay_reuse_target = 2.0

but appears to still use fixed:

max_updates

as the actual training workload.

Verify this.

If true, the target is telemetry rather than control.

Design data-dependent update budgeting.

Conceptually:

effective_batch =
train_batch * accumulation_steps

desired_examples =
new_trainable_positions * replay_reuse_target

desired_updates =
ceil(desired_examples / effective_batch)

Need carefully define:

NEW POSITIONS
vs
TRAINABLE NEW POSITIONS
vs
TOTAL ACTIVE REPLAY
vs
EXAMPLES ACTUALLY CONSUMED.

Use the correct denominator.

Add:

- minimum updates where appropriate
- configurable maximum safety cap
- time-budget awareness
- explicit requested reuse
- achieved reuse
- reason if target could not be reached

Do not hide a miss.

============================================================
P0 — OPTIMIZER CONTINUITY
============================================================

Audit:

crates/recur64-runtime/src/pilot.rs

Current behavior appears to:

1. load the promoted snapshot's model weights
2. construct a fresh AdamW optimizer each cycle
3. continue the global LR schedule using cumulative update count

If confirmed, this means:

continuous model weights
+
continuous LR schedule
+
RESET Adam moments

between cycles.

That is not a coherent continuation of the same optimizer trajectory.

Fix the design.

A promoted training checkpoint should preserve and restore:

- model weights
- Adam moments/state
- update counter
- LR schedule step
- relevant metadata

The next cycle should continue from the promoted checkpoint.

The frozen initial reference remains separate.

Add a deterministic CPU test proving:

continuous N updates

matches:

K updates
→ save
→ next pilot cycle/load
→ remaining N-K updates

within the strongest deterministic guarantee the backend supports.

Do not merely test same-cycle checkpoint reload.

============================================================
P0 — FIX LINEAGE SEMANTICS
============================================================

Audit:

LineageRecord
pilot snapshot promotion
parent_model_id
candidate_model_id
replay_model_ids
git_revision
config_hash

Current code appears capable of updating `snapshot_model_id` to the new candidate
BEFORE writing lineage.

If confirmed, a promoted cycle may record the candidate as its own parent.

Fix this.

Capture immutable per-cycle identity before training:

parent_snapshot_id
replay_generator_id
candidate_id

Then make the lineage record reflect actual chronology.

Also populate git revision / branch where available.

Lineage must answer:

Which checkpoint generated this replay?

Which checkpoint was trained?

Which candidate resulted?

Which candidate was promoted?

Which checkpoint became the next parent?

============================================================
P0 — PROMOTION GATE MUST MEAN SOMETHING
============================================================

Audit the current "conservative" snapshot policy.

Current health appears close to:

audit passes
AND updates > 0
AND candidate arena score >= 0.35

This is insufficient.

A draw-only arena scores ~0.5 and can therefore promote a model with no evidence
of improvement.

Main already observed almost entirely repetition-draw arenas.

Design a meaningful promotion gate.

DO NOT invent arbitrary thresholds without exposing them as config / documented
experimental choices.

At minimum consider:

SYSTEM HEALTH:
- audit passes
- no inference errors
- no NaN/Inf
- checkpoint valid

DATA HEALTH:
- acceptable truncation
- termination distribution recorded
- repetition/fifty-move fraction visible

LEARNING HEALTH:
- updates > 0
- achieved replay reuse reasonably close to requested target
- losses finite
- gradients finite
- update norms sane

EVALUATION HEALTH:
- enough decisive information for searched arena OR mark arena uninformative
- raw-policy signal recorded

IMPORTANT:

Separate two evaluation roles:

A. CANDIDATE VS PARENT
used for promotion.

B. CANDIDATE VS FROZEN INITIAL REFERENCE
used to measure longitudinal progress.

Do not use one comparison for both purposes.

============================================================
P0/P1 — SELF-PLAY DATA-HEALTH TELEMETRY
============================================================

The actual F15 pilot must report:

- games
- plies
- mean / median game length if practical
- W/D/L
- checkmate
- stalemate
- insufficient material
- threefold
- fifty-move
- truncated
- aborted
- repetition share
- draw share

Inference:

- submitted evaluations
- errors
- actual concurrency
- batch mean
- p50
- p95
- max
- queue wait
- forward latency

Search/data target health if practical:

- root visit entropy
- target entropy
- top-1 visit share
- repeated positions/game

Do not make target-level diagnostics expensive enough to materially alter the
experiment.

============================================================
P1 — STRICT EVALUATION INPUTS
============================================================

Audit:

eval_policy.rs
arena.rs
opening-suite loading

Current code appears to silently fall back to startpos if:

- an opening FEN is invalid
- an opening suite fails to load

This is unacceptable for a frozen scientific evaluation.

Configured evaluation data should fail visibly.

No silent substitution.

Add tests:

invalid FEN → error
missing configured suite → error
malformed suite → error

Standard-start fallback is acceptable ONLY when no suite was requested.

============================================================
P1 — RAW POLICY EVALUATION
============================================================

`raw_policy_vs_random` is useful, but insufficient for the eventual recurrence
claim.

Keep it as a weak anchor.

Plan an additional raw-network comparison that can evaluate:

candidate vs parent

without tree search.

Eventually it must support:

R15 R1
R15 R2
R15 R4

Potential outputs:

- raw game score
- W/D/L
- move agreement
- entropy
- WDL change
- top-k overlap

Do not introduce Stockfish labels.

============================================================
P1 — SCIENTIFIC CONFIG HASH VS HARDWARE HASH
============================================================

Current `config_hash()` appears to hash the entire resolved RunConfig.

That means changing:

batch size
concurrency
batch timeout

may change the same identity used for the scientific experiment.

We specifically want to distinguish:

SCIENTIFIC CONFIG

from

HARDWARE SCHEDULING CONFIG.

Design:

scientific_config_hash

and:

resolved_config_hash

or equivalent.

Scientific identity should include things such as:

- model architecture
- recurrence
- search simulations
- c_puct
- temperature schedule
- training objective
- optimizer hyperparameters
- replay policy
- seed policy

Hardware identity should include:

- concurrent games
- inference max batch
- timeout
- physical train batch where appropriate
- host worker settings

Be deliberate about ambiguous parameters such as physical batch when gradient
accumulation keeps effective batch fixed.

============================================================
P1 — GRADIENT ACCUMULATION METRICS
============================================================

Audit:

crates/recur64-runtime/src/learner.rs

Verify Burn's `GradientsAccumulator` semantics.

Determine whether accumulated gradients are:

summed
or
averaged

and make the effective learning-rate interpretation explicit.

Current update metrics appear to retain loss/entropy components from only the
LAST microbatch in an accumulation cycle.

If confirmed:

fix metrics to represent the entire effective batch or a clearly documented
mean.

Do not report last-microbatch loss as though it describes the optimizer update.

Keep gradient clipping semantics explicit:

global grad norm should clearly say whether it is PRE-CLIP or POST-CLIP.

The optimizer currently appears configured for norm clipping at 1.0; verify.

============================================================
P1 — HP-SPECIFIC BATCHING SWEEP
============================================================

Do NOT use workstation-oriented grids blindly.

The current general grid includes values such as:

32 / 64 / 128 / 256 concurrent games.

HP hardware:

6c / 12t Ryzen
RTX 2050 4 GB

and the pre-merge experiment found a useful point near:

12 workers
batch p50 ~11
~334.6 evaluations/sec.

Design the HP sweep around the machine.

Suggested candidates to consider:

concurrency:
6
8
12
16
24

max inference batch:
8
16
32
possibly 64

timeout:
500
1000
2000 us

Do not blindly use this exact grid if code inspection suggests something better.

Evaluate using USEFUL throughput:

- evaluations/sec
- positions/sec
- wall time
- batch quality
- CPU utilization if available
- GPU utilization if available

Not merely GPU %.

============================================================
P1 — DO NOT CONFLATE BATCH CAP WITH OPTIMAL BATCH
============================================================

Direct F15 inference was approximately flat from batch ~16 onward.

Therefore:

"batch 64 fits"

does NOT imply:

"batch 64 is optimal."

The self-play workload should naturally coalesce into the most useful batches.

Select `max_inference_batch` as a ceiling based on measured latency/throughput,
not as a target that must be filled.

============================================================
P1 — VRAM MEASUREMENT
============================================================

Audit `sweep.rs::sample_vram_mb`.

Current sweep code appears to sample VRAM around workload boundaries rather than
continuously during the workload.

If so, `peak_vram_mb` is not a true peak.

Either:

A. implement a bounded periodic sampler during each cell,

or

B. rename the metric accurately and use an external/parallel sampler for peak
   memory qualification.

Never claim a boundary sample is peak VRAM.

============================================================
P1 — SEARCH-BUDGET QUALIFICATION
============================================================

Do not assume 128 simulations is automatically better.

Main observed repetition-dominated PUCT with an untrained network.

Deeper search can amplify a bad value prior.

Run a bounded PRE-LEARNING comparison using the frozen random F15.

At minimum consider:

32
64
128

Only add 256 if throughput and diagnostic value justify it.

Measure:

- positions/sec
- evaluations/sec
- game completion
- threefold rate
- fifty-move rate
- checkmate rate
- truncation rate
- mean game length
- search-target entropy
- root-visit concentration
- batch behavior

The goal is NOT:

maximum simulations.

The goal is:

a search budget producing useful non-degenerate learning targets at acceptable
cost.

Freeze the selected budget before the learning smoke.

============================================================
P1 — F15 REAL-LEARNING SMOKE
============================================================

Only after all P0 issues are resolved and the HP scheduling/search configs are
frozen:

run a SHORT genuine F15 learning smoke.

Suggested wall-clock target:

15–30 minutes

but prefer explicit cycles/positions in addition to wall clock.

Requirements:

- standard start
- F15
- CUDA FP32
- real PUCT
- frozen selected search budget
- real search visit targets
- completed game WDL
- streaming replay
- dynamic reuse-based update budget
- persistent optimizer state
- strict checkpoint/lineage
- raw-policy evaluation

Before training capture baseline:

- random-init raw policy
- frozen opening suite results
- self-play data health

After each cycle record the complete metrics.

============================================================
F15 SMOKE GO GATE
============================================================

Do NOT require chess strength yet.

GO requires SYSTEM HEALTH:

- no corruption
- no illegal moves
- no inference errors
- no NaN/Inf
- checkpoint/resume valid
- optimizer continuation valid
- no uncontrolled memory growth

DATA HEALTH:

- sufficient completed games
- truncation acceptable
- termination distribution known
- replay audit clean
- search targets non-degenerate
- no overwhelming repetition pathology without explanation

TRAINING HEALTH:

- requested reuse approximately achieved OR shortfall explained
- losses finite
- policy and WDL loss separately visible
- gradients finite
- clipping behavior visible
- weights actually change
- no obvious collapse

EVALUATION HEALTH:

- raw evaluation completes
- searched arena is marked informative or uninformative based on its outcome
  distribution
- no strength claim from tiny samples

If these fail:

NO-GO.

Diagnose before a longer pilot.

============================================================
F15 QUALIFICATION PILOT
============================================================

Only if the learning smoke passes:

run a bounded qualification pilot.

Suggested:

~45–90 minutes

NOT 24 hours.

Use:

multiple cycles
bounded position budget
bounded replay
frozen scientific config
persistent optimizer lineage

Purpose:

prove actual learning behavior is stable enough to justify the recurrence phase.

============================================================
R15 ENTRY GATE
============================================================

Do NOT implement/train R15 in this phase.

At the end, classify:

R15 GO
R15 CONDITIONAL GO
R15 NO-GO

Minimum R15 GO evidence should include:

- F15 systems stable
- F15 data non-degenerate
- replay reuse under control
- learner stable enough to interpret
- raw-policy signal not catastrophically degrading
- evaluation machinery trustworthy
- optimizer/checkpoint lineage correct
- recurrence-control interpretation explicitly decided

============================================================
RECURRENCE EXPERIMENT DESIGN — PLAN NOW, DO NOT RUN
============================================================

Produce the proposed next-phase design.

Explicitly address the F15/R15 confound.

I want a clear answer to:

WHAT IS THE CLEAN RECURRENCE CONTROL?

Consider:

A.
F15 vs R15 R1

B.
R15 R1 vs R15 R2 vs R15 R4

C.
the SAME trained R15 checkpoint evaluated with different recurrence counts

D.
separately trained R15 recurrence conditions

E.
multi-R training where recurrence is sampled per batch

These answer different questions.

Do not blend them.

For example:

same-checkpoint R1/R2/R4
tests inference-time iterative refinement.

separately trained R1/R2/R4
tests training + architecture adaptation.

multi-R training
tests whether one network learns scalable internal compute.

F15 comparison
tests shared/recurrent architecture against a conventional feed-forward control.

Design a ladder of experiments that isolates these.

============================================================
R15 TRAINING INFRASTRUCTURE GAP
============================================================

Current LearnerConfig appears to have one fixed:

recurrence: usize

for a training segment.

If the eventual experiment needs recurrence sampled from:

R ∈ {1,2,4}

per physical/effective batch, that infrastructure does not yet exist.

PLAN it for the NEXT recurrence phase, not H1 implementation unless required by
a correctness fix.

Need deterministic recurrence schedule / sampling with provenance.

Potential distributions should be treated as experimental choices, not magic
defaults.

============================================================
MATCHED-DATA EXPERIMENT — PLAN
============================================================

Before online F15-vs-R15 feedback loops, design a frozen-data control:

1. generate a fixed Recur64 self-play corpus
2. freeze it
3. train F15 and R15 conditions from controlled initialization strategies
4. same examples
5. same effective example count
6. same optimizer family
7. same scientific schedule where applicable
8. multiple seeds when affordable

This isolates architecture/training effects from differences in self-play data.

============================================================
MATCHED-COMPUTE EXPERIMENT — PLAN
============================================================

The eventual central experiment remains:

Does internal recurrent computation outperform spending the same compute on
external search?

Plan accounting for:

F15 / R15 R1 / R2 / R4

Primary fairness should likely include:

MATCHED WALL CLOCK PER MOVE

and separately:

MATCHED NEURAL COMPUTE

Track:

- GPU inference time
- executed transformer blocks
- number of neural evaluations
- CPU search time
- total move wall time

Equal search-node count is NOT the primary fairness criterion because recurrence
changes evaluation cost.

Do not run this yet.

============================================================
ARENA STATISTICS
============================================================

Current arena has a simple per-game normal CI.

Review whether this is sufficient when games are paired by opening/color.

For early diagnostics it may be acceptable.

For headline recurrence conclusions, plan a stronger paired statistical method:

- pair-level outcomes
- bootstrap over opening pairs
or another defensible method

Do not over-engineer it during H1 unless necessary.

============================================================
REPLAY CRASH SAFETY
============================================================

Review replay capacity archiving for unattended future runs.

Current flow appears capable of moving old shard files before atomically
rewriting the manifest.

Ask whether a power loss between those operations could leave the old manifest
pointing to files that have already moved.

If yes:

design a crash-consistent archive transition before any 24-hour run.

This is P2, not necessarily required for the short F15 smoke.

============================================================
LONG-RUN STATUS
============================================================

24-HOUR RUN:

NO-GO during H1.

Do not create/start one.

Future long-run gate must include:

- stable optimizer resume
- crash-safe replay
- bounded disk use
- sustained thermals
- VRAM stability
- remote recovery
- Windows restart recovery
- frequent atomic checkpoints
- proven pilot learning health

============================================================
DOCUMENTATION CLEANUP
============================================================

Audit:

docs/HP_EXPERIMENT.md
docs/HP_CHANGES.md
docs/F10_BASELINE.md
docs/STATUS.md
README.md

The HP branch now legitimately contains main Phase 3 history.

Ensure docs distinguish:

MAIN F10 RESULT

from:

HP F15 RESULT.

Do not accidentally present workstation F10 measurements as HP measurements.

Also inspect:

Grok-plan.md

It currently appears empty.

If it is accidental/unnecessary, recommend removing it.
Do not preserve empty artifacts merely because they exist.

============================================================
DO NOT CHANGE THESE CONTRACTS
============================================================

Unless you find a demonstrated correctness bug:

DO NOT change:

Observation V1
Action V1
Rules Profile V1
legal move semantics
WDL perspective
PUCT backup sign convention
Replay V1 meaning
checkpoint contract versions

Do not add:

Gumbel
diffusion
SSRL
geometric attention
Stockfish training labels
LC0 training labels
human game corpus
BF16
distributed training

during H1.

============================================================
NO SILENT FAILURES
============================================================

For the next phase:

invalid config → visible error

invalid opening → visible error

missing requested opening suite → visible error

incompatible checkpoint → visible error

replay corruption → visible error

OOM → visible failed cell/run

unsupported precision → visible error

failed health gate → explicit NO-GO / CONDITIONAL GO

Never silently substitute another experiment.

============================================================
TEST REQUIREMENTS
============================================================

Plan targeted tests for at least:

1. collector concurrency != total-game count
2. pilot uses authoritative collector
3. self-play health metrics survive into pilot report
4. dynamic reuse/update arithmetic
5. optimizer survives cycle promotion/resume
6. LR schedule survives cycle promotion/resume
7. correct lineage parent IDs
8. git revision in lineage
9. promotion cannot pass solely because of draw-only 0.5 arena
10. invalid opening FEN fails
11. missing configured opening suite fails
12. accumulation metrics average across microbatches correctly
13. scientific hash stable under hardware-only scheduling changes
14. resolved/full hash changes when scheduling changes
15. F15/R15 current parity test claims documented accurately
16. cancellation/interruption remains recoverable
17. Phase 0/1/2 contracts do not regress

Run existing tests after implementation.

No stubs.
No mocks replacing real production paths.
No placeholders.
No disabled tests used to fake completion.

============================================================
PERFORMANCE PRINCIPLE
============================================================

Do not optimize for:

GPU utilization %

Optimize for:

useful legal training positions / wall time
at healthy search/data quality.

A 100% GPU workload producing repetitive garbage is worse than a 70% workload
producing useful targets.

============================================================
PLAN DELIVERABLE
============================================================

Before modifying anything, produce:

SECTION 1 — CURRENT BRANCH STATE

Exact commit / branch / ancestry.

SECTION 2 — MERGE REVIEW

What main Phase 3 contributed and what HP preserved.

SECTION 3 — CONFIRMED BUGS

Only issues supported by source.

For each:

severity
file/function
evidence
impact
fix

SECTION 4 — HIGH-CONFIDENCE RISKS

Keep distinct from confirmed bugs.

SECTION 5 — SCIENTIFIC CONFOUNDS

Especially F15 vs R15-R1.

SECTION 6 — P0 PRE-F15 FIXES

Exact files and tests.

SECTION 7 — P1 QUALIFICATION IMPROVEMENTS

SECTION 8 — P2 PRE-24H HARDENING

SECTION 9 — COLLECTION/CONCURRENCY REDESIGN

SECTION 10 — REPLAY/REUSE DESIGN

SECTION 11 — OPTIMIZER/CHECKPOINT CONTINUITY

SECTION 12 — PROMOTION/EVALUATION DESIGN

SECTION 13 — LINEAGE/PROVENANCE FIXES

SECTION 14 — HP BATCHING SWEEP

Exact proposed matrix and stop conditions.

SECTION 15 — SEARCH-BUDGET SWEEP

32/64/128 or justified alternative.

SECTION 16 — F15 LEARNING SMOKE

Exact bounded design.

SECTION 17 — F15 QUALIFICATION PILOT

Exact bounded design.

SECTION 18 — R15 ENTRY GATE

SECTION 19 — FUTURE RECURRENCE EXPERIMENT

R1/R2/R4 methodology.

SECTION 20 — MATCHED-DATA CONTROL

SECTION 21 — MATCHED-COMPUTE CONTROL

SECTION 22 — TEST PLAN

SECTION 23 — DOCUMENTATION CHANGES

SECTION 24 — RISK REGISTER

SECTION 25 — IMPLEMENTATION TICKETS

For EVERY ticket provide:

ID
OBJECTIVE
RATIONALE
FILES
IMPLEMENTATION
TESTS
METRICS
DEPENDENCIES
FAILURE MODES
GO CONDITION
STOP CONDITION

SECTION 26 — EXECUTION ORDER

Give exact sequence.

SECTION 27 — FINAL GATE TREE

At minimum:

H1 HARNESS GO
↓
HP SYSTEMS SWEEP GO
↓
SEARCH-BUDGET GO
↓
F15 SMOKE GO
↓
F15 PILOT GO
↓
R15 ENTRY GO

SECTION 28 — EXPLICIT DEFERRED LIST

============================================================
REVIEW THESE SUSPECTED ISSUES — VERIFY, DO NOT ASSUME
============================================================

I have independently reviewed the branch and suspect:

1. pilot uses a duplicate collection implementation and loses HP
   SelfPlayMetrics/inference metrics.

2. `active_games` currently controls both total game count and concurrent thread
   count.

3. `cpu_workers` may no longer constrain pilot concurrency.

4. `replay_reuse_target` is measured but does not determine update count.

5. optimizer state is reset at each pilot cycle while LR schedule continues.

6. lineage parent identity can be mutated before the lineage record is written.

7. lineage git revision may remain `None`.

8. conservative promotion can promote a draw-only candidate because 0.5 >= 0.35.

9. opening/FEN failures may silently fall back to startpos.

10. accumulation metrics may represent only the final microbatch rather than the
    full effective batch.

11. full config hash mixes scientific and hardware scheduling identity.

12. bench-runtime's reported peak VRAM may only be boundary samples.

13. current HP pilot concurrency=32 may be inappropriate for the measured
    6c/12t HP, given the earlier ~12-worker result.

14. F15 and R15-R1 are not functionally identical controls because input
    injection occurs at different depth locations.

15. the eventual recurrence learner lacks per-batch recurrence scheduling.

CONFIRM OR REJECT EACH ONE FROM CODE.

Do not implement a fix merely because I suggested it.

If one is wrong, say why.

============================================================
GO / STOP PHILOSOPHY
============================================================

A negative result is acceptable.

A failed gate is useful information.

Do NOT weaken a gate to keep the experiment moving.

Do NOT start R15 because it is more interesting.

Do NOT start a 24h run because the machine is available.

We want interpretable evidence.

The target is not:

"make the model run."

The target is:

"build an experimental system whose eventual claim about internal recurrence
versus external search is believable."

============================================================
FINAL INSTRUCTION
============================================================

Inspect the current branch thoroughly.

Challenge both the code and the experimental design.

Produce the complete H1 plan.

DO NOT MODIFY ANYTHING.

STOP AFTER THE PLAN AND WAIT FOR APPROVAL.
