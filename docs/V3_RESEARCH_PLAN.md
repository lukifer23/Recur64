# Recur64 V3 — Research Plan

Status: P0 + P0 review addendum (2026-09-30). No science has been run.

**Revision note.** The P0 commit (`2cf14b3`) left "frozen" ambiguous: it allowed Gate II/III
magnitudes to be restated after P4. The addendum removes that. See "Meaning of frozen" and
"P4 feasibility rule". Git history holds the earlier text.

## Meaning of frozen (single definition)

A rule, gate, threshold, control or schedule is **frozen** when it is committed to this repository
before the data it governs exists, and it is never edited afterwards. A frozen item may be
**superseded only by a new versioned experiment identity with its own preregistration**. It is
never relaxed, tightened or re-stated in place, and never after the measurement it governs. There
are no provisional items in this document: every number below is frozen as of the commit that
contains it. Quantities that are measured later (for example certificate coverage at P4) are
governed by rules that are frozen now.

## Thesis (frozen)

> The missing capability is selective acquisition and integration of useful future-state
> information under a constrained compute budget.

Primary question: does the SAME weight set improve as its exact state-query budget increases,
B0→2→4→8→16? Trained at budgets through 8; B16 is an explicit extrapolation diagnostic.

Evidence carried in (V2.5 / HP): recurrence over identical information gave no benefit; real
future information did; bulk reveal + answer-adjacent summaries made the V2 task trivially
decodable; one-pass plateaued (CF M3 .659; KQRvK M3 weakest; 5× unique data no help).

## Primary stress cell (frozen)

`KQRvK M3`. Rationale from already-seen V2.5 evidence only (large pool, not data-starved, facts
don't solve it, weakest heavy M3, plateau under 5× data). HOLDOUT_C has not been inspected to
check this. M1 = ceiling control. M2 and other M3 families = secondary. Report pooled, macro-cell
and every cell.

## Data

- Train: `P25_DATA_V1` TRAIN positions (44,332); root labels unchanged; query/process traces derived.
- New V3 TUNE from unused canonical classes (esp. KQRvK, KRRvK), complete exclusion manifest.
- CONFIRM: V2.5 `HOLDOUT_C`, digest
  `4ab951c6edd8dd4f531bb87d2f4373895d1fddb70efdf052b24c09a1b71d87d5`; verify via
  `ProofTargets::load`; add exposure guard; **no evaluation of any V3 model until the architecture,
  recipe, comparison rules and numeric gates are committed.**

## Controls (identical positions)

| | Control | Role |
|---|---|---|
| A | B0 | same checkpoint, zero queries; primary baseline |
| B | ACTIVE B2/4/8/16 | learned selector |
| C | FIXED = `fixed_bfs_actionid_v1` | same checkpoint, same caps, **frozen now** (see below); the Gate III comparator |
| D | RANDOM | seeded; diagnostic only |
| E | ORACLE | replay the exact-teacher admissible-set schedule through the same neural integrator; upper bound, never deployable |
| F | ALL-INFO | separately trained; see "ALL-INFO interpretation" |
| G | EXTERNAL SEARCH | interface designed now; run only after the active-search gate; PUCT on the same net, matched expansions/evals/wall; no Stockfish/Leela/Syzygy/solver |

### Primary FIXED schedule: `fixed_bfs_actionid_v1` (frozen now, before any active result)

Model-independent and label-independent. It reads only the query tree's structure, never root
correctness labels, proof labels, selector scores, network outputs, or any CONFIRM data.

At each step the frontier is ordered by the key
`(parent ply depth ascending, parent discovery index ascending, ActionId ascending)` and the
first edge is queried. This is canonical breadth-first expansion. The schedule never depends on
the budget, so B2 ⊂ B4 ⊂ B8 ⊂ B16 by construction.

Stated consequence, recorded so it cannot be discovered after the fact: when the root has at least
B legal moves, the schedule spends its entire budget on the first B root moves in ActionId order
(depth 1) and never looks at a reply. For KQRvK roots (typically ≥ 16 legal moves) this holds at
every budget through B16. The comparator is therefore deliberately a *non-selective breadth
control*, not a strong heuristic. More sophisticated schedules (e.g. round-robin over root
branches with depth-first deepening, or check-first) may be added later only as **secondary**
controls under distinct identities. They can never replace the Gate III comparator.

## Training (`budget_0_2_4_8_v1`)

One weight set; caps sampled from {0,2,4,8}; never 16. B0: root policy loss only. B>0: root
policy loss at the FINAL allowed step only + selector/process loss on query decisions. No
per-step same-target deep supervision. Loss weights configurable and in scientific identity.
Selection on TUNE (policy CE/top-1 + selector health), never train loss. Bounded LR screen on TUNE
over {7.5e-5, 1.5e-4, 3e-4} (exact rule committed before running). Teacher-forced proof traces
first; evaluate the selector on its own choices; ONE pre-registered scheduled-sampling/DAgger
rescue allowed; no RL in V3.0.

## ProofTraceV1 (training-only, set-valued)

`active_selector_v1` must not learn an arbitrary serialization of a valid proof. ProofTraceV1
therefore stores the adversarial structure, never a fixed visiting order, and exposes for
training only the **set** of admissible next edges at any partial search state.

### Structure (AND/OR proof graph)

For a position with minimal mate depth n and correct root moves `C`:

- **Attacker (OR) node** at remaining depth r: every legal attacker move that forces mate within r
  is recorded as an alternative, each with its exact minimal sub-certificate size. Moves that
  deliver checkmate are leaves.
- **Defender (AND) node**: every legal defender reply is recorded. A certificate must contain all
  of them (an AND node cannot be proven by one reply).
- **Refutation records**, for each incorrect root move m: the set of defender replies after m
  from which the attacker cannot force mate within n−1 (exact solver result), kept as a set.

A *certificate* is a proof tree: choose one alternative at each reachable OR node; include all
replies at each AND node. `Q*(p)` is the minimum edge count of a certificate of a correct root
move, computed exactly by dynamic programming over the solver and cross-checked by the independent
audit.

### Admissible set (the selector target)

For a partial search state S (the set of already queried edges) define

- `r(T, S) = |T \ S|`, the edges of certificate T not yet queried;
- `T*(S)` = all certificates of correct root moves that minimise `r(T, S)` (all ties kept);
- `A_proof(S)` = frontier edges of S that belong to at least one `T ∈ T*(S)`;
- `A_refute(S)` = for every queried child of an *incorrect* root move, its refuting defender
  replies that are frontier edges of S;
- `A(S) = A_proof(S) ∪ A_refute(S)`; if empty, the state is complete and no selector loss applies.

The selector target is the **uniform distribution over `A(S)`**. A separately versioned,
solver-derived efficiency weighting could replace uniform later, only under a new identity and only
with a principled definition. `A(S)` is a function of the *set* S alone: it cannot depend on the
order in which edges were queried, nor on how a generator serialized the certificate. Choosing
admissible edge B before equally admissible edge A is never penalised.

### Audit (P4 gate, fixtures committed with the generator)

- fixtures with several interchangeable defender branches (KRvK / KQvK positions where the lone
  king has two or more equivalent flights): `A(S)` must contain every interchangeable edge;
- order-invariance property test: for random permutations of the same query set S, `A(S)` is
  identical;
- every admissible edge exists in the legal frontier; every certificate verified against the
  exact solver and the independent audit;
- `StatePacketV1` is unchanged and carries none of this. Inference never sees mate depth, proof
  status, labels except through the loss, `A(S)`, or any teacher utility.

Report trace lengths and `C_k` (below) for k = 2, 4, 8, 16.

## P4 feasibility rule (frozen now, deterministic, measured on TRAIN only)

Budget coverage: `C_k(cell)` = fraction of that cell's `P25_DATA_V1` **TRAIN** positions with
`Q*(p) ≤ k`. Measured with the ProofTraceV1 generator. HOLDOUT_C is never used.

**Rule.** The primary experiment is *scientifically qualified* iff

`C_8(KQRvK M3) ≥ 0.25`.

If `C_8(KQRvK M3) < 0.25` the primary experiment is classified
**NOT SCIENTIFICALLY QUALIFIED / BUDGET MIS-SPECIFIED**. Then:

1. P5–P8 are not run under identity `budget_0_2_4_8_v1`.
2. Gates II and III are **not** lowered, re-scoped or re-read. They remain as written and simply
   cannot be tested under this budget.
3. HOLDOUT_C stays sealed and unexposed.
4. Any changed training or query budget (for example a longer budget set) is a **new versioned
   experiment identity with a new preregistration**, which cannot claim that the coverage
   measurement was blind.
5. The measurement, the classification and the decision are recorded in `V3_EXPERIMENTS.md` and
   reported to the owner.

Rationale, recorded as a judgement and not derived: Gate II asks for +0.10 top-1 at B8. If fewer
than a quarter of the stress cell's positions even admit a complete certificate within eight
queries, then reaching +0.10 would require resolving a large share of everything that is
resolvable, leaving no room for an imperfect selector, so a null result could not be read as a
failure of the hypothesis. The 0.25 figure is committed now, before the measurement exists, and is
frozen under this experiment identity. If it is ever to change, it must be superseded by a new
versioned preregistration and experiment identity; HOLDOUT_C remains sealed throughout.
`C_2, C_4, C_16` and all other cells are reported but do not enter the rule.

## Gates (frozen)

Use ≥3 final seeds if the measured cost projection is reasonable, else stop and report the
projection. Paired position-level bootstrap CIs plus per-seed direction.

- **Gate I information sufficiency (TUNE):** AllInfo − B0 ≥ +0.20 top-1 on KQRvK M3, paired 95% CI
  wholly > 0. That is the whole pass condition. The AllInfo absolute top-1 on KQRvK M3 is reported
  as a **diagnostic that does not affect Gate I pass/fail**; 0.75 is only a reference value for
  reading it. Interpreted as in "ALL-INFO interpretation": it shows the model family can exploit
  raw future states, not a pure causal information effect. Failure → STOP and diagnose; no direct
  jump to active confirmation.
- **Gate II useful same-weight compute:** ACTIVE_B8 − ACTIVE_B0 on KQRvK M3 ≥ +0.10, CI > 0, every
  seed positive. Also report correct mass and CE.
- **Gate III learned selection:** ACTIVE_B8 − FIXED_B8 (`fixed_bfs_actionid_v1`) on KQRvK M3 ≥
  +0.05, CI > 0, every seed positive.
- **Gate IV compute curve:** B0/2/4/8/16 same checkpoint: top-1, correct mass, CE, entropy, query
  depth, branch coverage, terminal discoveries, compute/wall, VRAM. No per-step significance
  requirement; investigate substantial regressions.
- **Gate V extrapolation:** claim only if ACTIVE_B16 − ACTIVE_B8 > 0, CI > 0, every seed positive.
  A tie = no extrapolation signal (V3 may still succeed). A regression is recorded and
  investigated; no retraining through B16 under this identity (`train_to_16` = new identity).
- **Gate VI stability:** finite outputs; bounded workspace/branch RMS; no silent query failures;
  exact counts; no duplicate edges; root encoder once; query-encoder runs = successful queries;
  parameter count budget-independent; stable lifecycle/VRAM.

Gates II and III are only testable if the P4 feasibility rule qualifies the experiment.

## ALL-INFO interpretation (control F)

ALL-INFO is a **separately trained**, parameter-matched upper-bound model. It is not the active
checkpoint given more states.

- Shares where practical: the V3 root encoder and root candidate representation, and the
  `query_state_encoder_v1` contract (same architecture; trained weights are its own).
- Input: the raw exact states of all root successors plus all opponent replies (depth 2), with
  explicit parent / root-branch / depth structure. No proof labels, mate counts, reply summaries or
  any answer-adjacent field. Report the number of states supplied. One depth-3 TUNE-only extension
  may be designed before CONFIRM.
- Deliberately not compute-matched.
- **What Gate I can and cannot say.** AllInfo vs B0 establishes *information sufficiency for the
  model family*: a model of this family can exploit raw exact future states on this task. It is
  not a pure causal "information-only" treatment effect, because the integrator, the optimisation
  and the training distribution differ between the two models. Gate I is a precondition for
  interpretation, not evidence about the active selector.
- A same-active-model bulk-information control (the active checkpoint fed all states) is not part
  of critical V3.0 scope. It may be added only if the implementation makes it natural, and as a
  secondary control under its own identity.

## Outcomes (frozen)

- **FULL GO:** I, II, III, VI pass.
- **PARTIAL GO — COMPUTE:** active beats B0, not FIXED.
- **PARTIAL GO — INFORMATION:** all-info beats B0, active does not.
- **NO-GO:** no raw-state information gap, or untrustworthy/unstable execution. No same-identity
  rescue by repeated tweaks.
- **NOT SCIENTIFICALLY QUALIFIED / BUDGET MIS-SPECIFIED:** the P4 feasibility rule failed. This is
  not a NO-GO for the hypothesis; the hypothesis was not tested.

## Phases

| Phase | Content | Status |
|---|---|---|
| P0 | branch, lineage, architecture, plan (+ review addendum) | committed |
| P1 | StateQueryV1 + audit + whitelist + semantic identity | implemented |
| P2 | model skeleton, identity, CPU correctness | done (P2.1 hardened) |
| P3 | CUDA/system qualification, compute/VRAM envelope | done |
| P4 | data/process layer, ProofTraceV1, **feasibility rule**, V3 TUNE, seal HOLDOUT_C | done: C_8(KQRvK M3) = 0.4336, QUALIFIED (V3-D15); P5 awaits owner approval |
| P5 | bounded TUNE screen, freeze recipe | NOT RUN; P4 qualified; needs owner approval |
| P6 | information-sufficiency control (all-info vs B0) | needs approval |
| P7 | primary active training, final seeds | needs approval |
| P8 | ONE CONFIRM on sealed HOLDOUT_C | needs approval |
| P9 | adaptive STOP (`adaptive_stop_v1`) only if earned | — |
| P10 | conversion transfer only if earned | — |
| P11 | short self-play only if earned + owner approval | — |

## Later gated work (not part of the primary claim)

- **adaptive_stop_v1:** ≥30% mean-query reduction vs forced baseline, ≤1pp aggregate and ≤2pp KQRvK
  M3 top-1 loss, no pathological premature stop; report usage by M1/M2/M3 and family.
- **Conversion transfer:** frozen suite, paired starts, `root_player_v1` semantics; ACTIVE beats B0
  by ≥10 converted wins per 128 TARGET starts, CI > 0, no catastrophic HEAVY regression.
- **Self-play:** only after conversion gate; very short, separately pre-registered; watch failed
  conversion and draw composition; no 24h runs; no job that would run longer than 2 hours wall-clock without owner approval.

## Stop conditions

Any non-finite/runaway state, query-tool correctness failure, unexpected HOLDOUT_C exposure, a
failed P4 feasibility rule, a projected job longer than 2 hours wall-clock without owner approval, or a NO-GO outcome.

## Integrity rules carried over

Never inspect CONFIRM early; never change a threshold after seeing the gated result; never reuse a
voided run; quarantine invalid runs unread; record every deviation; MEASURED vs INFERRED; "not
demonstrated" ≠ "disproved"; no generalization from fixed mate targets to chess strength; no
inference of conversion from root accuracy; no inference of learned search from more information
alone; no extrapolation claim from a B16 tie.
