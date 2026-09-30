# Recur64 V3 — Research Plan

Status: P0. Gates below are pre-registered **before any science and before any CONFIRM
exposure**. Where a number depends on a quantity not yet measured (marked ⏳), it is frozen at the
named phase, before CONFIRM, and never after seeing a gated result.

## Thesis (frozen)

> The missing capability is selective acquisition and integration of useful future-state
> information under a constrained compute budget.

Primary question: does the SAME weight set improve as its exact state-query budget increases,
B0→2→4→8→16? Trained at budgets through 8; B16 is an explicit extrapolation diagnostic.

Evidence carried in (V2.5 / HP): recurrence over identical information gave no benefit; real
future information did; bulk reveal + answer-adjacent summaries made the V2 task trivially
decodable; one-pass plateaued (CF M3 .659; KQRvK M3 weakest; 5× unique data no help).

## Primary stress cell (pre-registered)

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
| C | FIXED | same checkpoint, same caps, frozen deterministic non-learned schedule (documented and frozen before CONFIRM; not tuned on CONFIRM) |
| D | RANDOM | seeded; diagnostic only |
| E | ORACLE | replay exact-teacher query schedule through same neural integrator; upper bound, never deployable |
| F | ALL-INFO | raw states of all root successors + all opponent replies, parent/branch/depth structure kept, no counts/labels/summaries; shared query encoder where possible; deliberately not compute-matched; report states supplied. One depth-3 TUNE-only extension allowed, designed before CONFIRM |
| G | EXTERNAL SEARCH | interface designed now; run only after the active-search gate; PUCT on the same net, matched expansions/evals/wall; no Stockfish/Leela/Syzygy/solver |

## Training (`budget_0_2_4_8_v1`)

One weight set; caps sampled from {0,2,4,8}; never 16. B0: root policy loss only. B>0: root
policy loss at the FINAL allowed step only + selector/process loss on query decisions. No
per-step same-target deep supervision. Loss weights configurable and in scientific identity.
Selection on TUNE (policy CE/top-1 + selector health), never train loss. Bounded LR screen on TUNE
over {7.5e-5, 1.5e-4, 3e-4} (exact rule committed before running). Teacher-forced proof traces
first; evaluate the selector on its own choices; ONE pre-registered scheduled-sampling/DAgger
rescue allowed; no RL in V3.0.

## ProofTraceV1 (training-only)

Deterministic minimal/near-minimal adversarial certificate from the exact solver: for a correct
move, all defender alternatives and one attacker continuation each; for an incorrect move, a
refuting reply. A single PV is not a certificate. Stored outside `StatePacketV1`. Inference never
sees mate depth, proof status, labels (except through loss), teacher utility.
Audit against the solver. Report trace lengths and the fraction of problems whose ideal
certificate fits in 2/4/8/16 queries.

### ⏳ Known risk to flag now (owner review before gates freeze at P4)
For M3, a full certificate needs every defender reply at each attacker step; the fraction fitting
in 8 queries may be small. Gate II (+0.10 at B8) and Gate III (+0.05) magnitudes are contingent on
that distribution. **P4 measures it; if the ideal-certificate fit fraction makes a gate
arithmetically unreachable, that is reported to the owner and the gate is re-stated before
CONFIRM, never after.** The hypothesis and direction of every gate do not change.

## Gates (frozen; ⏳ = confirmed at P4/P6 before CONFIRM)

Use ≥3 final seeds if the measured cost projection is reasonable, else stop and report the
projection. Paired position-level bootstrap CIs plus per-seed direction.

- **Gate I information sufficiency (TUNE):** AllInfo − B0 ≥ +0.20 top-1 on KQRvK M3, paired 95% CI
  wholly > 0; secondary sanity AllInfo absolute ≥ ~0.75. Failure → STOP and diagnose; no direct
  jump to active confirmation.
- **Gate II useful same-weight compute:** ACTIVE_B8 − ACTIVE_B0 on KQRvK M3 ≥ +0.10, CI > 0, every
  seed positive. Also report correct mass and CE.
- **Gate III learned selection:** ACTIVE_B8 − FIXED_B8 on KQRvK M3 ≥ +0.05, CI > 0, every seed
  positive.
- **Gate IV compute curve:** B0/2/4/8/16 same checkpoint: top-1, correct mass, CE, entropy, query
  depth, branch coverage, terminal discoveries, compute/wall, VRAM. No per-step significance
  requirement; investigate substantial regressions.
- **Gate V extrapolation:** claim only if ACTIVE_B16 − ACTIVE_B8 > 0, CI > 0, every seed positive.
  A tie = no extrapolation signal (V3 may still succeed). A regression is recorded and
  investigated; no retraining through B16 under this identity (`train_to_16` = new identity).
- **Gate VI stability:** finite outputs; bounded workspace/branch RMS; no silent query failures;
  exact counts; no duplicate edges; root encoder once; query-encoder runs = successful queries;
  parameter count budget-independent; stable lifecycle/VRAM.

## Outcomes (frozen)

- **FULL GO:** I, II, III, VI pass.
- **PARTIAL GO — COMPUTE:** active beats B0, not FIXED.
- **PARTIAL GO — INFORMATION:** all-info beats B0, active does not.
- **NO-GO:** no raw-state information gap, or untrustworthy/unstable execution. No same-identity
  rescue by repeated tweaks.

## Phases

| Phase | Content | Status |
|---|---|---|
| P0 | branch, lineage, architecture, plan | this commit |
| P1 | StateQueryV1 + audit + whitelist | planned |
| P2 | model skeleton, identity, CPU correctness | planned |
| P3 | CUDA/system qualification, compute/VRAM envelope | planned |
| P4 | data/process layer (P25_DATA_V1 verify, ProofTraceV1, V3 TUNE, seal HOLDOUT_C) | needs approval |
| P5 | bounded TUNE screen, freeze recipe | needs approval |
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
  conversion and draw composition; no 24h runs; no job > ~2h without owner approval.

## Stop conditions

Any non-finite/runaway state, query-tool correctness failure, unexpected HOLDOUT_C exposure, a
gate that becomes unreachable (report to owner), projected run cost beyond the ~2h/job rule, or a
NO-GO outcome.

## Integrity rules carried over

Never inspect CONFIRM early; never change a threshold after seeing the gated result; never reuse a
voided run; quarantine invalid runs unread; record every deviation; MEASURED vs INFERRED; "not
demonstrated" ≠ "disproved"; no generalization from fixed mate targets to chess strength; no
inference of conversion from root accuracy; no inference of learned search from more information
alone; no extrapolation claim from a B16 tie.
