# X1 / Chimera V1 - conclusion

Status: **CLOSED FOR RECURRENCE-RESCUE EXPERIMENTS.** This is not a claim that recurrence
is disproved. It is a decision that further small rescues of the V1 loop are not the best
use of the next experiment. Nothing in `HP_X1_*` is rewritten; this file summarises it.
Ledger of every experiment, rule and outcome: `HP_X1_EXPERIMENTS.md`.

## What V1 established (engineering)

- A 16.0M-parameter model with four pathways runs and trains on the RTX 2050 (4 GB):
  T=4 update of 96 positions about 1.4 s, 2.4 GB peak; lifecycle plateaus; the D57 device
  check catches a broken CUDA install visibly.
- Exact-history teacher targets, batched teacher labelling (5x faster, identical labels),
  streaming trainer, fixed candidate widths, position-clustered paired bootstrap, and a
  discipline of pre-registration with hard source-game / FEN disjointness.
- A deterministic native/WebAssembly coprocessor whose two providers are byte-identical
  because they are one source compiled twice.

## What V1 established (science)

1. **CandidateFacts worked.** Exact one-ply facts per legal move moved frozen-suite mate-in-1
   top-1 from about 0.30 to 0.95, reproducibly across seeds. The first version failed at
   gain 1 (bias gap +0.014 logits after 100 updates) because a zero-initialised final layer
   under AdamW cannot reach a useful scale; a gain of 128 made it work. That gain is a symptom
   of a bad injection mechanism, not an architectural constant.
2. **Recurrence never beat a one-pass network** in any pre-registered comparison (E7-E15):
   teacher KL (E7, E11), mate-in-1 (E11b), forced mate-in-2 with 1.5k exact positions (E12),
   with 15k (E13), under deep supervision (E14) and with a bounded latent (E15). Best
   recurrent-minus-one-pass differences on mate-in-2: -0.033, -0.033, +0.033, 0.000, every
   confidence interval containing zero.
3. **A real recurrent-state defect existed.** The latent state grew about 1.3x per thought
   (unbounded residual accumulation); T=8 collapsed to 0.30 mate-in-2 accuracy with higher
   training loss. Deep supervision (0.62) and explicit latent normalisation (0.60) each
   repaired it. Both made recurrence *trainable*; neither made it *useful*.
4. **Same-target deep supervision made later thoughts redundant.** With it, the first
   thought already reaches the network's ceiling (0.600 / 0.600 / 0.608 / 0.608 at T=1-4).
   The network solved the task at its earliest thought instead of refining an answer.
5. **Data mattered, depth did not.** Ten times more exact mate-in-2 data moved the one-pass
   network from 0.60 to 0.79; extra recurrent depth moved nothing.
6. **Static global cross-attention to the compute bank and the visual CNN inside the loop
   showed no causal benefit** in any comparison, and repeating the full square-transformer
   core once per thought is expensive (T=4 costs 1.6x the time of T=1) and has not earned it.

## Why V2 is architecturally different

V1 repeated *the same computation on the same information* and hoped useful reasoning would
emerge. Every extra thought saw exactly what the previous one saw. V2's principle:

> Thought N must have a meaningful opportunity to know or represent something thought N-1
> did not.

V2 encodes the board once, treats it as immutable working context, and runs a small bounded
planner whose available information *grows* with the thought budget: root candidates and their
exact facts, then exact successor positions, then exact opponent-reply sets. The planner state
is renormalised and gated so it cannot grow. The model is trained across budgets with one set
of weights, and it is compared against a one-pass network that is handed all the same
information at once, so that "more information" and "iterative integration" can be
separated. See `HP_X2_ARCHITECTURE.md`.

## What is deliberately NOT being carried into V2

Repeating the square core per thought; an unbounded latent; static global compute-bank
cross-attention per thought; the visual CNN inside the loop; fixed-T-only training;
same-target-at-every-thought supervision; CandidateFacts as `logit += gain * MLP(facts)`; a
new large PUCT-teacher dataset where exact labels exist.
