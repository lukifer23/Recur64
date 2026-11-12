# V500 — learned proof search for deep forced mates: results

**Outcome (pre-registered gates, KRvK): G0 PASS, G1 PASS, G3 PASS, G2 FAIL. P6 (KBNvK escalation): NOT MET.**
A 0.64M-parameter network used only as a move-ordering function inside an exact AND/OR mate prover lets the prover find sound,
independently checked proofs of every KRvK mate up to the 16-move maximum on held-out positions, in at most 36,266 nodes, using
10-20x fewer nodes than the best classical heuristic search. It does not extrapolate for free to mates deeper than it was trained on,
and a first attempt at the hard KBNvK ending failed. Every claim below is on the exact ground truth of a retrograde tablebase built
in this repo (the tablebase is the label source and the grader; it is never used inside the search being evaluated).

Plan: `docs/V500_RESEARCH_PLAN.md`. Ledger: `docs/V500_EXPERIMENTS.md` (V500-E1..E9). Evidence: `docs/evidence/v500/`.
Branch `experiment/workstation-v500-learned-prover`. Code: `crates/recur64-prove` (+ `scripts/v500_*`).

## 1. What was built (all TESTED)
- **Exact tablebase** (retrograde, parallel, flat arrays): KQvK (longest mate 10 moves), KRvK (16 moves = 31 plies), KBNvK (24.5M legal positions,
  10.8M won with white to move, longest mate 33 moves = 65 plies). All three maxima match the known values; the table agrees with the repo's
  independent forward `MateSolver` on thousands of positions (white and black to move). A legality defect (adjacent kings accepted by the
  chess library) was found by that cross-check and fixed.
- **Sound prover**: iterative-deepening AND/OR search with transposition table, pluggable move ordering, an optional width limit
  (incomplete but never unsound) and a node budget; proofs are shared DAGs. An **independent proof checker** re-verifies every returned
  proof against the chess rules only (every defender reply covered, legal moves, mate at the leaves). The evaluation harness aborts if
  any proof fails the checker; none ever did (5,760 attempts in the gated runs plus 17,088 in the exhaustive sweep).
- **Learned orderer**: piece-square tokens + king-relative displacement tokens, summed through an embedding into a 256 -> 64 -> 1 MLP
  (0.64M parameters), trained with burn on the GPU with a **listwise ranking loss** (softmax over each position's children, target uniform
  over the exactly optimal children), deployed as an allocation-free Rust forward pass. The Rust forward matches a burn forward of the exported
  weights to 1.5e-5 (G0).
- 8 tests (`cargo test -p recur64-prove`): known maxima, agreement with the independent solver, optimality at full width, width-limited
  soundness, tamper rejection, symmetry-invariant split, token invariants.

## 2. Gated results (KRvK, frozen thresholds in V500-E3; three training seeds; TEST = held-out symmetry classes)
| gate | rule | measured | verdict |
|---|---|---|---|
| G0 | 0 unsound proofs; Rust forward = burn forward within 1e-4 | 0 unsound; max diff 1.53e-5 on 14,452 validation children, every net | PASS |
| G1 | mean solve rate over depths 11-16 at 1e5 nodes >= 0.546 (reference 0.396 + 0.15), CI above 0, every seed > 0.446 | **1.000** for all three seeds (all 240 deep positions solved) | PASS |
| G2 | EXTRAP nets (trained only to 8 moves): mean over depths 11-16 >= 0.395 and >= 0.75 x FULL | **0.322** | **FAIL** |
| G3 | median nodes to a proof at depths 10-14 <= 0.5 x the best classical config | ratios 0.095 / 0.048 / 0.060 / 0.049 / 0.063 | PASS |

Width 2 (chosen on validation by the frozen rule). Median nodes to a checked proof, learned vs best classical config (KRvK TEST, 3 seeds):
depth 10: 1,174 vs 12,374; 12: 3,185 vs 53,188; 13: 5,284 vs 107,019; 14: 9,289 vs 148,174; 16: about 22,000 vs 957,000.

**Exhaustive sweep (seed 5101, width 2, budget 1e5): all 17,088 won KRvK positions of the held-out classes solved**, maximum 36,266 nodes;
proof length equals the tablebase optimum in 96.0% of positions, mean overhead +0.04 moves, maximum +5. One-sided 95% bound on the failure
rate for this net: about 0.018%.

## 3. Extrapolation (G2) and the honest nuance
Frozen G2 fails: nets trained only on mates up to 8 moves are reliable to about 10 moves and degrade after (width 2: 0.75-0.85 at depth 11,
0.53-0.85 at depth 12, 0.23-0.38 at depth 13). A **post-hoc, non-gated** diagnostic (width and budget chosen after seeing the data) shows
the same nets at width 3 with 1e6 nodes solving 0.754 of depths 11-16 (>= 0.88 through depth 14, 0.40-0.62 at 15, 0.03-0.23 at 16). Reading
(INFERRED): a net trained to 8 moves can prove mates of about 14 moves, nearly twice its training depth, but only by spending more
search; the pre-registered claim of free extrapolation is false.

## 4. KBNvK (P6): not met
Baselines (blind and every classical config) solve 0.000 of the 280 deep positions (optimal depth >= 20) at 1e6 nodes and never solve past depth 14 of 33.
The first learned net, nucleus width, colour-conditioned bishop tokens and a 4x larger net all failed on validation or plateaued at about 0.87
top-1 on exactly optimal moves (v2: 0 of 140 deep validation positions solved at every width; top-p 0.9: 0 of 140). A **perfect-oracle ordering**
at width 1 solves one TEST position per depth from 12 to 33 in 370 to 275,935 nodes, so the node budget is not the obstacle: ranking quality is.
The train/validation gap of the larger net (0.903 vs 0.866) points to data scale (about 6% of the 20M parents were sampled), which needs a much larger
run than this phase carried out. KBNvK is therefore a limit of this phase, not a result.

## 5. What is established, what is not
- **Established (MEASURED, exact ground truth, checked proofs):** on KRvK a tiny learned ordering function turns a sound exact prover into one that solves
  every held-out position up to the maximum depth, 10-20x more node-efficiently than the strongest tablebase-free heuristic I could build, and replicates
  across three training seeds. The mechanism is robust to its own errors: a wrong ranking costs nodes, never correctness.
- **Not established:** free extrapolation beyond the training depth (G2 failed); KBNvK or any four-piece ending; wall-clock superiority (see below);
  anything about positions with pawns, other material or full chess; a comparison against a proof-number search with a strong prior (plain df-pn and df-pn+ with move-count initial numbers were built and run and are
  much weaker than the learned prover: depth 9 / 10 / 11+ at budget 1e6 for plain df-pn, but their initial numbers were not tuned); the contribution of each architectural choice (no ablation of the relative tokens was run; the only controlled comparison is
  the failed value-regression first attempt, which used neither the listwise loss nor the relative tokens, so the two changes are confounded).
- **Wall-clock:** per search node the learned scorer costs about 25 microseconds against about 2 for the classical ordering (after a 3.8x optimisation with identical proofs). A single-thread, quiet-machine timing study (V500-E11; 10 positions per depth, one seed) shows the learned prover solving 10 / 10 at every depth 10-16 in 0.03 to 0.56 s; among classical configurations that solve reliably it is about as fast at depth 10 and about 1.6x to 3.2x faster from depth 11 to 16.
- The classical baseline is a heuristic I wrote; a better-tuned one could narrow the gap. It was frozen before any learned result and given the most
  generous reading (best config per depth) in the gates.

## 6. Process notes (kept visible)
- First learned attempt (value regression, piece-square tokens) failed badly (V500-E4) and changed the design; it is logged, not hidden.
- Retrograde bug (adjacent kings) caught by the cross-check test; a proof-extraction design bug (full tree unfolding, 46 GB) found while diagnosing KBNvK and fixed
  with shared DAG proofs, node counts bit-identical.
- GPU matmul precision made the in-training parity check differ by 1e-2; the frozen G0 parity is the CPU burn forward of the exported weights.
- A pair of concurrent GPU trainings was measured to be slower in total and stopped; the machine slept about ten hours mid-run; the system low-memory guard stopped one
  of my watcher commands (not restarted).
- Build artifacts from `target-opt/` were committed once by mistake (commit `31e0f87`), untracked in `450043f`; a history rewrite was blocked by permissions
  and not performed, so those blobs remain in that one commit.

## 7. Visual summary
A published page (private; share from its menu) presents the gates, both curves, the extrapolation result, the KBNvK limit and a move-by-move replay of a checked 16-move proof: https://claude.ai/artifact/6p87bzRYUwj2kL9FpLoXbX . It is generated from the committed evidence; the page was not visually previewed in this session.

## 8. Next steps (not started)
Scale the KBNvK training data (all parents, not 6%) and add on-policy hard-example mining; a proof-number/df-pn search with the network as the prior
(robust to ranking errors by construction); an ablation of the king-relative tokens and the loss; a quiet-machine timing study.
