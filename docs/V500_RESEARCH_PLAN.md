# V500 — learned proof search for deep forced mates

**Status: PRE-REGISTRATION (2026-11-11), written before any V500 code or result.** Owner granted full authority to pursue the
direction the V3-V420 evidence suggests: single-position networks are not limited by depth or capacity, so test the SEARCH side.
Branch `experiment/workstation-v500-learned-prover` from V420 HEAD `9a93365`. New identity; touches no frozen V3/V3.5/V4/V420 artefact.
`v4_tune_v1` and HOLDOUT_C stay sealed and are never loaded.

## 1. Question and why it is different from V3/V3.5/V4
Can a tiny learned move-ordering/selection function let an EXACT AND/OR mate prover find sound proofs of deep forced mates (up to the full
KRvK / KQvK distances, 10-16 moves) within a small node budget, far beyond what blind search and a strong classical heuristic reach, and
does it do so on mates DEEPER than any it was trained on?

Difference: the network is not a policy that must be right (V2.5-V420) and not an evidence channel (V3-V4). Search stays exact: a proof is
a verified AND/OR tree, so a wrong or noisy network costs nodes, never correctness. The only learned thing is where to look first.

## 2. Components (all real, no mocks)
1. **Tablebase** (`recur64-prove`): retrograde analysis for KQvK and KRvK (white to move and black to move, exact plies to mate, draws),
   validated against the repo's independent `MateSolver` on every position of depth <= 3 attacker moves and against known maxima
   (KRvK 16 moves / 31 plies, KQvK 10 moves / 19 plies; KRvK/KQvK standard DTM tables).
2. **Prover**: iterative-deepening AND/OR search, memoised, with OR-node (attacker) move ORDERING and optional WIDTH LIMIT k (only the top-k
   attacker moves are tried), full enumeration at AND nodes (defender), node budget, and an independent **proof checker** that re-verifies
   every returned tree against the chess rules without using the scorer. Width limits make search incomplete but never unsound.
3. **Scorers** (same search, only ordering differs): BLIND (generation order, full width); CLASSICAL (checks, defender-mobility reduction,
   king proximity, box shrinking; a strong tablebase-free heuristic); LEARNED (a small MLP predicting the exact distance-to-mate of the
   resulting position, trained with burn on tablebase labels, deployed as a plain Rust forward with a tested parity check against burn).

## 3. Evaluation protocol (frozen before any LEARNED result)
- Test positions: held out by symmetry-canonical key from training; stratified by exact optimal depth.
- Metrics per depth stratum and per node budget B in {1e3, 1e4, 1e5, 1e6}: solve rate (a CHECKED proof of mate within N attacker moves,
  N = optimal + 2 allowed slack, recorded), mean nodes, proof-length overhead, wall time.
- Extrapolation: train on positions with optimal depth <= T (T chosen to leave at least the deepest third of each material unseen), test
  on depth > T.
- Seeds: three independent training seeds; every interval is a paired bootstrap over test positions pooled over seeds.

## 4. Gates (thresholds frozen at step P2, after the BLIND and CLASSICAL baselines are measured and before any LEARNED evaluation)
G0 plumbing: tablebase agrees with MateSolver and known maxima; every returned proof passes the independent checker; 0 unsound proofs.
G1 usefulness: at a fixed budget B* chosen from the baselines, LEARNED solve rate on the deepest unseen third exceeds CLASSICAL by at least a margin M
   (to be frozen), CI wholly positive, every seed positive.
G2 extrapolation: LEARNED solve rate on depths > T is not worse than 0.5x its rate on depths <= T at the same budget.
G3 efficiency: mean nodes to proof at matched solve rate is lower than CLASSICAL by at least a factor to be frozen.
KILL: if G0 fails, fix once and log. If G1 fails, V500 stops and is written up as a negative result (the baselines are still a real result).
If G1-G3 pass, escalate (P6): add KBNvK or KBBvK (hard mates) as an unseen-material test, again with a pre-registered rule.

## 5. Phases
P0 this plan. P1 tablebase + tests. P2 prover + checker + BLIND/CLASSICAL baselines, freeze G1-G3 thresholds. P3 training data + burn trainer + Rust
forward parity. P4 LEARNED evaluation (3 seeds). P5 review, ablations (width, depth, features), iterate within the frozen rules and log every deviation.
P6 escalation if gated. Docs, ledger and the remote are updated at each phase.

## 6. Prohibited
Opening any sealed set; unsound shortcuts (every claimed mate is proof-checked); using the tablebase inside the search being evaluated (it is
the label source and the grader only); changing a frozen threshold after seeing LEARNED results (a change needs a new logged identity).
