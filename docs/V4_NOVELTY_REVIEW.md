# V4 — literature / novelty review

**Status: PRE-REGISTERED (2026-10-02), before V4 implementation.** Scope: does `evidence_belief_v4` materially duplicate
existing work? Method: web search to verify each citation's existence and abstract-level claims (2026-10-02). Abstract-level
only: I did not read full papers, so claims below about internals are limited to what the abstracts state, and are
labelled INFERRED where they go further. A fuller read of the closest works (§2.3, §2.7) is recommended before any
publication-style novelty claim.

Target combination: **root hypotheses + explicit content-causal evidence messages + immutable base belief + learned
counterfactual query value + exact one-edge information acquisition.**

## 1. Verified references
| Ref | Citation |
|---|---|
| MCTSnets | Guez, Weber, Antonoglou, Simonyan, Vinyals, Wierstra, Munos, Silver. *Learning to Search with MCTSnets.* ICML 2018, arXiv:1802.04697 |
| Searchformer | Lehnert et al. *Beyond A\*: Better Planning with Transformers via Search Dynamics Bootstrapping.* arXiv:2402.14083 (ICLR 2024) |
| Recurrent depth | Geiping et al. *Scaling up test-time compute with latent reasoning: a recurrent depth approach.* 2025 (plus looped-transformer line) |
| AlphaZero | Silver et al. *Mastering Chess and Shogi by Self-Play with a General RL Algorithm.* arXiv:1712.01815 |
| NAR | Veličković & Blundell. *Neural algorithmic reasoning.* Patterns 2(7), 2021 |
| I2A | Racanière, Weber et al. *Imagination-Augmented Agents for Deep RL.* NeurIPS 2017, arXiv:1707.06203 |
| Metacontrol | Hamrick, Ballard, Pascanu, Vinyals, Heess, Battaglia. *Metacontrol for Adaptive Imagination-Based Optimization.* ICLR 2017, arXiv:1705.02670 |
| VOC | Russell & Wefald. *Principles of metareasoning.* Artificial Intelligence 49, 1991 |
| BMPS | Callaway, Gul, Krueger, Griffiths, Lieder. *Learning to select computations.* UAI 2018, arXiv:1711.06892 |
| DAD | Foster, Ivanova, Malik, Rainforth. *Deep Adaptive Design: Amortizing Sequential Bayesian Experimental Design.* ICML 2021 |

## 2. Comparisons
### 2.1 MCTSnets (learned search)
- **Shares:** search inside a neural network; learned embeddings instead of hand-written statistics; learns *where* to search; end-to-end gradient training; exact simulator in the loop.
- **Deliberately not:** MCTS skeleton, simulation-based expand/evaluate/backup of a vector embedding, tree-structured backup. V4 has no backup and no tree value.
- **Distinct:** V4's evidence is attached to *root hypotheses* as additive logit deltas over an immutable base belief; the selector's target is realised decision improvement, not RL return through a search procedure. INFERRED: MCTSnets has no base-isolation or zero-content invariant.
- **Risk:** the "learns where to search" claim is *not* novel; only the training signal and structure are.

### 2.2 Searchformer / trace imitation
- **Shares:** transformer, planning-ish domain, exact symbolic teacher.
- **Deliberately not:** imitation of search traces; A\* execution tokens. V4 never trains on a teacher's search order. ProofTrace stays diagnostic.
- **Distinct:** the signal is outcome utility measured by counterfactual execution, not a demonstrated trace.

### 2.3 Recurrent / looped latent reasoning
- **Shares:** extra inference-time compute without new parameters (V4 parameter count is budget-independent).
- **Deliberately not:** recurrence over unchanged information. V2.5/HP evidence showed that gave no benefit; V4's extra compute is *new exact information*, not more iterations of the same state.
- **Distinct:** compute scales with acquired evidence, auditable message by message.

### 2.4 Active perception / value-of-information learning
- **Shares:** the core idea: choose observations by expected benefit to a downstream decision. This is the closest conceptual ancestor (VOI / expected value of sample information; Russell-Wefald VOC; Bayesian experimental design).
- **Deliberately not:** hand-derived analytic VOI, or information gain about a latent variable as the objective. V4's utility is realised root-CE change.
- **Distinct:** a learned amortised predictor of counterfactual CE change for *exact one-edge game-state queries*, trained from detached forks.
- **Risk:** DAD (amortised sequential design) is the same *family* (learned acquisition policy from previous data). Difference: DAD optimises information about parameters via contrastive bounds; V4 optimises decision loss and adds content-causal evidence messages.

### 2.5 AlphaZero / conventional MCTS
- **Shares:** exact rules engine, policy over legal moves, search improves a decision.
- **Deliberately not:** PUCT, visit counts, UCB, minimax backup, self-play RL, value-head tree evaluation.
- **Distinct:** a single root belief is corrected by a handful of exact observations with a learned acquisition rule; no tree statistics.

### 2.6 Neural algorithmic reasoning / message passing
- **Shares:** message vectors, set-like aggregation, permutation-equivariance goals.
- **Deliberately not:** imitating a classical algorithm's intermediate states. Messages carry exact transition content, supervised only through the root decision loss.
- **Distinct:** zero-content ⇒ zero-message is architectural, and messages are targeted at root-action hypotheses.

### 2.7 Closest prior art that must be reported
- **Russell & Wefald VOC / BMPS (Callaway et al.):** formal framework for selecting computations by expected decision improvement; BMPS *learns* computation selection and notes the "complementary computations" problem (one step may have zero/negative myopic value). V4's utility is a learned, one-step instance of VOC with exact observations. **V4 does not claim the VOC idea.**
- **I2A / Hamrick metacontrol:** neural aggregation of imagined/simulated outcomes into a policy, and learned selection of which simulation to run. They use *learned models* for imagination; V4 queries the *exact* engine and gates evidence by content.
- **Verdict:** no source found that combines all five elements of the target combination. Close art duplicates sub-components (VOC-driven selection, learned acquisition, neural aggregation of simulated outcomes). **No material duplicate of the full combination found; proceed.** The honest novelty claim is the *combination* and the *structural guarantees* (base immutability, exact-zero content causality), not any single idea. Absence-of-evidence caveat: web search is not an exhaustive survey.

## 3. What V4 shares / drops / adds (summary)
| Shares | Deliberately not implemented | Appears distinct |
|---|---|---|
| exact simulator in the loop; VOC objective; learned amortised acquisition; set aggregation | MCTS/PUCT/backup; trace imitation; recurrence over fixed info; model-based imagination | root-hypothesis evidence deltas; immutable `z0`; exact-zero content invariant; utility labels from detached counterfactual forks |

## 4. Consequence for the plan
- Frame V4 as testing whether VOC-style learned acquisition plus content-causal evidence works for this exact-query chess setting, not as a new theory of search.
- Add BMPS-style multi-step diagnostics to the Stage C utility report (one-step vs two-step value), because complementary queries are a known failure mode of myopic VOC.
- Controls later (not now): AlphaZero/MCTS and Searchformer-style baselines.
