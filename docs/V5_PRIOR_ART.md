# V5 prior art and claim boundary

Status: initial primary-source review, frozen before results. It will be expanded
only to correct or sharpen attribution, never to change V5 after observing results.

- Universal Transformers share recurrent application of transformer parameters
  and discuss adaptive computation; V5 has fixed R, no halting and separates Q
  from R. <https://arxiv.org/abs/1807.03819>
- Latent recurrent-depth work shares test-time depth scaling with fixed weights;
  V5 instead studies a typed acquired chess graph and paired payload-null
  candidate corrections. <https://arxiv.org/abs/2502.05171>
- Tiny Recursive Models share small shared recursive modules; V5 does not copy
  their puzzle training or supervision. <https://arxiv.org/abs/2510.04871>
- Set Transformer motivates attention over unordered elements; V5 additionally
  carries explicit remappable graph relations. <https://proceedings.mlr.press/v97/lee19d.html>
- MCTSnets learn components of tree search; V5 performs no learned or handwritten
  backup and its pilot acquisition is frozen. <https://proceedings.mlr.press/v80/guez18a.html>
- Learning to Select Computations concerns value-directed computation selection;
  V5 defers its controller and tests only fixed-graph reading.
  <https://arxiv.org/abs/1711.06892>
- ReZero studies residual parameterization at depth; V5 uses a fixed nonzero
  alpha 0.5 and makes no ReZero identity-initialization claim.
  <https://proceedings.mlr.press/v161/bachlechner21a.html>

Recurrence, graph attention, simulator queries, attention pooling and residual
updates are not individually novel. The narrow experimental contribution is the
combination of bidirectional hypothesis/evidence refinement, immutable-input
recall, matched factual/null readout, and independently controlled acquisition
and integration axes. This is not claimed to be a Bayesian update, causal
estimator, general reasoner or strong chess engine.

