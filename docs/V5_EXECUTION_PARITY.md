# V5 execution experiment: matched attention softmax

Decision V5-D8, 2026-10-03. Chosen under the owner's next-step delegation,
before any dataset-dependent drill, training or evaluation. This is a bounded
execution-parity correction, not a new architecture or training recipe.

## Retained failure and pinned-source cause

CPU qualification at source `7737640572c158fbda9c4cbadfe1332eb9303abc`
exited 1. Only `graph_free_baseline_matches_reference` was false. Context,
hypotheses and z0 had maximum absolute differences 1.6689300537109375e-6,
2.205371856689453e-6 and 7.078051567077637e-8 respectively. These are real
FP32 value differences, not TensorData metadata differences.

Pinned Burn 0.21 source explains a dispatch difference:

- `burn-flex/src/ops/activation.rs:157` overrides softmax with its optimized
  row implementation (including SIMD reduction paths).
- `burn-autodiff/src/ops/activation.rs` does not override softmax.
- `burn-backend/src/backend/ops/activation.rs:250` supplies the default
  primitive max-shift, exp, sum, division equation; the max-shift is detached.

An isolated fixed nonsaturated `[2,8,64,64]` input reproduces a built-in
graph-free/autodiff maximum difference 7.450580596923828e-9. The explicit
primitive equation on graph-free Flex matches the autodiff result bit-for-bit.
This isolates the operation, but full-model parity is still a required gate.

## Frozen corrective experiment and acceptance

Use the SAME pinned Burn Tensor operations in every V5 attention softmax:

```text
m = max_dim(detach(logits), attention_dimension)
e = exp(logits - m)
w = e / sum_dim(e, attention_dimension)
```

Masks are applied before this equation as before. Production FP32, geometry,
precision settings, weights, parameters, losses, alpha, both-stream gradients,
full BPTT, query schedules and training recipe are unchanged. This is not a
custom kernel or custom autodiff. The only detach is the framework's numerical
max-shift; neither stream nor recurrent state is detached.

The configuration hashes the strict execution identity
`burn_0_21_explicit_detached_max_exp_sum_div_v1`. Old configurations without this
field refuse, rather than silently inheriting a different forward implementation.
Actual model-info execution reports the new configuration digest
`849133a5cdf169f187778bace2f858aa4747d2e8defef3bb5ac1bffc839774ee` and
unchanged 7,162,896 parameters.

Before interpreting this correction as qualified, require:

1. The ORIGINAL full baseline graph-free/autodiff exact equality, without a new
   tolerance. CPU null equality and exact resumed continuation stay unchanged.
2. Same-weight old-dispatch versus explicit-equation autodiff reader logits,
   centered deltas and payload gradients EXACTLY equal at Q2/Q4/Q8 × R1/R2/R4
   on the fixed two real chess fixtures. The old dispatch is available only in
   an isolated test-scoped reference; there is no production switch.
3. Existing four-direction finite differences, masks, recall, feedback,
   permutation, loss and historical refusal tests remain passing.
4. Fresh CPU and intended-device CUDA qualification, including 50 resident
   Q8/R4 paired differentiated updates and complete checkpoint continuation.

If these fail, preserve the failure and stop for diagnosis; do not change
tolerances or train around it. Test/build outcomes belong in V5_EXPERIMENTS and
compact evidence, not inferred from this decision memo.

The failed source report is retained at
`docs/evidence/v5/qualification-7737640-cpu.json`. Its passing null, gradient,
baseline-after-update and complete restoration checks do not make the aggregate
report PASS. No CUDA qualification was launched at that failed source.
