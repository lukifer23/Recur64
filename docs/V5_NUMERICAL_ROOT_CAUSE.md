# V5 finite-difference measurement correction

Recorded 2026-10-03, before any drill, pilot training or DEV evaluation.

After the complete-frontier correction, the unchanged FP32 finite-difference
test failed: direction 0xA502 had relative error 0.1643647402524948 at epsilon
0.05 against the unchanged 0.12 threshold. A bounded diagnostic reported all
four original directions at the fixed ladder 0.0125/0.025/0.05/0.1/0.2. Direction
0xA504 had error 0.3099137842655182 at the original epsilon. No point was selected
as a replacement epsilon. Those failures remain historical engineering evidence.

The candidate-relative correction derivative is about 2e-6 to 8e-6. Its central
FP32 difference subtracts already-subtracted factual/null readouts; the numerator
is only about 1e-7 to 1e-6. The ladder is irregular rather than convergent, which
suggested roundoff, but was not sufficient to prove autodiff correct.

## Same-weight reference evidence

A test-only FP64 reference uses the same pinned Burn Flex backend, the actual
reader code, the exact FP32-initialized weights promoted to FP64, the same graph,
the same four directions, and epsilon 0.05. It is not a new initializer, model
geometry, training backend, or production precision. The original FP32 parameter
digest is checked unchanged after the reference execution.
FP64 here describes parameter/tensor dtype, not every internal operation: pinned
Burn 0.21 RmsNorm explicitly casts its squared-mean statistics to FP32 even for
FP64 inputs, then restores the input dtype. The reference retains this exact
implementation; no fully FP64 RMS reference is claimed. Reduced readout
cancellation, not a backend replacement or rewritten normalization, is tested.

Across directions 0xA501..0xA504:

- FP32 versus FP64 autodiff relative error: 0.00008472903967668629,
  0.00014744318395504692, 0.00001743600712082628,
  0.0001412353174677674.
- FP32 autodiff versus FP64 finite differences relative error: approximately
  0.04229, 0.05259, 0.0002944, 0.05314; all below the original 0.12 limit.
- FP64 autodiff versus FP64 finite differences also passes that limit.

This supports a numerical measurement failure in the FP32 finite-difference arm,
not a missing payload-gradient path. It is engineering evidence on fixed fixtures,
not learned chess reasoning evidence.

The qualifying test now uses FP32 autodiff versus the same-weight FP64 numerical
arm. Its fixture, directions, original FP32 epsilon value (promoted exactly),
nonzero-direction requirement, and 0.12 assertion are unchanged. An additional
test requires FP32/FP64 autodiff agreement below 0.001 and both numerical
comparisons below 0.12. The full FP32 ladder remains an explicitly invoked,
non-qualifying diagnostic; its prior failures are not relabelled PASS.

## Reference-path implementation errors retained

The first FP64 diagnostic attempts failed with a dtype mismatch: the legal mask
was created in the backend's default FP32, and a direction TensorData constructor
also defaulted to FP32. These were reference-path implementation errors, not
measured model qualification failures. Masks now explicitly match the correction
dtype, and the reference direction explicitly casts to FP64. Production FP32
centering is tested exactly against its previous operation, including the real
paired computation. No production dtype, tensor geometry or normalization changed.

The corrected focused release suite passes 28 tests, with one intentionally
ignored, explicitly runnable diagnostic. Full workspace and fresh CPU/CUDA
qualification must be recorded separately before the engineering gate is cleared.
