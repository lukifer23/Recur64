# V5 CUDA D9 mask recovery

This is an engineering recovery, not a learned-reader result. Historical D9
reports remain FAIL and unchanged. Recovery implementation source:
`003d296` (full SHA recorded in subsequent measured receipts).

## First divergence and cause

Fresh independently restored model/AdamW snapshots showed pure clones,
self-exact NORMAL and PROFILE execution, and differing cross-mode execution
(CASE C). The returned-state encoder was the first differing reader boundary:
2048 second-example payload values became zero under profiling, with exact
original states, flags and baseline context/hypotheses. Maximum absolute
payload difference was 2.9452357292175293. This is not harmless FP32 roundoff.

The pinned CUDA allocator pitches the `[2,8]` U8 boolean mask at 16 bytes.
The pinned burn-std 0.21.0 `split_strides` routine mishandles an old trailing
singleton during a subsequent dimension insertion:

| Shape | Expected batch stride | Actual two-step stride |
|---|---:|---:|
| `[2,8]` | 16 | 16 |
| `[2,8,1]` | 16 | 16 |
| `[2,8,1,1]` | 16 | 8 |
| expanded `[2,8,4,256]` | 16 | 8 |

The second fixture consequently reads padding bytes instead of its mask row.
Allocation history and completion fences change the contents of those bytes.
Holding all temporary inputs happens to alter this history; it is not the fix.

Actual CUDA primitive evidence at source 9f3c3d0 records wrong expanded strides
`[8,1,0,0]` and correct single-reshape strides `[16,1,0,0]`. The single-reshape
control is exact in 24/24 trials. Expanding the malformed view before negation
incorrectly masks all 8192 second-example values in 24/24 trials. Negate-first
controls happen to pass for that simple allocation history despite invalid
strides; this does not establish correctness of their padding reads.
Full evidence: `evidence/v5/mask-primitive-9f3c3d0.json.gz`.

## Repair and validation boundary

Replace only the returned-payload mask's two trailing unsqueezes with one
reshape to `[b,qn,1,1]`, followed by the same expansion and mask fill. Boolean
logic, intended model equations, architecture, parameter count, FP32 storage,
configuration identity, optimizer, fences and pinned dependencies are unchanged.
No environment setting or permanent synchronization fence is adopted.

The new independent host-mask forward/backward test covers distinct mixed
rows at Q2/Q4/Q8/Q16. CUDA primitive controls additionally cover mixed masks
and an invalid second row. Fresh full diagnostic, unchanged D9 CPU/CUDA
qualification and release workspace results are PENDING until measured.
No training, drill, DEV evaluation or confirmation evaluation is authorized by
this implementation alone. Native dataset quotas remain a separate blocker.

## Fresh unchanged qualification at the repaired source

Actual CLI CPU and CUDA qualification both PASS at full source
`003d296e094c28fc488cd56ef0944b60299983f9`. Original exact D9 fields are true
on both devices. All nine shapes, 50 resident updates, baseline integrity,
null correction, complete checkpoint/moment restoration and continued AdamW
checks pass. Contract remains `v5_synchronized_execution_profile_v1`; no new
contract or relaxed gate is adopted. Qualifiers ran concurrently with focused
example compilation, so timings are operational engineering observations.
Reports: `evidence/v5/qualification-003d296-{cpu,cuda}.json`.
Standalone graph CLI provenance/refusal receipt also PASS at this source.
Historical qualifications remain byte-identical. Repeated independent replay
and complete workspace results are still pending.

## Independent repeatability and causal controls

All 12 fresh independent CUDA replay pairs are EXACT (three repetitions of
NORMAL/NORMAL, PROFILE/PROFILE and both cross-mode orders); clone purity PASS.
Every recorded difference is zero, including ULP distance, across outputs,
all parameter/payload gradients, post-AdamW parameters, moments and counters.
The corrected primitive passes 72/72 all-valid, invalid-second-row and mixed
controls. The old two-step mask fails all 48 invalid/mixed controls, confirming
the stride cause independently of full model arithmetic.
Receipt: `evidence/v5/cuda-mask-recovery-003d296.json`; full numerical reports
are losslessly compressed, with both raw and compressed hashes verified.
A new execution contract is not required. Historical CASE_C remains recorded;
this is a layout repair, not a retrospectively invented Clone/harness defect.
