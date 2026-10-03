# V5 synchronized engineering accounting

Decision V5-D9: pre-pilot operational protocol, 2026-10-03, under the owner's
next-step delegation. No geometry, weights, precision,
acquisition, optimizer, loss, training recipe or practical pilot gate changes.
The earlier 64c4dd4 functional results remain historical and do not satisfy the
new timing prerequisite. The exact TRAIN file is still absent.

## What is timed

`recur64 v5 qualify` now uses the pinned Burn 0.21.0 `Backend::sync(device)`
completion fence at the end of EVERY AdamW update, including unprofiled matrix
and 50-update resident measurements. A fence error causes a nonzero process
exit, not CPU substitution or a published passing report.

A dedicated disposable-clone Q8/R4 update additionally measures consecutive
synchronized phases through the SAME input builders, frozen graph-free baseline,
returned-state encoder, paired reader, correct-set loss, full backward, gradient
health readback and AdamW path used normally. Callbacks receive phase names only;
they cannot inspect, detach or replace tensors. Both streams and all loop blocks
are charged. Profiling does not add an alternate model or change a checkpoint.

Recorded phases include root legal-list/candidate upload, host allocation,
exact root CandidateFacts, root observation/geometry/fact upload, acquired packet
encoding/relation construction/upload, graph-free model view, root encoder plus
candidate construction/head, constant lifting, returned-state encoding, payload
boundary, initialization and each evidence/hypothesis update in each stream,
paired readout/centering, target/loss/scalar read, backward, gradient health and
AdamW completion. Checkpoint save/load, full contents and forward verification,
and two resumed/uninterrupted continuations are measured separately.

Host preparation and transfers are combined intervals, NOT purported isolated
DMA timings. Completion-fence overhead is INCLUDED. Neither this instrumented
update nor fixed-graph evaluation is an online active-query decision benchmark.
The whole qualifier wall is reported separately from its CPU acquisition interval
and detailed disposable-update wall; adding disconnected intervals is not called
a continuously measured decision latency. Resident wall measurements include
the `nvidia-smi` subprocess where used. Cold compilation and warm work remain
distinguished; no profiling interval is substituted for uninstrumented throughput.

## Exact work and parity

Current Stage B preparation builds V5 inputs twice: once for the reader backend
and once for the graph-free baseline backend. Thus raw packets are prepared and
uploaded twice, and root CandidateFacts is called twice per position. This cost
is reported, not hidden or assigned to Q. Per call, the existing authoritative
CandidateFacts implementation plays one successor board per legal root candidate,
classifies terminals and inspects immediate legal replies. Reply enumeration is
not separately instrumented; no invented transition or reply counts are claimed.
Only one root encoder executes, and only one returned-state encoder executes per
forward computation; the baseline's duplicate raw non-root tensors are NOT encoded.
Removing duplicate host work is deferred, not assumed to be a measured cache win.

Every matrix shape records requested/actual Q, exhaustion, maximum depth,
root-branch coverage, graph digests, valid and padded state/candidate rows, encoder
examples, duplicated host work and `4R` shared-core applications per example
(`4R * physical_positions` for the batch). Q remains successful StateQuery edges.

The synchronized timing contract is `v5_synchronized_execution_profile_v1`;
its operational declaration is SHA256-hashed in the report in addition to the
scientific source and unchanged architecture/configuration digests. Each device
must show exact same-device/same-weight/same-graph equality of logits, centered
deltas, scalar loss, returned-input gradients, EVERY parameter gradient (including
absence markers), and ALL post-AdamW parameters/moments. This is not cross-device
parity. CPU tests additionally exercise unequal Q/padding at R1/R2/R4. Failure
invalidates this execution qualification; it cannot be fixed by loosening a
tolerance or launching training.

Training, drill and pilot-report loaders require the current timing contract and
both exact-parity flags plus observed phase-accounting consistency. Old reports
remain readable historical evidence but cannot authorize current training.
No qualification checkpoint initializes the drill or pilot. Failed/interrupted
unique checkpoint temporaries are retained; successfully verified disposable
temporaries are removed only after their absolute cleanup target is constrained.

## Measurement status

New-source CPU/CUDA measurements: NOT RUN at protocol registration. Compile/API
errors while adding the observer remain visible in V5_EXPERIMENTS. Final measured
reports and test counts will be recorded only after their commands complete.

Measured at committed source d970049: CPU PASS; CUDA FAIL at combined exact
normal/profile output-gradient and AdamW parameter/moment parity. Accounting,
null, baseline and ordinary resume checks pass but do not clear this failure.
The gate remains unchanged; no dataset-dependent work was launched. Full release
suite: 581 passed, zero failed, two preserved ignores, native exit 0. See
`V5_PROFILING_ROOT_CAUSE.md` and source-tagged reports; diagnostic GPU timings
are not a qualified performance envelope.
