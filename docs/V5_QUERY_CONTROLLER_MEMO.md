# V5 future query-controller memo

Status: **DESIGN ONLY / NOT AUTHORIZED TO IMPLEMENT OR TRAIN.** This memo does
not add a controller module, targets, checkpoints or commands.

The earliest eligible follow-up is after reader replication, not after this
single-seed pilot alone. It would freeze the replicated reader before producing
any controller labels so the interpreter and target semantics cannot co-adapt.

For known frontier edge `e`, current acquired information `S`, and charged
remaining horizon `h` in {1,2,4}, the target is

```text
U_h(e | S) = L_set(S) - E[L_set(after e and h-1 further charged queries)]
```

The continuation policy would be frozen before target generation: stable-hash
uniform choice from the complete legal unqueried frontier, with terminal and
exhausted states stopping early. Each probe transition and each continuation
transition consumes one StateQuery unit. Probe/continuation work is reported
separately from deployed-trajectory Q and is never presented as free inference.

Counterfactual branches live in isolated QueryManagers cloned from the same
authoritative state/history. Only the actually selected edge return enters the
deployed trajectory. The proof oracle may label TRAIN outcomes after a probe but
may not select, rank or reveal a frontier edge in the controller's forward input.
No proof label, correct-root marker, mate depth, solver answer or counterfactual
return enters deployed state.

An initial controller may read only the declared known frontier action geometry,
root-relative turn/structure context, current hypothesis state, explicit budget
and horizon. It must not read unseen child content. Separate heads or explicit
horizon encoding would be a new hashed architecture contract. Training data,
loss, estimator variance controls, schedule and gates require a new
preregistration and owner authorization.

Deferred beyond that experiment: joint reader/controller updates,
THINK/QUERY/STOP allocation, adaptive halting, persistent warm starts,
transposition merging, diffusion, multiple experts and self-play.

LEARNED QUERY CONTROLLER NOT TRAINED.
