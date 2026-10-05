# Revised V6 P0 contract ? frozen before implementation execution

Owner authorization supersedes the older proposed ticket. Base closure publication:
1b397f465a463d28e46d1b0134afc1b861f428be; V5 remains closed NO_SIGNAL.
New worktree Recur64-v6, branch experiment/hp-v6-branch-backup. V5 source/run bytes
are immutable. This contract authorizes implementation, CPU/CUDA qualification
and exactly two disposable TRAIN learnability endpoints; no DEV model invocation.

## Equations and full-information refinement

Architecture v6_branch_full_information_backup_v1. Width256, eight heads, two
returned board encoder blocks, FFN768, four returned slots, FP32. Copy the pinned
V5 returned encoder equations into a separately owned V6 module; no V5 edit.
The frozen root adapter imports only the exact completed Stage A binding and
lifts graph-free outputs as constants, never registers root tensors with optimizer.

Each acquired state has four encoded slots projected by Linear(1024,256) to x_v.
Structure s_v is Linear(48,256): depth one-hot16, root-attacker/defender one-hot2,
root-frame incoming action geometry11, padded native action path16 (index/65535),
legal/observed/unqueried counts3 (count/256). Dataset IDs/family/depth-class/targets
never enter neural tensors. Legal counts refer only to already acquired states.
Frozen root-candidate hypothesis gets Linear(256,256), giving owner_a.
n_v = x_v + s_v + owner_a. Unknown tokens U_attacker,U_defender are learned256.

A common full-branch summary immediately pools ALL observed n_v owned by a,
separately by attacker/defender turn, plus the appropriate unknown token when
unqueried legal replies remain. Attention scores Linear(256,1) are positive-sign
for attacker, negative-sign for defender, temperature1. Unknown logit includes
log(max(1,total unqueried replies in that turn group)); empty groups receive one
unknown token. Virtual unqueried root branches receive unknown summaries, no x.
I_a = owner_a + MLP([attacker_pool,defender_pool,owner_a,s_root]),
MLP1024?768?256 GELU. This common nonlinear initialization, encoder and head are
identical in both arms. Deep depth-five payload has a direct path to I_a and R1.
Cost: two full-branch Q8 attention reductions and one MLP per legal candidate;
report separately from encoder and repeated backup, no hidden free initialization.

One-pass: head(I_a), with head256?256?1 GELU, final linear deliberately NO BIAS.
Principal: initialize every node with n_v + I_owner, and virtual candidate with I_a.
For each simultaneous shared iteration r, locally pool ONLY observed children
plus a same-turn unknown token weighted by unqueried count, using the same learned
signed score inductive bias. Update h?h+0.1?MLP([h,child_pool,I_owner,s]);
MLP1024?768?256 GELU shared across nodes/candidates and iterations. Terminal nodes
freeze according to PAYLOAD terminal flag. Read candidate virtual states at R1/R4.
No other-branch evidence/state communication before final candidate centering.
R now means refinement of full-information initialization, not information access.
Internal scores are latent compatibility scores, NOT win probabilities or proofs.
Conservative WIN/NOT_WIN/UNKNOWN diagnostic stays outside deployed model inputs.

Factual and null execute identical weights/topology. Null zeros returned anchors
AND payload terminal flags; structural counts/path remain matched. Raw ?=headF?headN,
logits=B0+??mean_legal(?). All-null factual and null are exact identical ? exact B0.
Terminal/check/draw/rule status flags and board observations are returned PAYLOAD;
no second terminal Boolean in structure. Rule-derived legal/unqueried counts and
acquisition topology remain STRUCTURE, so shuffle does not claim removal of all
board-derived information. A separate turn-blind structural intervention keeps
payload fixed and changes ONLY defender pooling sign to positive; report it as
structural inductive-bias dependence, never interpret all-null as structural benefit.

## Eligible supervision

Policy correct-set loss covers every legal action. Eligible branches contain ?1
acquired returned payload; use root minimum-mate correct-set membership targets.
For each root P=eligible?correct and N=eligible?correct. BCE(?,1) averaged over P;
BCE(?,0) averaged over N; average available class means, zero if neither exists.
Composite=policy loss+0.5?mean_executed_iteration(eligible BCE). No invented
positives, no labels in acquisition. Report P/N counts and roots with no acquired
correct branch separately. Unqueried branches have identical F/N and exact ?=0.

## Frozen Q8 acquisition

Seed0x7A60_E001. Exploit policy selects up to two B0-ranked legal root actions.
Broad policy selects up to two B0-ranked plus up to two remaining legal actions
by SHA256(contract+NUL+seedLE+rootID+ActionIdLE), tie ActionId. Rank B0 ties by
ActionId; finite logits required. Small action sets take available unique actions.
Query selected roots in ranked-then-hash order; round-robin their branches.
In each branch choose shallowest observed node with unqueried legal action,
then lexicographic native path; actions stable hash(contract,seed,rootID,path,action).
Skip exhausted/terminal nodes; never inspect labels/reader outputs. If all initial
branches exhaust, add next remaining root action by stable hash (including exploit
fallback); stop Q8 or complete depth16 frontier exhaustion. All state/history/action
transitions use QueryManager. Persist source/config/graph digests; same packet
bytes for both readers, including legality/count/unknown history metadata.
Measure both policies on the OLD frozen216 TRAIN panel before fitting, by cell
and B0 wrong/right; correct branch presence only offline reporting, no policy tuning.

## Qualification and paired disposable experiments

Each arm total including immutable root?8M, device-wide peak?3.2GiB. FP32 MB2.
Not compute matched. Future four-layer untied control remains a SPECIFICATION,
not a scaffold: same full summary/interface, independent local layers and matched
receptive field; measured parameter/FLOP/latency accounting before future science.

Qualification: source/custody/import refusal, exact inventory, gradient present/
intentionally absent names, deep dependency, target independence, branch isolation,
unknown counts, loss numeric references, null/B0, independent checkpoint-replicas,
clone purity, exact normal/profile output/gradient/postAdamW parameters+moments,
50 resident updates, model/moment restore and exact continuation CPU/RTX2050.
No V5 GPU probe rerun. Failure stops progression, no fitting on a failed gate.
Formatting/scopedClippy/CLI boundaries/focused+full release tests/serialCUDA build.

Frozen96 panel: KQR/KRR M2/M3,12 B0 wrong+12 right/cell; hash contract
v6_competent_base_drill_select_v1 seed0x7A60_D101, B0 strata once, no substitution.
Freeze exact IDs/strata/graphs/episode bags before BOTH invocations. Seed6300,
200updates, batch24, physical2, accumulation12, peak1e?3,warmup20, fixed existing
warmup/cosine and AdamW contracts. Fresh disposable optimizer per arm, shared-shape
initial tensors identical. Alternate policies by microbatch; same deterministic
root bags and six exposures per each policy per update. ?45min process chunks,
validated resume, ?2h fit per arm. No DEV, no extra steps or performance retry.

Endpoints0/200 both arms, complete BOTH before comparing performance; integrity
failure stops remaining execution. Count ROOTS, not policy rows: corrected root
must be correct under BOTH policies (from B0 wrong); harmed root is wrong under
EITHER policy (from B0 right). Require?12/48 corrected and?2/48 harmed. Also
require finite/exact B0/null,?20% mean policy loss reduction unless start<0.05,
shuffle?real loss?0.05 after averaging policies within each root then roots,
and?6 real-corrected roots losing correctness under shuffle in BOTH policies.
Report each policy's raw counts and paired counts. Shuffle same cell/turn,
exact depth else minimum-distance same-turn, seed0x7A60_E002 stable donor keys;
no self mapping, no labels in donor choice, unresolved donor STOP. Complete
four-slot payload+flags replaced, all recipient structure preserved.
Loss components/margins/ranges/direction/eligibility/payload sensitivity are fixed
endpoint diagnostics, memorization evidence only. No reuse of disposable weights.

Both useful: propose held-out same-information recurrence/compute campaign.
One-pass only: evidence learnable, this recurrence has not earned science.
Principal only: comparator fidelity review before superiority claim.
Both fail: stop, separate information/learning/engineering limits. No outcome
licenses800updates/6301..6303/DEV/CONFIRM/controller/selfplay or V5 rescue.
V5 primary750 vs pooled4500 discrepancy and retrospective report remain unchanged;
future primary predicate is KQRvK M3 n750 paired-policy position averaging.
