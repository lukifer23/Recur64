# V3 P6 plan: ALL-INFO information-sufficiency control and Gate I

**Status: PRE-REGISTERED (2026-10-01). Frozen before any trained ALL-INFO model is evaluated on V3_TUNE_V1.**
Decisions: V3-D22 (baseline replication), V3-D23 (ALL-INFO model and input contract), V3-D24 (training recipe and
layout rule), V3-D25 (Gate I operationalisation). Contract digest of this plan's recipe:
`4d95dda0e87d067b25659166fbfe6d1e0c6f246c80aace92b056b2af62f6b110` (`docs/evidence/v3/v3-p6-recipe.json`).

What was and was not known when this was frozen is stated in section 12.

## 1. The question

> Can a model of the V3 family exploit raw exact future-state information on KQRvK M3 when information acquisition
> itself carries no proof-derived selection signal?

This is Gate I. It does not test learned search, it does not test whether ACTIVE selects states well, and it does not
authorise CONFIRM, Gate II, Gate III, DAgger, P7, B16 or HOLDOUT_C. P6 stops after Gate I.

## 2. Accepted inputs (not reopened)

- P5 selected peak LR `3.0e-4`; selected recipe digest without seed
  `a069ba9d18befed65f970aca253b47780365fd7019be38283f270d79d6c1db33`; P5 contract
  `105ac3133877f954ed00e6ce9caaadabf6d5da1cf78a7ab99d44a195d03009d2`.
- TRAIN P25_DATA_V1 `3b25dc85...f2e6` (44,332 positions); TUNE V3_TUNE_V1 `c6601865...1b53` (4,500 positions).
- HOLDOUT_C `4ab951c6...d5d` stays sealed: P6 never loads it, builds states from it, trains on it or evaluates it. The
  loaders call `load_working_split`, which refuses it.

## 3. Input contract `all_info_depth2_v1`

For every root position the model receives:

1. the normal V3 root observation, root legal candidates and root `CandidateFactsV1` (exactly what ACTIVE receives);
2. every exact root successor state (depth 1);
3. for every non-terminal root successor, every exact opponent reply state (depth 2).

All states are obtained through the same exact `StateQuery` tool (`QueryManager::query`), one query per supplied state,
and the builder asserts `successful_queries == states supplied`, so nothing is skipped. No pruning, ordering heuristic,
sampling, top-K or truncation exists; a position that the tool cannot represent is a visible error. Terminal depth-1
states carry no replies. Depth-3 states are never expanded.

Per-state raw fields used: observation (64 x 119), terminal, in-check, incoming `ActionId`/own legal actions (to build the
incoming-edge embedding), explicit parent, root branch, depth. **Never** supplied: descendant `CandidateFacts`, mate depth,
`Q*`, proof status or trace, winning-move flags, correct-root labels, solver scores, mate/win counts, tablebase values,
PUCT statistics, neural values, best move, teacher admissibility, or any aggregate saying whether a branch wins. Root
policy targets are training labels only. A compile-time exhaustive destructuring of `StatePacketV1` (test) breaks the
build if any field is ever added to the packet.

TRAIN-side census (`docs/evidence/v3/v3-p6-state-census.json`, 44,332 positions): future states per position min 21,
median 140, p90 220, p95 244, p99 288, max 353, mean 147.2 (depth-1 median 37, max 60; depth-2 median 103, max 303);
6,523,727 exact state transitions in total; terminal depth-1 fraction 4.46%; the largest legal list of any depth-1 state
is 8 (the tool's cap is 256). No state-count cap exists, so no difficult position is excluded.

## 4. Model `all_info_v1`

A separately trained model (its weights are its own). It shares, as module types and contracts, with `active_search_v3`:
the V2.5 root encoder (`v25_root_encoder_v1`: width 640, 10 heads, FFN 1280, 8 blocks, executed once), the root candidate
tokens (`candidate_token_v3_root_v1`, dim 256, 1 candidate block, root facts only), `query_state_encoder_v1` (256 wide, 4
heads, 2 blocks, ONE shared instance for every supplied state in a single batched call) and the V3 root readout function
(`root_policy_v3_v1`). It replaces the selector and planner by a set integrator (`all_info_tree_integrator_v1`):

1. each future state becomes a token `[own pooled state, parent pooled state, incoming-edge embedding, root candidate
   token of its branch, terminal, in_check]` projected to 256 and added to a depth embedding (depth-1 vs depth-2);
2. per root branch, one masked set-attention block (4 heads, FFN 768) mixes that branch's tokens;
3. an attention pool, whose query is the branch's root candidate token, summarises the branch;
4. one masked set-attention block (4 heads, FFN 768) mixes the branch summaries;
5. the readout scores every legal root candidate from `[root token, branch summary, mean of valid summaries]`.

No positional encoding of reply order exists anywhere, so serialisation order cannot become an answer signal (test:
reversing and rotating the replies of every branch leaves the policy unchanged, max |diff| < 1e-4). Within-branch
attention and cross-branch attention are the only places states interact; there is no hand-written minimax backup.

**Parameters (`docs/evidence/v3/v3-p6-model-info.json`):** 30,842,524 total (root 27,600,275; query encoder 1,235,208;
integrator 1,842,688; readout 164,353) against ACTIVE's 30,853,790: difference
-11,266 (**0.0365%**, limit 0.5%). The match comes from the natural FFN width 768 of the two set blocks; there are no
dummy or inert padding parameters. Every parameter tensor (236) receives a finite non-zero gradient in a real
full-geometry update, except two documented mathematically inert kinds: the WDL head (no WDL loss in P6) and attention
key biases (a constant added to every key cancels in the softmax; their gradient is exactly zero, asserted at noise
level only). The pooling key projection has no bias.

**Same-seed initialisation is NOT identical across architectures** (Burn initialises linear layers lazily, in module-field
order, so the backend RNG stream a shared module sees depends on what was built before it). This is stated rather than
engineered around: ACTIVE's construction is frozen. Paired seeds label the comparison; they do not equalise
initialisation. Within ALL-INFO, one seed gives one initialisation (tested).

**Isolation.** `Architecture::AllInfoV1` carries its own optional config block and `AllInfoContracts` (skipped when absent,
so every historical and V3 scientific hash is unchanged), its own checkpoint contract check, and `refuse_all_info` at
every historical command. The five-architecture ordered-pair refusal matrix, a tampered-contract test and a
cross-architecture load test pass; the real binary refuses `bench`, `v25-qual`, `proof train`, `proof eval` and `v3-qual`
on an all_info config before any output is created. Configs: `configs/v3/all-info-v1-{cpu,cuda}.toml`.

## 5. Training recipe (no P6 LR screen)

Peak LR 3.0e-4 (selected in P5); 800 optimizer updates; warmup 80; linear warmup + cosine (`lr_at`); `adamw-v1`
(AdamW, beta 0.9/0.999, eps 1e-5, weight decay 1e-4, per-parameter L2 clip 1.0); FP32; one `cell_balanced_v1` sampler over
the 15 P25 TRAIN cells (seed `mix(run_seed ^ mix(0x5A3D6000))`); effective batch 128; loss = exact root-policy
cross-entropy over the correct set, summed over the update and divided by 128 (selector loss 0, WDL loss 0, ProofTrace
process targets unused); no early stopping, no best checkpoint, no intermediate TUNE look. Final update 800 is the model.
TUNE is evaluated exactly once per ALL-INFO model, after update 800. Seeds {5101, 5102, 5103}. Checkpoints alternate
every 50 updates and are strictly validated on resume (same invariants as P5.1). Any external interruption uses the exact
recipe resume and is recorded in the run provenance.

## 6. B0 comparator

B0 means the selected ACTIVE V3 recipe evaluated at zero queries (the P5-selected recipe, update 800).

- Seeds 5101, 5102: the accepted P5 final checkpoints (`v3-p5-run-lr3e-4-seed{5101,5102}/final`).
- Seed 5103: one additional ACTIVE run under the exact selected recipe, identity `p6_baseline_replication_v1` (V3-D22),
  trained fresh and uninterrupted before this plan was frozen (summary
  `docs/evidence/v3/v3-p5-run-lr3e-4-seed5103.json`, recipe digest `92384763...b0f4`, 800 updates, exposure 25,600 per
  budget). It is not a screen run and not a Gate II/III or P7 result.
- `v3-p6 b0-reference` loads each checkpoint through the strict `Trainer::load`, checks LR, seed, update 800 and the
  recipe digest, evaluates B0 on all 4,500 TUNE positions with per-position results, and refuses unless the B0 pooled and
  per-cell metrics reproduce the committed update-800 evaluation to 1e-4.

## 7. Gate I (frozen; unchanged from the research plan)

`Gate I passes iff ALL-INFO - B0 >= +0.20 top-1 on KQRvK M3 (V3_TUNE_V1, n = 750) AND the paired 95% position-level
bootstrap CI is wholly above zero.` Nothing else enters pass/fail. The ALL-INFO absolute top-1 of 0.75 is a reference
diagnostic only.

Exact three-seed paired estimator: for each KQRvK M3 position `i` and seed `s`,
`d[i,s] = 1(AllInfo_s correct on i) - 1(B0_s correct on i)`; `d_i = mean_s d[i,s]`; `Delta = mean_i d_i` (equal to the mean
of the three paired seed deltas). Bootstrap: resample the 750 positions with replacement keeping all three seed pairs
together; 20,000 resamples; generator SplitMix64 seeded `0x7A160001`; index = high 64 bits of `next_u64 * 750`;
percentile interval = the sorted resample means at 0-based ranks 499 and 19499 (2.5% and 97.5%). Pass iff
`Delta >= 0.20` (inclusive) and `CI_lower > 0`. There is no "all seeds positive" condition (the frozen rule has none);
each seed's delta and the seed range are reported prominently as a stability diagnostic. The gate command refuses to run
if its output already exists (applied once) and refuses per-position inputs that are not V3_TUNE_V1 in dataset order or
that carry another recipe digest.

## 8. Secondary diagnostics (reported, never gating, no statistical correction)

Train policy loss; TUNE B0 and ALL-INFO policy loss/mass/entropy (gate cell and pooled; per cell, family and mate depth in
the per-seed summaries); parameter counts; number of raw future states supplied; training wall. Gate I is explicitly not a
pure causal information-only effect: the integrator, the optimisation and the training distribution differ between the two
models. A same-architecture root-only ALL-INFO ablation is not part of this plan.

## 9. Interpretation (fixed in advance)

- **Pass:** record `GATE I PASS - RAW FUTURE-STATE INFORMATION IS SUFFICIENT FOR THIS MODEL FAMILY ON V3_TUNE_V1`.
  Narrow meaning only: ALL-INFO can exploit answer-free raw depth-2 future states. It does not prove learned selective
  search, Gate II, ACTIVE beating FIXED, or CONFIRM. Given the P5 selector exposure gap it would be a rationale to
  consider the one preregistered DAgger rescue; that rescue is not implemented or run here.
- **Fail:** record `GATE I FAIL` and stop. No DAgger, P7, HOLDOUT_C, ALL-INFO depth change, answer summaries, longer
  training or new LR. A failure makes the P5 teacher success more suspicious as query-pattern leakage or indicates the
  raw-state integrator is inadequate, and per the research plan there is no direct jump to active confirmation.
- Depth-3 remains a separately versioned future identity, never a same-identity rescue.

## 10. Systems rule and resolved layout

Frozen fallback ladder of physical layouts, all with effective batch 128: micro16 x accum8, micro8 x accum16, micro4 x
accum32, micro2 x accum64. The first that runs correctly, stays within 95% of device VRAM, has a stable resident-VRAM
plateau (<= 5% spread after warm-up) and projects under 2 hours per run is chosen, from TRAIN-side evidence only; if none
qualifies, stop and report. Effective batch and objective never change.

**Resolved: micro16 x accum8** (`docs/evidence/v3/v3-p6-cuda-preflight.json`, TRAIN only, CUDA FP32, RTX 2000 Ada 16,380 MiB):
steady 7.43 s/update; tree building 0.037 s/update (parallel across cores, so no cache is warranted and none is used);
about 16,200 supplied states per update, at most 2,276 in one microbatch; peak VRAM 11,089 MiB (67.7%), resident
plateau 9,425 MiB stable; GPU busy 77% mean; all 236 parameter tensors covered; projection 1.76 h per run (6,332 s: train
5,948 + one TUNE evaluation 144 + 16 assumed checkpoints 240). Three runs project to about 5.3 h. The remaining ladder
entries were not needed. The workload is GPU-bound, unlike P5.

## 11. Files and evidence

Code: `crates/recur64-model/src/all_info/`, `crates/recur64-runtime/src/p6/`, `crates/recur64-cli/src/v3_p6.rs`. Evidence
(`docs/evidence/v3/`): `v3-p6-recipe.json`, `v3-p6-state-census.json`, `v3-p6-model-info.json`,
`v3-p6-cuda-preflight.json`, `v3-p5-run-lr3e-4-seed5103.json`; after measurement: one summary per ALL-INFO seed, the B0
manifest and `v3-p6-gate1.json`. No checkpoints or raw caches are committed.

## 12. What was known when this was frozen (disclosure)

- The P5 results, including the observation that ACTIVE does not beat B0 on CE and the teacher-forced confound, were known
  and motivated P6, but no P5 number selected any P6 architecture or recipe choice. The architecture was fixed from the
  plan's requirements, the V3 module inventory and the parameter budget; the layout was fixed from TRAIN-side preflight.
- The seed-5103 ACTIVE run, like every P5 run, evaluated TUNE at its scheduled updates before this freeze (update-800
  `S_run` 1.5707). Those are ACTIVE/B0 numbers; no ALL-INFO model had been trained or evaluated, and nothing about
  ALL-INFO depended on them.
- The P5.2 refined-selector re-evaluation had not yet been run and cannot influence P6.
- No ALL-INFO model had been trained at freeze time. Short CPU training on a tiny geometry (unit tests) and four TRAIN
  updates of the full model (preflight) are the only ALL-INFO training that existed.
