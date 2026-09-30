# Chimera V2 - architecture contract

Architecture id `chimera_v2`. `probe_v1` and `chimera_v1` remain loadable historical
architectures; a V2 checkpoint is refused by their loaders and vice versa (architecture and head
version are recorded in checkpoint metadata and checked on every load path).

Contract versions (all part of the scientific identity):

| contract | version |
|---|---|
| head | `CHIMERA_V2_HEAD_VERSION = 1` |
| planner | `chimera-v2-planner-v1` |
| world model | `world_model_v2` |
| candidate token | `candidate_token_v1` |
| visual fusion | `visual_fusion_v1` |
| candidate facts semantics | `candidate_facts_v1` (unchanged) |

## Principle

**Thought N must have a meaningful opportunity to know or represent something thought N-1
did not.** V1 repeated the same computation on the same information. V2:

```
encode once -> propose / inspect -> obtain new exact information
            -> update bounded working memory -> inspect deeper -> update -> decide
```

## Diagram

```
                      canonical board (Observation V1)
                                  |
              +-------------------+--------------------+
              |                                        |
     symbolic square path                    optional visual path (OFF in the
     input_proj + square_emb                  first causal experiment)
              |                               64x64 top-down render -> small CNN
              |                               -> 64 square-aligned features
              +------- one-shot fuse (g_visual gate) --+
                                  |
                 BoardEncoderV2 : 2 prelude + 4 body blocks, ALL UNIQUE, ONE PASS
                                  |
                        B  [batch, 64, 512]   (immutable during thought)
                                  |
        +-------------------------+---------------------------+
        |                                                     |
  CANDIDATE TOKENS M  [batch, W, 192]                 WORLD MODEL (exact, native or WASM)
  = f( B[from], B[to], promo, move facts )            root candidate facts      (thought 1)
        |                                             successor placement       (thought >= 2)
        |                                             opponent reply set + next-
        |                                             action fact summaries     (thought >= 3)
        |                                                     |
        |            SuccessorEncoder (small)  ->  S  [batch, W, 192]
        |            ReplySetEncoder  (small)  ->  R  [batch, W, 192]
        |                                                     |
        +-------------------------+---------------------------+
                                  |
              PLANNER: K = 8 workspace tokens, width 192, ONE shared PlannerStep
                  Z_t --self-attn--> --x-attn B--> --x-attn M--> --x-attn S_t--> --x-attn R_t-->
                  proposal MLP --> GRU-like gate --> RMSNorm  ==> Z_{t+1}   (bounded, no drift)
                                  |
                  policy: per-candidate MLP( M_i, S_i, R_i, ctx(Z) )   (sparse, legal order)
                  WDL:    Linear( mean B , mean Z )                    (neutral at init)
```

The planner may READ `B` every thought. It never re-runs the board encoder.

## BoardEncoderV2

`ModelConfig` geometry: width 512, heads 8, ffn 768, `input_blocks = 2`, `core_blocks = 4`
(the historical 4 shared core blocks become 4 body blocks, each executed once),
`output_blocks = 0` (no output blocks; V2 has its own readout). Input projection, learned
square embedding and the geometry-aware relative bias are reused unchanged. `B` is computed
once per decision; a counter in the model proves it is 1 at T = 1 / 2 / 3 / 4.

## Visual fusion (`visual_fusion_v1`, optional, OFF in the first experiment)

```
X_i = RMSNorm( symbolic_embedding_i + g_visual * proj( CNN(render)_i ) )
```
computed once, before `BoardEncoderV2`. `g_visual` is an independent scalar gate. When the
pathway is off the CNN is not executed (asserted by a test).

## CandidateTokenV1

One token per legal move, in exactly `GameState::legal_actions()` order (the sparse-policy
order); padding masked. Token dimension 192:

```
M_i = RMSNorm( Linear([B[from_i], B[to_i]])            # 1024 -> 192
             + promo_embedding[promo_i]
             + FactsEncoder(candidate_facts_i) )       # 8 -> 192, NON-zero init
```
`candidate_facts_v1` fields (each 0..1): mate, check, capture, captured value / 9,
destination attacked after the move, promotion, promotion gain / 8, stalemate. Facts are part
of the token representation. There is no `gain` multiplier. An optional independently gated
direct fact logit shortcut exists for ablation, **off** by default.

## WorldModelProviderV2 (`world_model_v2`)

An exact chess world model: **consequences and legal structure, not an evaluation.** No PUCT,
no visit counts, no rollouts, no scalar score, no forced-mate bit. One source
(`recur64-coproc::world`) compiled natively and to WebAssembly, so `NativeWorldModelV2` and
`WasmWorldModelV2` are byte-identical by construction and by test.

Input: the existing fixed coprocessor input (canonical observation + canonical legal move
list). Output: a packed `u8` buffer whose length is a pure function of `(w_cap, r_cap)`:

| section | per | bytes | content |
|---|---|---|---|
| header | 1 | 8 | `n_cand: u16`, `w_cap: u8`, `r_cap: u8`, reserved |
| root facts | candidate | 8 | mate, check, capture, captured value (0..9), attacked-after, promotion, promotion gain (0..8), stalemate |
| successor | candidate | 72 | `terminal, in_check, n_replies, 0`; 64 piece codes in the child's canonical frame (0 empty, 1-6 mover's P N B R Q K, 7-12 opponent's); castling nibble, ep-file+1, halfmove clock, 0 |
| reply | candidate x reply | 20 | `valid`; the reply's 8 facts; `terminal, check, n_next`; next-player summary: legal, mates, checks, captures, promotions, max captured value, max promotion gain; 0 |

`terminal` codes: 0 ongoing, 1 checkmate, 2 stalemate, 3 insufficient material, 4 fifty-move.
A reply-set larger than `r_cap`, or more candidates than `w_cap`, is a visible error, never
silent truncation.

Documented contract limits: the reference has no game history, so **repetition is not
modelled** and the successor "observation" is the exact canonical placement, castling rights,
en-passant file and halfmove clock, not the history planes of Observation V1. Fresh-clock,
no-history fixtures (all primary V2 data) are therefore exact.

## Tool schedule (what each thought newly receives)

| thought | new information |
|---|---|
| 1 | root board `B`, candidate tokens `M`, root candidate facts |
| 2 | + one `SuccessorToken` per root candidate |
| 3 | + one `ReplySummaryToken` per root candidate (from the exact opponent reply set) |
| 4 | nothing new: one more planner integration over everything revealed |

`info_schedule`: `progressive` (above), `all_at_once` (every tool visible at every thought;
used with one planner step), `root_only` (tools never visible). All three construct the same
modules, so trainable parameter counts are identical.

SuccessorToken: per-square `Linear(13 -> 48)` on the one-hot placement + learned square
embedding, flatten (64 x 48), `Linear -> 192`, plus a `Linear(13 -> 192)` of the exact global
flags. ReplySummaryToken: each reply becomes a token by `Linear(20 -> 128)` + a candidate-conditioned
addition, then one shared masked set-attention block with a learned pooling query gives
one 192-d token per root candidate (permutation invariant; padding masked). Both are small;
neither runs the board encoder. Every tool token carries the root candidate's token (`+ M_i`)
and a tool-depth embedding, so the planner can associate `M_i`, `S_i`, `R_i`.

## Planner (`chimera-v2-planner-v1`)

`K = 8` workspace tokens, width 192, 4 heads, FFN 384. Parameter count independent of thoughts.
`Z_0 = workspace_slot_embedding + Linear(mean_squares B)`. One shared `PlannerStep`:

```
Z  = Z + thought_embedding[t]                     # t in 1..8 (T5-8 reserved for diagnostics)
n  = RMSNorm(Z)
a  = SelfAttn(n)
b  = XAttn(n + a, B);  m = XAttn(n + a, M);  s = XAttn(n + a, S_t);  r = XAttn(n + a, R_t)
proposal = FFN(RMSNorm(n + a + b + m + s + r))
u  = sigmoid( W_u [ n , proposal ] )
Z' = RMSNorm( (1 - u) * n + u * proposal )        # bounded, gated, no unbounded residual
```
Tool attentions are masked to valid candidates/replies and skipped (exactly zero) for tokens not
yet revealed. Diagnostics per thought: mean |Z|, RMS(Z), delta norm, gate mean/std, attention
mass to board / candidates / successor / reply.

## Readout

Policy: `logit_i = MLP([M_i, S_i*avail_s, R_i*avail_r, ctx_i])`, `ctx_i = XAttn(M_i, Z)`; sparse over
legal candidates, padding masked; no dense action head; no world-model "forced mate" bit.
WDL: `Linear([mean B, mean Z])` with a zero-initialised last layer (neutral at init).

## Training regime

`budget_final_v1`, `budget_training = uniform_1_4_v1`: each update / micro-batch is assigned a
final budget T in {1, 2, 3, 4} (deterministic cycle, equal frequency); the same weights serve every
budget; only the FINAL readout at the chosen T receives the target. No same-target intermediate
supervision. Process supervision (`ProcessTargetsV1`) is an extension point only.

Primary reasoning screen: policy is the training objective (value loss weight 0, chosen and
recorded before results); train only on the exact V2 mate-in-2 dataset.

## Compute accounting

Every inference budget reports board-encoder executions, planner steps, world-model states
expanded, successor and reply observations generated, tool CPU wall, GPU forward wall,
end-to-end wall, VRAM. Wall time is measured separately for cold JIT and steady state.
