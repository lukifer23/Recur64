# Recur64 HP — X15 / "Chimera" architecture

Branch: `experiment/hp-r15-h3-integration` (the deliberately experimental line).
Mainline remains the conservative conventional control line.

Status of every claim in this document is separated in
`docs/HP_X1_BUILD_RESULTS.md` as **MEASURED / INFERRED / NOT RUN**.

---

## 1. Research question

The HP evidence base removed "more of the same recurrence" as an option:

- Training produces strength (untrained → M = 0.617 [0.563, 0.671];
  M → Train1-final = 0.633 [0.595, 0.671]).
- Train1 was stopped by the D49 threefold/fifty health stop (0.625).
- 111 of its 192 final-arena games were threefold draws; in 109 of those the
  trained side was **≥ +3 material**, 103 of them **≥ +9**, median ≈ +20.
- Held-out value went *worse* than uniform late in training.
- Plain recurrence showed no value-learning benefit: at a matched LR, R1 beat
  R4 in both seeds.

So X15 asks a different question:

> Can recurrent internal reasoning become useful if the reasoning state has
> (A) an explicit persistent latent scratchpad, (B) exact deterministic
> computed chess facts, (C) an independent visual/spatial representation, and
> (D) structured supervision that gives the additional thought steps a job?

X1 builds the architecture, the interfaces, the deterministic compute, the
visual path, and the probe tooling. **X1 makes no claim that the answer is
yes.** That is the next agent's question.

## 2. Architecture

```
ObservationV1 [B,64,119]                     ComputeBankV1 [B,72,24]      VisualBoardV1 [B,3,64,64]
   │                                              │                              │
   ├─ input_proj + square_emb                     │                              └─ residual CNN
   │                                              │                                 (stem 48c, 3 stride-2
   ▼                                              │                                  stages, 3x3 residual
 S0 = input_blocks(x)      (prelude, 2 blocks)    │                                  blocks, 1x1 proj)
   │                                              │                                   -> [B,64,128]
   ├─ pooled ─► latent_init ──┐                   │
   │                          ▼                   │
   │            Z0 = latent_emb[K,128] + init     │
   │                 (position-conditioned at t=0)│
   │                                              │
   └──────────────► for t in 1..=T:               │
        S = core_blocks(inject_norm(S + x·α))     │   (shared recurrent square core)
        Z += CrossAttn_sq(Q=Z, KV=proj(S))        │   (always on)
        Z += g_compute · CrossAttn(Q=Z, KV=───────┘     (gated)
        Z += g_visual  · CrossAttn(Q=Z, KV=visual tokens)  (gated)
        Z += latent_FFN(norm(Z))
        S += g_reason  · proj_back(mean_K(Z))          (gated feedback)
        readout_t = sparse_readout(output_blocks(S), wdl_from_latent(mean_K(Z)))
   final output = readout_T
```

Text form of the contract, in order, per thought:

1. update the square state through the shared recurrent square core;
2. reasoning latents cross-attend to the square state;
3. reasoning latents cross-attend to the deterministic compute tokens;
4. reasoning latents cross-attend to the visual tokens;
5. apply the reasoning-latent FFN / normalization;
6. the square state receives gated feedback from the reasoning latents;
7. optionally produce an intermediate readout.

This order is the architecture contract
(`REASONING_CONTRACT_VERSION = "chimera-thought-loop-v1"`); changing it changes
the scientific identity.

## 3. Module interfaces

| Interface | Where | Contract |
|---|---|---|
| `ChimeraModel::forward_thoughts(input, cands, t)` | `recur64-model/src/chimera.rs` | `t ∈ 1..=8`; parameter count is independent of `t`; returns one readout per reading thought plus one `ThoughtMetrics` per executed thought |
| `ChimeraInput { board, compute, visual }` | same | `board` `[b,64,119]`; `compute` `[b,72,24]` floats in 0..255 (or `None`); `visual` `[b,3,S,S]` in 0..1 (or `None`) |
| `ComputeProvider` (`None` / `NativeV1` / `WasmV1`) | `recur64-compute` | `compute_batch(&[Vec<u8>], &mut [Vec<u8>])`; never silently falls back to zeros |
| `coproc::compute::compute_bank(in, out)` | `recur64-coproc` | the one algorithm; compiled natively **and** to `wasm32-unknown-unknown` |
| `coproc::visual::render_board(in, out, size)` | `recur64-coproc` | deterministic RGB, canonical orientation |
| `build_x15_batch(states, exp, provider, device)` | `recur64-runtime/src/x15_inputs.rs` | the single wiring point from `(ObservationV1, legal)` to model tensors, with per-stage phase timings |
| `ExperimentalConfig` | `recur64-model/src/experimental.rs` | the `[experimental]` run-config block; validates and refuses unknown keys |
| `Architecture` | same | `probe_v1` \| `chimera_v1`, each with its own readout-head version |

## 4. Parameter breakdown

`recur64 x15 info --config configs/x15.toml` prints the live numbers, split by
subsystem, and asserts the parts sum to `Module::num_params`. The trunk
geometry is R15's (width 512, 8 heads, ffn 768, 2/4/2), so it is directly
comparable with the historical 15,154,632-parameter F15/R15 family; the new
subsystems are additive and are reported separately rather than folded in.

| Subsystem | Contents |
|---|---|
| symbolic | `input_proj`, `square_emb`, 2 prelude blocks, `inject_norm`, `alpha` |
| recurrent square core | 4 shared blocks (counted once) |
| output blocks | 2 blocks |
| reasoning latents + bus | latent embeddings, position-conditioning projection, latent norm + FFN, feedback projection, 3 gate scalars |
| compute embedding + xattn | bank projection, token-type embedding, one cross-attention |
| visual CNN + xattn | residual CNN + one cross-attention |
| square xattn | one cross-attention (always on) |
| heads | final norm, source/dest, promotion MLP, WDL head, latent WDL term |

The exact totals are in `docs/HP_X1_BUILD_RESULTS.md` (MEASURED).

## 5. Scientific identity fields

The X15 scientific identity is the probe identity plus, for a non-default
`[experimental]` block:

- `identity_version: 5`;
- `architecture`, `architecture_head_version`, `reasoning_contract_version`;
- `thought_steps`, `reasoning_tokens`, `deep_supervision`;
- the full `reasoning`, `compute`, `visual` and `retrieval` sub-configs;
- `compute_bank_version` (`compute_bank_v1`) and `visual_render_version`
  (`visual_board_v1`).

An absent `[experimental]` block is byte-identical to the historical identity:
the block is skipped when serializing, so **every existing resolved and
scientific hash is unchanged**.

Provider *selection* (`native_v1` vs `wasm_v1`) is recorded in the identity for
now. Once `x15 parity` proves byte equality, a later agent may demote the
execution backend to execution-only, exactly as D55 did for
fusion/autotune/candidate buckets; that demotion must be a documented ADR with
the parity evidence attached.

## 6. Compute-bank schema — `ComputeBankV1`

Input buffer, fixed `INPUT_LEN = 31,496` bytes:

```
u8   version = 1
u8   mate_search_depth (0/1/2)
u16  reserved
u32  n_legal
256 x (from u8, to u8, promo u8, 0)      canonical squares, promo 0..4
[64,119] ObservationV1 as little-endian f32
```

Output buffer, fixed `OUTPUT_LEN = 1,664` bytes = `64 × 24 + 8 × 16`.

Per-square (24 bytes): piece identity; attacked-by-own/opponent; attacker
counts by piece type for both sides; own/opponent defender counts; the
candidate list's legal-from, legal-to, legal-capture-to and
legal-promotion-to counts; attacker totals; and a ray/flag byte (own and
opponent diagonal/orthogonal slider relations, en-passant target, own king in
check, opponent king attacked, occupied).

Global (8 tokens × 16 bytes): piece counts by type and side with king
presence; castling rights, en-passant availability and square, canonical side
to move, insufficient material, halfmove clock, repetition count, in-check
state, both king squares, checker count, attackers on the own king; legal,
capture, promotion, en-passant, castling and quiet move totals; tactical state
(mate-in-1 available, stalemate moves, fifty-move proximity, hanging-piece
*counts*); bounded exact tactics (mate-in-1 count, mate-in-2 availability and
count, stalemate), mobility summary; move classes (capture/quiet/promotion
checks, capture promotions, en-passant captures, double-check moves); and a
version stamp.

Deliberately **absent**: any engine evaluation, any tablebase value, any
opening-book value, any human strategy label and any hand-tuned scalar
"material evaluation". The bank gives piece counts and relations; the network
learns their strategic value.

Known exactness limit: the bank is derived from `ObservationV1`, so the
halfmove clock and repetition count are recovered from their normalized
features (lossy beyond the observation's own clamp at 150 and 5). That is
documented rather than hidden.

## 7. Visual-render schema — `VisualBoardV1`

- top-down, canonical current-player orientation (the side to move is always at
  the bottom of the image);
- deterministic integer pixel arithmetic: no perspective, no lighting, no
  shadows, no random rotation, no texture augmentation;
- **no system fonts, no bundled artwork, no network assets** — every piece is a
  procedural shape;
- piece *type* changes the silhouette (pawn disc, knight slant, bishop diamond,
  rook box, queen disc + crown, king disc + cross) and piece *ownership*
  changes the fill luminance (own light with a dark outline, opponent dark with
  a light outline), so the two are separable in grayscale;
- supported sizes 64 and 96 (`render_board`); the visual *encoder* requires the
  token grid to be a power of two, i.e. 64 or 128, so X1 encoders use 64.

## 8. Reasoning loop and gates

Gates: `use_compute`, `use_visual`, `use_reasoning_latents` (`reasoning.enabled`)
and `thought_steps`. Learned residual scales `g_compute`, `g_visual`,
`g_reason` are all `sigmoid(logit)` initialized at `logit = 0`, i.e. **0.5** —
fully open, and therefore no subsystem is switched off at initialization. The
module-gradient probe (`recur64 x15 grads`) asserts a non-zero, finite gradient
for every gated subsystem on the first step.

All-off control: `thought_steps = 1` with `reasoning.enabled = false` reduces
the graph to the symbolic square-token transformer (prelude + shared core +
output blocks + the sparse heads). The trunk geometry is the new X15 config, so
this is *not* claimed to be bit-identical to R15; the difference is the
geometry, not the function.

## 9. Training and readout modes

| Mode | Behaviour |
|---|---|
| `final_only_v1` | only the final thought contributes loss (default) |
| `same_target_v1` | every thought uses the same target; intermediate readouts carry weight 0.25 |
| `progressive_search_v1` | thought `t` supervises against the teacher target searched to `ladder[t]` simulations |

`reasoning_tokens` and `thought_steps` change compute but **not** the trainable
parameter count; that is asserted at every `T`.

`ReasoningTargetsV1` (design): a fixed probe dataset built from Recur64's *own*
search — a configurable simulation ladder (e.g. 16/32/64/128), noise-free — so
a later agent can ask whether thought 4 moves closer to a deeper-search target
than thought 1. Positions are stored as `(start_fen, prefix moves)` so the
exact history (and therefore the repetition feature) survives. No external
engine and no self-play are involved. **X1 ships the schema and the intent; the
full generator/trainer is the first task of the experiment-running agent.**

## 10. Known risks

- **The architecture may simply not work.** Nothing here shows that latent
  reasoning helps; that is the open question.
- **Compute-bank cost.** Depth-2 mate search is exhaustive and is a
  diagnostic-only path; the training path runs depth 1.
- **Visual encoder bottleneck.** The CNN is the most expensive per-position
  addition on a 4 GB card; phase timings in `x15 sanity` exist precisely to
  catch that before a long run.
- **Repetition pathology persists.** X15 does not by itself address the
  conversion failure that motivated it; a later agent must still measure
  conversion.
- **Self-play integration is absent by design.** X15 is wired into batched
  inference and the probe harness, not the pilot; a pilot run needs a separate
  integration decision.
- **`wasm` embedding** means the artifact's digest is part of the repo; a
  toolchain upgrade can change it (a test catches the mismatch).

## 11. Future `RetrievalProvider` interface (X2, inactive)

```rust
pub enum RetrievalProviderKind { None }          // the only X1 value
pub struct RetrievalConfig { provider: RetrievalProviderKind, memory_tokens: usize }
```

The reasoning bus has a designed `memory_tokens` input alongside
`compute_tokens` and `visual_tokens`: a future provider would return
`[b, m, Daux]` state to cross-attend into, exactly like the compute path, with
its own gate. The intended X2 direction is **self-generated solved-position
retrieval** (positions Recur64 itself has searched to a proven result), not an
external chess corpus. No vector store, no HNSW, no opening database, and no
external corpus exists in X1; `ExperimentalConfig::validate` refuses any
setting other than `none` with `memory_tokens = 0`.
