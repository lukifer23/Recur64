# Recur64 — Architecture (Phase 0)

Phase 0 is a systems probe. It contains no chess rules, search, replay, or
runtime. Its purpose is to prove the model-shaped graph trains on this
workstation with honest device/precision reporting.

## Selected backend

- **Framework:** Burn **0.21.0** (stable), pinned in `Cargo.toml`/`Cargo.lock`.
- **CPU baseline:** `burn-flex` (`burn::backend::Flex`) with `burn-autodiff`.
- **GPU target (pending):** `burn-cuda` (CubeCL/CUDA). Requires a CUDA 12.x
  runtime on `PATH`; none is installed yet (see `DECISIONS.md`, D3).
- **Precision:** FP32 only for now. BF16/FP16 requests fail visibly until the
  full graph is verified on the selected device.

## Probe graph

Square-token bidirectional transformer, generic over the Burn `Backend`:

```
board [B, 64, 119]
  -> Linear(119 -> d) + learned square embedding [64, d]
  -> InputBlocks
  -> for t in 1..=R:
         h = Core( RMSNorm(h + alpha * x) )
         (deep supervision: read out here)
  -> OutputBlocks
  -> heads -> (policy over legal candidates, WDL logits)
```

- Pre-RMSNorm, multi-head self-attention with a learned relative-displacement
  bias (225 buckets, gathered per head), GeLU FFN at 2x width, residual paths,
  biases on all projections.
- `alpha = sigmoid(a)`, `a` a learned scalar parameter (`alpha` init = 0.1).
- No persistent move-to-move hidden state; `h0` is rebuilt per position.
- Full backpropagation through every recurrent execution; no detach.

### Profiles (`configs/`)

| Config | width | heads | ffn | input/core/output | unique blocks | unique params |
|---|---:|---:|---:|---|---:|---:|
| micro | 192 | 6 | 384 | 0/4/0 | 4 | 1,351,840 |
| f10 | 384 | 12 | 768 | 0/8/0 | 8 | 9,805,288 |
| r10-probe | 384 | 12 | 768 | 2/4/2 | 8 | 9,805,288 |

F10 and R10 have **identical unique parameter counts** (the shared core is
counted once), which is what makes the eventual matched-parameter comparison
meaningful. Executed blocks: R=1 → 8, R=2 → 12, R=4 → 20 (final-output
inference); deep supervision → 8/14/26. Neural-compute multiplier vs F10 R=1:
1.0x / 1.5x / 2.5x.

## Policy path

- Action ID `((from*64 + to)*5 + promo)`, `promo ∈ {none,N,B,R,Q}`, space 20,480.
  Storage/index convention only — **no dense hidden→20,480 layer**.
- Base score from `source @ dest^T` (a 64×64 grid) gathered at legal candidates.
- Promotion head `[h_from, h_to, pooled] -> 128 -> 4` emits per-type deltas added
  to the base score for legal promotion candidates. Queen promotion is a distinct
  action from the non-promotion move.
- **One** masked log-softmax over legal candidates. Padding is exactly zero.
  Terminal positions (no legal candidates) bypass the softmax; a terminal-only
  batch is a visible error, never an all-masked softmax.

## Value path

Pooled WDL head `[B, d] -> [B, 3]`, ordered `[win, draw, loss]` from the
side-to-move perspective. Losses: policy CE over legal candidates + WDL CE.
Recurrent deep supervision averages the per-readout loss so increasing R does
not scale the loss.

## Recurrent sharing

- `Core` blocks are stored once and executed `R` times; parameter identity and
  gradient aggregation are proven by tests (see `STATUS.md`).
- R=1 parity: the recurrent loop at R=1 is asserted equal to an explicit
  straight-line control graph built from the same weights.

## Alternatives considered

- **tch-rs (LibTorch)** — fallback C. A CUDA LibTorch build bundles its own CUDA
  runtime, which could sidestep a toolkit install, but it adds a C++ runtime and
  is deprecated inside Burn. Not implemented.
- **Candle** — fallback D. CUDA kernels are compiled at build time and also need
  a CUDA toolkit; no obvious advantage over Burn here. Not implemented.
- **WSL2 + Burn CUDA** — fallback B. Preferred only if native Windows CUDA is
  blocked or materially inferior.

## Notable implementation finding

Burn 0.21 initializes parameters **lazily**. Cloning a module whose parameters
have not yet materialized copies the deferred initializer, so each clone
re-samples on first access — breaking value-preserving clones and therefore
resume proofs. `ProbeModel::new` calls `force_init()` to materialize every
parameter eagerly. See `DECISIONS.md` D6.
