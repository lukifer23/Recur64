# Recur64 — a Rust-first, from-scratch chess learning laboratory

**Research and implementation specification · 23 September 2026**  
**Status:** proposed design, not an implemented engine or a measured training result.  
**Working name:** Recur64. This is a project label, not a claim of architectural priority.

## 1. The experiment

Build a small chess model that learns from its own games and search, rather than imported chess-engine labels or pretrained weights. Use a square-token transformer, then investigate whether repeatedly applying a shared transformer core produces useful additional computation for chess. Keep a conventional feed-forward transformer as the control. Use Rust for the model, training loop, search, self-play, storage, evaluation, and command-line application.

The central question is:

> At a fixed end-to-end compute budget, does a roughly 10M-parameter recurrent transformer become a better chess player than a similarly sized feed-forward transformer, and when should computation go into internal refinement rather than external tree search?

A second, later question is whether discrete denoising of short, valid future state-and-action trajectories adds anything beyond ordinary auxiliary training or simply spending the same time searching more nodes.

This is not a proposal to reproduce every component of HRM, TRM, Chessformer, DiffuSearch, and AlphaZero simultaneously. Their useful ideas must enter as separately measurable changes. A negative result that leaves a reliable, useful transformer chess engine is a successful research outcome.

### 1.1 Non-negotiable scope

- Random initialization for the main model lineage. No pretrained LLM, LC0 network, Stockfish training labels, or human-game corpus in the main training run.
- No convolutional/ResNet trunk. Ordinary residual connections inside transformer blocks are permitted; they are not a ResNet chess architecture.
- Rust is the primary implementation language, including actual differentiation and optimizer updates through a Rust-accessible training stack. Do not quietly replace training with Python.
- No Docker. Do not rent cloud GPUs, alter drivers, enable Windows features, upload private files, or start expensive unattended runs without owner approval.
- Use established tensor and legal-move libraries. “From scratch” here means our own randomly initialized model, search/learning integration, and experimental implementation—not writing a GPU compiler and move generator before testing a learning hypothesis.
- External engines and curated positions may be used in explicitly isolated evaluation. They must never be silently ingested into zero-mode replay.
- Preserve working baselines, checkpoints, configuration contracts, and reproducible experiments. No fake benchmarks, silent fallbacks, placeholder implementations, or passing tests achieved by weakening assertions.

### 1.2 Deliverables

A buildable workspace; tested rules and representations; working GPU training; PUCT and Gumbel search; batched self-play; compact validated replay; exact-enough resumable training; UCI play; reproducible arenas; a feed-forward baseline; a controlled recurrent-model experiment; and a report with raw measurements, uncertainty, and failure cases. Diffusion, geometric attention bias, auxiliary objectives, and 30M scaling are gated research extensions, not requirements for the initial vertical slice.

## 2. Hardware assumptions and the first decision gate

The supplied screenshot establishes a Dell Pro Max Slim FCS1250, Intel Core Ultra 9 285K, 64 GB installed RAM at 5600 MT/s, and 1.86 TB storage with 354 GB used. Its graphics summary says “16 GB” and “Multiple GPUs installed.” It does **not** identify the GPU, establish NVIDIA CUDA availability, or prove that one adapter has 16 GB of dedicated VRAM. Do not pool integrated/shared memory with discrete VRAM.

Start with read-only discovery in Windows PowerShell:

```powershell
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion
wsl --status
wsl --list --verbose
```

A missing command is information, not permission to install a driver or change the OS. Do not rely on WMI `AdapterRAM` alone to establish modern GPU memory capacity. Record adapter identity, dedicated memory, driver, available memory under normal desktop use, OS, CPU topology, Rust toolchain, and existing WSL/toolkit status. Avoid copying Windows product/device IDs into reports.

If the actual accelerator is NVIDIA and WSL2 is available/approved, prefer a WSL2 Linux development environment with CUDA, project files, and active replay in its Linux filesystem. CUDA on WSL uses the Windows host NVIDIA driver; do not install a Linux display driver inside WSL. Consult the current NVIDIA instructions before changing toolkit packages [S15]. Native Windows remains an option if its selected backend passes the same tests. Non-NVIDIA hardware changes the backend selection, not the chess research question.

### 2.1 Initial resource envelopes — proposals, not measured requirements

For a **confirmed 16 GB dedicated accelerator**, begin with an approximately 12 GB project GPU-memory ceiling, reducing it if desktop use requires more. Begin with a 24 GB host-memory envelope, then adjust from observed available RAM; 64 GB is total system RAM, not all available to the learner. Cap initial project data/checkpoints at 50 GB. Start a smaller interactive profile when the workstation is in use.

A 10M-parameter model stores about 40 MB of FP32 weights; a 30M model about 120 MB. A rough 16–20 bytes/parameter training-state allowance gives 160–200 MB and 480–600 MB respectively, **before activations, allocator reservations, recurrent unrolling, workspaces, copies, and runtime overhead**. These are arithmetic estimates, not GPU measurements. Recurrent backward passes and useful examples per second matter much more than weight-file size alone.

Do not promise games/day, rating, or days to a strong player before a real training-and-self-play benchmark. Use measured new training positions per second, the fraction of time allocated to self-play, and the actual learner speed to set run budgets.

## 3. Evidence that informs the design

The following are concise findings, not claims that the proposed combination has already been validated.

### 3.1 Chessformer: borrow the board-specific representation

Chessformer uses square tokens, geometric attention biases, and an attention-based source/destination policy. Its small-model ablations are directly relevant. However, its controlled architecture comparisons largely use a fixed LC0 self-play corpus as supervised distillation, not fresh zero-learning runs for each architecture [S01]. We borrow structural ideas, not published ratings or training-efficiency claims.

### 3.2 HRM and TRM: borrow recurrence, not an assumed reasoning guarantee

HRM describes a 27M-parameter two-timescale recurrent model evaluated on reasoning puzzles; TRM simplifies the recursive approach using a small shared network and repeated supervision [S02, S03]. Neither original result establishes strong full-game chess learning from self-play. A subsequent mechanistic study reports nonmonotonic refinement and incorrect fixed points [S04]. Consequently, extra loops must earn their cost in our arena, including positions where more loops make a move worse.

### 3.3 DiffuSearch: a credible later branch, with a different data regime

DiffuSearch studies discrete denoising for chess with a default roughly 7M-parameter model, Stockfish-labelled examples, and 20 denoising steps. Its reported action-plus-state formulation outperformed action-only future prediction in the relevant ablation [S05]. Our self-play-only, short-horizon extension is a new hypothesis, not a replication or an established route to the paper’s performance.

### 3.4 Gumbel AlphaZero: prioritize low-budget search quality

Gumbel policy improvement uses sampling without replacement and sequential halving to make better use of limited search. The algorithm includes its own action selection and improved-policy target construction; a root-noise tweak on ordinary PUCT is not a faithful implementation [S06, S07]. We will build PUCT first, then verify Gumbel against the paper and official reference.

### 3.5 SSRL: distinguish a named algorithm from general self-supervision

The published SSRL method selects high-return trajectories and imitates them using supervised losses [S08]. That is distinct from rules-derived auxiliary labels or ordinary search-target learning. For adversarial chess, my concern is that winning trajectories can contain poor moves rescued by an opponent’s mistakes. Keep all game outcomes in main replay; treat selective self-imitation as an optional isolated experiment.

### 3.6 Rust: feasible, but backend correctness precedes architectural work

Burn supports Rust deep learning with autodiff and accelerator backends. Its repository marks the LibTorch backend deprecated as of 0.22.0, so do not base the project on outdated Burn-plus-LibTorch examples [S09, S10]. Candle supports training as well as inference [S11]. Direct `tch-rs` is a separate Rust binding to LibTorch and remains a possible fallback [S12]. A documented feature is not proof that our exact backward graph is supported or fast.

## 4. Stack and implementation boundaries

### 4.1 Primary stack

Use a pinned, tested Burn release and the appropriate accelerator backend, preferably its CUDA path for a confirmed NVIDIA device. Commit `Cargo.lock` and a matching Rust toolchain. Read documentation for that exact release. Do not mix release dependencies with examples copied from repository main.

Use `cozy-chess` for legal move generation and board operations [S13]. Maintain full game history, repetition accounting, game termination, policy representations, and self-play conventions in our own core layer. Check dependency licenses and preserve notices. An independent oracle such as `shakmaty` can aid differential testing, subject to a deliberate license/dependency decision [S14]. Do not silently embed a separately licensed engine’s source into the project.

Use normal Rust crates for CLI/configuration, serialization, compression, hashing, bounded queues, and tracing. Select exact versions during implementation. Avoid unnecessary service frameworks, distributed databases, agent frameworks, or Kubernetes-style deployment machinery.

### 4.2 Backend stop/go test

Before the real engine, implement an actual small board-transformer forward/backward/update/checkpoint workload. It must exercise attention, normalization, gather, masking, softmax/log-softmax, policy and WDL losses, AdamW, recurrence, and record restore—not just a matrix multiplication benchmark.

Verify FP32 correctness first. Test a complete BF16 mixed-precision training path only if both device and selected backend support the required operations. Keep optimizer/master state and numerically sensitive reductions in FP32 as required by the implemented precision scheme. Do not blindly cast the entire model and optimizer to half precision. FP16 requires a deliberately tested scaling/overflow strategy. Unsupported BF16 should produce a visible decision to use FP32/smaller batches, not silent CPU execution.

Benchmark warm and cold behavior separately; accelerator work must be synchronized for timing. Measure batch 1, 16, 64, and 128 inference, plus 32/64/128 physical training batches where feasible. Test recurrence 1, 2, and 4. Report unsupported or out-of-memory combinations honestly.

If Burn cannot pass correctness or reasonable throughput on this graph after a time-boxed investigation, write an architecture decision with logs. The first fallback to investigate is direct `tch-rs`, which preserves a Rust application but uses a C++ ML runtime. Candle is another candidate, not a third framework to implement in parallel. Switching the trainer to Python or custom-building autodiff requires a separate explicit decision.

## 5. Model specification

### 5.1 Model family

| Profile | Width | Attention heads | FFN width | Unique transformer blocks | Approximate role |
|---|---:|---:|---:|---:|---|
| Micro | 192 | 6 | 384 | 4 | Approximately 1–2M; correctness and end-to-end smoke runs |
| F10 | 384 | 12 | 768 | 8 | Approximately 10M feed-forward control |
| R10 | 384 | 12 | 768 | 8 | Approximately 10M; 2 input + 4 shared-core + 2 output blocks |
| R30 candidate | 640 | 20 | 1280 | 9 | Approximately 30M; 2 input + 5 shared-core + 2 output blocks |

These totals include only approximate allowances for embeddings and heads. The implementation must print an exact count by submodule, count shared parameters once, and distinguish unique parameters from executed blocks. Geometric attention bias or auxiliary modules change the totals and experiment IDs.

Use bidirectional attention over 64 squares, pre-RMSNorm (initial epsilon 1e-5), GeLU FFNs at 2× width, standard transformer residual paths, and no dropout initially. Specify projection biases and the GeLU implementation in the model contract; do not change them silently between runs. Start with learned square embeddings plus a small learned two-dimensional relative-displacement bias. Do not import causal FEN-text modelling conventions or long-context optimizations simply because they are fashionable.

### 5.2 Observation version 1

The board is represented as `[batch, 64, 119]` before a learned input projection. For each square:

- Eight current/recent position frames, each with 13 one-hot piece categories, including empty, plus a validity bit: `8 × 14 = 112` features.
- Four current castling-right indicators, broadcast to all squares: own kingside/queenside and opponent kingside/queenside.
- One current en-passant-target indicator at the relevant square; all zeros when none.
- One halfmove-clock feature, `min(clock,150)/150`, broadcast.
- One current repetition-count feature, `min(count,5)/5`, broadcast.

This gives 119 features. Unavailable history frames use all-zero piece features and validity zero; they are not invented empty boards. Frame 0 is the current position, frame 1 the preceding ply, and so on. Validate the formula in a schema test.

Canonicalize from the **current** side-to-move’s perspective. With a1=0 and rank-major indexing, Black-to-move uses rank reflection (`square XOR 56`) and color swapping; White-to-move is identity. Apply the same current-player transform to every history frame. Do not independently rotate each historical frame into that frame’s moving player’s perspective. Files stay fixed, preserving kingside/queenside meaning. Move mapping must be exactly reversible.

The environment retains complete relevant history even though the network sees a bounded history window. This observation can alias positions with different distant repetition histories; do not falsely call it a fully Markov representation of every chess draw rule. Environment adjudication, replay provenance, and tree-path history remain authoritative.

### 5.3 Policy and value

Project final square embeddings into source and destination vectors of width 128. Their dot products produce a 64×64 matrix of base move scores. Add learned promotion-specific scores for actual legal promotion candidates. Gather scores for the position’s legal moves, then apply **one joint softmax across those legal moves**.

An action ID is `((from * 64 + to) * 5 + promotion_code)`, with promotion codes `{none,N,B,R,Q}` and total ID space 20,480. This is a storage/indexing convention, **not** permission to add a 20,480-by-hidden dense output layer. Store legal action IDs sparsely; pad/bucket candidate lists only inside batches, with an explicit mask. Never truncate the legal list to an arbitrary maximum.

The promotion head can use `[h_from, h_to, pooled_board] -> 128 -> 4`; merge its selected promotion delta into the corresponding base score. Queen promotions must not be merged with non-promotion moves. Test collisions and underpromotions exhaustively over legal fixtures.

Use a pooled WDL head with three logits ordered `[win,draw,loss]` from the side-to-move’s perspective. Scalar value is `P(win)-P(loss)`. Backups change sign every ply. Terminal nodes bypass neural evaluation, and terminal positions with no legal actions must never be sent through an all-masked softmax.

Cozy-chess internally represents castling as king-to-rook moves; use its conversion utilities or an equivalently tested adapter to keep standard UCI and our action encoding consistent [S13]. A move such as ordinary white kingside castling must not silently change policy IDs between the engine, replay, and UCI layers.

### 5.4 Recurrent core — our proposed design, not a faithful HRM reproduction

Let `x` be the embedded board, `h0 = InputBlocks(x)`, and define a shared refinement core:

```text
h_t = Core(Norm(h_(t-1) + alpha * x))
y_t = OutputBlocks(h_t)
(policy_t, wdl_t) = Heads(y_t)
```

Use a bounded, learned input-injection scale, initially `alpha = sigmoid(a)` with `a = log(0.1/0.9)`, giving alpha 0.1. Use RMSNorm for the displayed Norm. Record this parameterization in the architecture decision and compare alternatives only as explicit changes. The input features are immutable throughout refinement. Reset `h0` for each independently evaluated position; there is no persistent hidden state carried from one played move to the next in version 1.

For R10, final-output inference executes `2 + 4R + 2` blocks: 8, 12, or 20 for R=1,2,4. During deep supervision, evaluating the output blocks at each loop executes `2 + 6R` blocks (8, 14, or 26) before backward; include this cost in measured training compute. Output-block weights are shared across readouts and do not feed back as the next recurrent state.

Implement a feed-forward control whose R=1 graph can be made identical to the recurrent model’s R=1 graph at initialization. This lets us test weight-copy and implementation equivalence before comparing different training regimes. Label any additional architecture changes separately.

Start recurrent training with full backpropagation through the executed loops and a normalized mean of policy/WDL losses at the supervised readouts. Sample a single R for each physical batch from `{1,2,4}`; begin with probabilities `{0.25,0.25,0.50}` as a pilot setting, not an established optimum. Preserve an R=1-only training control. Do not silently detach the shared latent or claim that a last-step gradient approximation is identical to full backpropagation.

Later memory-saving gradient truncation, checkpointing, multiple latent timescales, and learned halting are independent experiments. Extra inference loops outside the trained range are exploratory and may reduce strength. A loop is not a chess ply or a proven reasoning step.

### 5.5 Geometric attention bias extension

After the basic transformer and recurrence comparisons work, implement the Chessformer paper’s board-conditioned geometric bias as a documented option [S01]. Reproduce its actual tensor construction in a separate specification before coding. Compare it with the simple relative-bias baseline, count its parameters and memory, and benchmark backward/inference with it enabled. Do not replace it with a vaguely named “geometry module” and call that a replication.

Do not hard-mask attention to current legal moves or attack rays; remote squares can matter. Legal move masking belongs at action selection, not as a universal restriction on internal representation.

## 6. Rules, state, and search contracts

### 6.1 Rules profile

Standard chess only initially, not Chess960. The move library supplies legal moves; our game wrapper supplies history-sensitive outcomes and a declared draw convention. FIDE distinguishes claimable threefold/50-move draws from automatic fivefold/75-move draws [S16]. Version 1 may use an explicit **auto-claim-on-current-position** convention for threefold and 50-move conditions, used consistently in self-play and arenas. Document that this is an engine-training convention, not a full model of optional human claim strategy.

Checkmate and stalemate must be tested before applying competing post-move termination conditions. Retain the full information needed to count repetitions, including side to move, castling rights, and legally relevant en-passant status. A board-placement-only hash is insufficient.

Implement and test sound known dead-position/insufficient-material cases. Do not describe a conservative material test as a complete solver for every possible dead position, including unusual blocked positions. State exactly what is recognized; do not add heuristic draws that can mislabel a winning game. This limitation is separate from legal move generation and should be visible in the rules-profile documentation.

A configurable 512-ply safety cap is an administrative truncation, **not a draw result**. Mark unfinished games as truncated and omit their terminal-value supervision, or persist and continue them. Do not quietly convert hangs, interruptions, or caps to draw targets. Disable resignation and heuristic material adjudication in the initial zero runs.

### 6.2 PUCT control

Build a transparent, correct PUCT tree first. Keep per-edge visit count, prior, accumulated value, and explicit perspective. Test it on small exact synthetic game trees and rule-generated mate/draw positions. Deterministic mathematical fixtures are test infrastructure, not mocked production inference.

Start with independent trees and no transposition graph merging. Permit safe root reuse only when full state/history and model identity agree. Terminal results use exact rules; neural values never override checkmate, stalemate, or declared draw outcomes.

Use a fixed small simulation budget for smoke tests and 64 simulations/move as a first pilot. Tune PUCT exploration on a development set; no single published constant should be assumed optimal for this representation and value scale.

### 6.3 Gumbel implementation

Implement root Gumbel sampling without replacement, sequential halving, the paper’s non-root selection, Q completion/normalization, and the corresponding improved policy targets. Use [S06] and official `mctx` [S07] as references. Preserve attribution if reusing code or derivations. Export small reference cases or construct independently verifiable mathematical golden fixtures; the production runtime does not require JAX.

Start with 8 sampled root candidates and a total 64-simulation budget, always bounded by the legal-move count. Validate that the halving schedule fits the budget and spends the declared number of simulations. Test ties, one legal move, low-budget exhaustion, terminal children, unvisited candidates, zero/very small priors, and NaN rejection.

Do not substitute ordinary visit counts for Gumbel’s improved-policy target. Do not bolt on Dirichlet noise by default; it is a separate exploration choice. Evaluation disables training exploration; any stochastic evaluation mode needs a recorded seed and symmetric protocol. A root-only Gumbel/PUCT hybrid may be useful, but it must have a distinct name and cannot be labelled a faithful reproduction.

### 6.4 Batching and GPU ownership

Initially run many independent games with at most one outstanding leaf evaluation per game. This fills GPU batches without the complexity of parallel search races inside a tree. Begin with 8 CPU workers and 128 active games, then sweep worker count, active games, maximum batch 32/64/128, and batching timeout. These are starting hypotheses, not universal performance settings.

Use one GPU inference owner, bounded requests, reusable buffers, explicit cancellation, and no unbounded task creation. Record CPU selection time, request queue delay, actual GPU batch size, inference time, and whole-move latency. A faster matmul that makes the queue slower is not a net win.

On one accelerator, initially alternate frozen-snapshot self-play phases with learner phases. Do not run two competing full-occupancy GPU processes by default. Publish new weights atomically at a phase boundary. Every replay example records the model identity that generated its search target.

Key neural-output caches by encoded observation, model hash, recurrence count, and precision/configuration. History-sensitive search values require stronger path handling; do not reuse them under a bare Zobrist key. Cache behavior must not change game outcomes through history aliasing.

## 7. Learning and replay

### 7.1 Main targets and loss

For a completed game, store the search policy target and terminal WDL outcome for each sampled state. Use:

```text
L = CE(search_policy_target, legal_policy_prediction)
  + CE(terminal_WDL_target, WDL_prediction)
```

With recurrent deep supervision, average this objective over the supervised readouts so merely increasing R does not multiply the loss scale. AdamW weight decay is an optimizer setting, not an undocumented extra label source. For Gumbel runs, `search_policy_target` means the algorithm’s improved target; for PUCT controls it is the documented visit-based target.

Begin with AdamW, learning rate 3e-4, 1,000-update warmup, weight decay 1e-4, global gradient norm clipping at 1.0, and a planned cosine decay over the run’s update budget. These are initial settings to test, not claims of optimality. Log the actual schedule and never restart warmup accidentally on resume.

Try physical batch 64 first and accumulate to effective batch 256. Increase or reduce based on verified memory and throughput, not parameter count alone. Mixed precision is enabled only after the Phase 0 checks. Log finite loss/gradient checks, gradient norm, update norm, policy entropy, WDL calibration, and parameter movement.

### 7.2 Replay representation

Use compact versioned binary shards with checksums, compression, an atomic manifest, and bounded in-memory indices. Avoid a giant JSON array of dense observations and 20,480-element policy vectors. Store compact boards/history or reconstructible game records, sparse legal action IDs and target probabilities, game outcome, termination reason, selected action, search budget, recurrence mode, model IDs, run ID, and schema version.

Write only complete validated records. On interruption, leave temporary files clearly separate from committed shards; a corrupted shard must cause a visible quarantine/error, not silent sample skipping that changes distributions. Validate target probability sums, legal IDs, finite values, history continuity, and result perspective at ingestion and in replay-audit tooling.

Begin with capacity 250,000 positions, growing toward 1–2M only when generation rate and sample age justify it. Capacity is not the minimum warm-start requirement: a pilot can start learning after 4,096 valid positions from completed games, revising this threshold from observed diversity. Never train a terminal-value target from an unfinished game merely to reach the threshold. Track average age in model versions and time. Keep all outcome types. Partition development/evaluation by entire games and controlled position families, not random neighbouring plies. Check duplicates and repeated opening prefixes when measuring validation quality.

Define replay reuse as **training examples consumed divided by new positions inserted**, counting gradient accumulation correctly. Start around 2, with pilot sweeps such as 0.5, 1, 2, and 4. Do not confuse thousands of optimizer updates on stale data with improved self-play learning. If generation stalls, the controller should not train forever on one old shard.

### 7.3 Early learning and collapse diagnosis

Sparse terminal feedback is deliberately part of the zero-learning experiment. Early random self-play may produce weak or draw-heavy data. Before changing rewards, examine search correctness, truncation rate, legal-action entropy, repetition loops, terminal-value signs, and opponent/model versioning.

Primary milestones are: valid complete games, learning the small fixture set, changes in policy/value on held-out own games, improvement over random play and earlier snapshots, and sustained improvement against a frozen evaluation suite. Low loss alone is not strength.

Do not add material rewards, winning-game-only filters, engine-generated openings, or imported tablebase labels to rescue the main run without declaring a changed experiment. Own-replay restart curricula, mixed historical opponents, and reanalysis with the model’s own deeper search can be investigated later, with provenance and their full compute cost counted. Standard-start and curriculum-assisted results must be labelled separately.

Use a current frozen self-play snapshot and a protected champion checkpoint. Periodic candidate arenas determine whether to replace the deployed champion; poor candidates must not destroy the previous checkpoint. A mixed-opponent population is deferred until same-snapshot training works. Log both players’ identities when introduced.

### 7.4 Auxiliary training and SSRL branch

The first optional auxiliary objective is a lightweight prediction of the legal-destination map from board embeddings, supervised by our rules engine, with a pilot coefficient 0.05. This label must not be fed back as an input shortcut to the auxiliary head. Compare against the same model without it. It is rules-derived self-supervision, not the published SSRL method.

Do not add a half-dozen auxiliary heads simultaneously. Selective high-return self-imitation, if tested, gets its own replay mixture and must be evaluated on defence, draws, and opponent-generalization, not just win frequency against the current weak self-play opponent.

## 8. Checkpoints, failure recovery, and reliability

A training checkpoint must contain model parameters, optimizer moments/master state, learning-rate schedule position, gradient-scaler state where relevant, random-generator states, sampler/controller counters, model configuration, observation/action/rules/replay schema versions, git revision, dependency lock digest, and backend/device metadata. A weights-only file is an inference export, not a resumable learner checkpoint.

Test interrupted-versus-uninterrupted continuation on CPU where deterministic equality is achievable. On GPU, specify numerical tolerances and non-determinism rather than asserting bitwise equality without evidence. Check that the same data order and learning-rate state resume and that optimizer moments were not reset.

Checkpoints and replay manifests use temp-write, flush as appropriate, atomic publication, and checksums. Keep at least last-good plus a separate champion. A retention policy must never delete the only recoverable checkpoint. Refuse incompatible schemas with an explanation; perform migrations in explicit tools, not guessed on load.

Ctrl+C should cancel safely, finish or persist in-flight game state as appropriate, publish valid completed shards, and preserve recoverable training state. A GPU error, NaN, inference timeout, disk-full condition, bad shard, illegal action, or nonfinite target must be reported clearly. Never turn an exception into a uniform policy or zero value and continue as though the run were valid.

## 9. Evaluation that can actually answer the question

### 9.1 Three comparisons, not one

1. **Matched-data architecture check:** train candidate models on the same fixed corpus of our own self-play to inspect loss, calibration, and move quality without changing data generation at the same time.
2. **Matched end-to-end training budget:** compare complete online learning runs, including self-play, learner time, refinement, auxiliary work, and reanalysis. This is the practical single-workstation result.
3. **Matched move-time inference:** compare R=1/2/4 and search budgets at equal actual time per move. Equal node counts alone favour slower, more expensive models unfairly.

Always report raw-policy play as well as searched play. This separates what the network has learned from what search is repairing. For root-heavy recurrence, measure `root R=4, leaves R=1` against a uniform R policy; do not assume the hybrid wins.

### 9.2 Opponents and test sets

Use random legal play, frozen earlier checkpoints, a fixed reference engine configuration, and a held-out tactical/endgame suite. External engine evaluations are quarantined from training. Record exact engine build, options, time/node budgets, threads, hash settings, and hardware contention. Engine skill-level options are not a conversion to FIDE or online-platform Elo.

For head-to-head matches use paired colours on the same opening positions and a stable rule/time-control profile. Development openings used for tuning are separate from the final test suite. On a shared GPU, serialize moves or otherwise impose symmetric resource sharing; one model cannot receive an uncontested device while the other waits behind training.

Use approximately 200 games only for screening. Confirm finalists with a larger predeclared sample such as 1,000–2,000 games, three training seeds when budget permits, and confidence intervals respecting opening-pair correlation. Report W/D/L, draw rate, raw score, sample size, and uncertainty. An internal rating scale is not an absolute playing-strength claim. Do not test hundreds of variants and report only the luckiest one.

### 9.3 Metrics

Record actual parameter count; peak GPU and host memory; warm/cold inference; batch-size distribution; recurrent executed blocks; self-play positions and completed games per second; training examples and updates per second; examples-consumed/new-position ratio; checkpoint overhead; game-length distribution; real draw versus truncation reasons; policy entropy; WDL calibration; tactical correctness; move changes across R; arena score/interval; and total end-to-end accelerator time.

Keep per-position “more thought helped/hurt” examples. A legal-move rate of 100% is primarily a rules-mask correctness requirement, not evidence that the network has learned chess. Attention maps or latent traces are not faithful explanations by default.

### 9.4 Minimal ablation ladder

| ID | Experiment | What it isolates |
|---|---|---|
| A | F10 + PUCT | Working transformer/search control |
| B | F10 + faithful Gumbel | Search/target algorithm |
| C | R10 + Gumbel, R=1-only versus mixed R | Shared refinement and its training regime |
| D | Best supported core + geometric attention bias | Board-conditioned attention |
| E | Best core + one legal-map auxiliary loss | Additional self-generated supervision |
| F | Future-state/action objective, one-pass versus denoising | Diffusion-specific inference benefit |
| G | Best core at approximately 30M | Whether parameter scaling earns its cost |

Do not execute a Cartesian product. Screen cheaply, preserve the best control, and spend repeated seeds on a small number of informative finalists. A complexity increase advances only if its benefit exceeds uncertainty and its operational cost is acceptable.

## 10. The diffusion branch

This extension starts only after genuine self-play improvement and reliable evaluation. It is not part of Phase 0 or the first engine release.

### 10.1 Hypothesis and representation

Train a small bidirectional denoising module conditioned on the current board to recover masked short future trajectories. Begin with two plies and later consider four. Use discrete absorbing-mask corruption, not Gaussian noise over board images. Preserve current-board conditioning and supply a denoising-time embedding.

Represent each future board with square tokens and each move with source, destination, and promotion slots, plus the state metadata needed to interpret the trajectory. Avoid a giant action-token embedding that consumes the parameter budget. Two future boards already take the sequence from 64 squares to roughly 200 tokens with moves, so measure attention cost instead of assuming this head is cheap.

Targets come from verified completed self-play trajectories or explicitly tagged, legal deeper-search lines from our own model. Keep the root player’s orientation consistent over the whole trajectory. Future valid-frame masks are mandatory near terminal positions. Behavior trajectories are not automatically optimal minimax continuations; model/version and generating-policy provenance must be retained.

The initial branch should add no more than roughly 2M parameters if feasible, for example through a narrow two-block decoder conditioned on the board encoder. That is a design target to validate, not an excuse to conceal an oversized head. Report the new total model size.

### 10.2 Training and inference controls

Use masked-token reconstruction over future state/action slots. All unknown future slots must be masked at inference; a unit test must detect accidental use of ground-truth future moves or boards during evaluation. Do not mistake a teacher-forced validation score for free-running planning ability.

Compare: identical auxiliary targets with a one-pass predictor; discrete denoising with 2/4 rounds; and an ordinary search baseline given the same elapsed time. Only consider 8+ rounds if the smaller experiment earns them.

Start with root-only proposals rather than denoising at every tree leaf. Decode a candidate line, validate all actions through the exact rules engine, and compare predicted future boards against actual simulated transitions. Invalid lines are rejected visibly. Base legal policy/search remains the fallback; log rejection and fallback rates. The rules engine, not predicted boards, determines successors and outcomes.

Never reward a line merely because the model imagines cooperative opponent moves or an optimistic future value. Any candidate move ordering or extra root prior must be separately measured against strong defensive replies. The first goal is valid, useful proposals at a tolerable cost, not replacing adversarial search with generated stories.

## 11. Build sequence and acceptance gates

All durations below are **run budget proposals**, not estimates of coding completion or promised learning speed.

### Phase 0 — hardware and real backend proof

Implement read-only doctor output, pinned workspace/toolchain, Micro and F10 parameterized models sufficient to exercise the true graph, a reproducible forward/backward/optimizer benchmark, and checkpoint restore tests. Record CPU FP32 reference results, tested GPU modes, warm/cold metrics, memory, and exact commands. Benchmarks may use fixed synthetic targets for systems testing, labelled as such; they are not chess-learning results.

**Gate:** the selected backend passes real update/restore correctness and exposes useful throughput on actual hardware. If hardware is unavailable, do not claim GPU validation. Stop with the CPU evidence and the exact missing probe; do not build months of architecture on an assumed backend.

### Phase 1 — chess contracts

Implement versioned observation/action encoding, core game/history, explicit draw profile, perft, UCI conversions, property tests, and independent differential tests. Cover castling, en-passant pins, all promotion types, checks, stalemate, repetition, and clock-sensitive outcomes.

**Gate:** deterministic fixtures and round trips pass; random legal-play differential checks have no unexplained mismatches; policy masks contain every legal move exactly once; terminal cases do not enter softmax.

### Phase 2 — complete Micro vertical slice

Connect Micro neural inference, PUCT, independent game workers, bounded batching, replay writer/audit, learner, checkpoint resume, and arena. Complete small real self-play batches, train on their real targets, resume, and play a UCI game. Add the phase coordinator rather than relying on a notebook or a manually edited loop.

**Gate:** one documented command can run a bounded collect/train/evaluate cycle. Output games replay legally, schemas agree, values have the right perspective, completed outcomes are correct, and interruption recovery works. Do not demand a strong chess rating from this systems test.

### Phase 3 — first F10 baseline

Run a short smoke budget, then a roughly two-hour pilot. Inspect actual throughput and data quality before approving a 24-hour baseline. Preserve the starting checkpoint and evaluation protocol. Adjust only clearly motivated resource settings or fix correctness bugs; do not change architecture repeatedly within one untracked run.

**Gate:** clear accounting of data, compute, termination reasons, and learning signals. A failure to improve triggers diagnosis, not a fabricated success report or immediate parameter scaling.

### Phase 4 — Gumbel

Implement paper-faithful targets and search behavior, mathematical/reference tests, and a matched-budget comparison against PUCT on the same model family. Preserve PUCT.

**Gate:** algorithmic tests pass and the comparison has a documented result, including uncertainty. Gumbel may remain an experimental mode if it does not improve this regime.

### Phase 5 — recurrent refinement

Implement verified shared weights, R=1 parity, full-gradient training, normalized intermediate losses, and recurrence-aware caches/configs. Compare R=1/2/4 raw policy and searched modes, then matched-data and matched-time learning runs. A first serious comparison can use an approved 24–72-hour budget per finalist; revise this after the pilot.

**Gate:** evidence addresses the central question. More loops may win, tie, or lose. Keep the measured best default and retain the research branch.

### Phase 6 — one extension at a time

Try geometric bias, one auxiliary loss, then the denoising branch only as justified. Scale toward 30M only after measuring whether the 10M model is under-capacity rather than starved of fresh search data. Count all extra compute.

**Gate:** an extension earns its place by a repeatable benefit or an explicitly interesting negative result, not a fashionable name.

### Phase 7 — enjoyable usable release

Complete UCI stop/time handling, clear CLI help, safe pause/resume, run comparison reports, and a lightweight board viewer if desired by the owner. A useful distinctive viewer shows the top legal moves and WDL after 1/2/4 loops and lets the user replay positions where refinement helped or hurt. Label this as an observable prediction trace, not a decoded private reasoning process.

**Gate:** clean installation on the target machine, reproducible sample run, correct loading of saved models, no accidental retraining on startup, and no undocumented dependence on a developer’s local files.

## 12. Workspace and intended interface

Start with a small workspace, splitting crates only where interfaces warrant it:

```text
crates/
  recur64-core/       # rules wrapper, observations, actions, schemas
  recur64-model/      # model, training graph, precision, model records
  recur64-search/     # PUCT, Gumbel, tiny-tree tests
  recur64-runtime/    # inference owner, self-play, replay, learner coordinator
  recur64-eval/       # UCI, arenas, paired statistics, reports
  recur64-cli/        # commands/configuration
configs/
  smoke.toml
  f10.toml
  r10.toml
  resource-interactive.toml
  resource-unattended.toml
docs/
  ARCHITECTURE.md
  REPRESENTATIONS.md
  RULES_PROFILE.md
  EXPERIMENTS.md
  BENCHMARKS.md
  DECISIONS.md
  STATUS.md
```

Proposed commands to implement, not commands that already exist:

```text
recur64 doctor
recur64 model-info --config <file>
recur64 bench --config <file> --output <directory>
recur64 perft --fen <fen> --depth <n>
recur64 selfplay --config <file> --checkpoint <file> --output <directory>
recur64 replay-audit --input <directory>
recur64 train --config <file> --replay <directory> --run-dir <directory>
recur64 run --config <file> --run-dir <directory> --budget-minutes <n>
recur64 arena --config <file> --candidate <file> --reference <file>
recur64 uci --checkpoint <file>
recur64 report --run-dir <directory>
```

Design `run` to coordinate bounded self-play and learner phases with one GPU owner. Every command has a validated config, explicit run-directory behavior, structured errors, and a dry/read-only mode where appropriate. Paths may not overwrite an existing experiment without an explicit option and safety checks.

## 13. Agent operating model

Use GPT-6 Astra as initial architecture owner/integrator and DeepSeek V4.1 Flash for bounded implementation/test tickets; alternate independent reviews of critical code. This is a suggested work division, not a benchmark claim about either model. Use the user’s available agent harness and credentials; verify actual model IDs and availability in that environment instead of inventing them. The coding agents are separate from the locally trained chess model.

One agent owns shared contracts. Delegate narrow modules only after the relevant interfaces are written. Use separate branches/worktrees and non-overlapping file ownership. Do not have two agents simultaneously rewrite the representation or backend abstraction. Each implementation has an independent review of signs, masks, serialization, threading, and measured behavior.

A ticket includes objective, prerequisites, files owned, exact acceptance tests, benchmark requirements, prohibited scope expansion, outputs, and stop conditions. Review actual diffs and machine logs rather than accepting a persuasive completion summary. Failed/unrun tests remain visible. A passing mocked path is not proof that CUDA, persistence, or a full game works.

The first assignment is **Phase 0 only**, followed by an evidence-based decision. Do not ask either agent to build the entire research program in one unattended goal. See the separate kickoff prompt and `AGENTS.md` for execution discipline.

## 14. Completion criteria and decision rules

The core project is complete when a fresh checkout can build on the documented target environment, play legal full games through UCI, generate and audit its own replay, train and resume the selected small transformer, run controlled arenas, and reproduce the claimed measurements from recorded configurations. The recurrent hypothesis is complete when comparisons under the relevant budgets have a credible result, even if recurrence loses.

Do not move to 30M merely because it fits in memory. Do not retain diffusion merely because its training loss falls. Do not claim that Rust automatically improves neural-network kernel speed. Do not claim superhuman chess, a platform rating, or a training-time forecast without appropriate measurements.

The desired outcome is a small, understandable laboratory with an actual “computation dial,” a playable model, and evidence for where this workstation’s compute is best spent.

## 15. Primary-source reference register

These references support the research findings and external API/rule observations above. Model dimensions, proposed experiment budgets, architecture adaptations, and acceptance criteria in this specification are our design choices. Versions and dependency APIs must be rechecked during Phase 0.

- **S01 — Chessformer: A Unified Architecture for Chess Modeling (2026).** https://arxiv.org/html/2605.19091v1
- **S02 — Hierarchical Reasoning Model (2025).** https://arxiv.org/abs/2506.21734
- **S03 — Less is More: Recursive Reasoning with Tiny Networks (2025).** https://arxiv.org/html/2510.04871v1
- **S04 — Are Your Reasoning Models Reasoning or Guessing? A Mechanistic Analysis of Hierarchical Reasoning Models (2026, v2).** https://arxiv.org/abs/2601.10679
- **S05 — Implicit Search via Discrete Diffusion: A Study on Chess (2025).** https://arxiv.org/html/2502.19805v1 ; authors’ implementation: https://github.com/HKUNLP/DiffuSearch
- **S06 — Policy improvement by planning with Gumbel (ICLR 2022).** https://openreview.net/forum?id=bERaNdoegnO ; author-hosted paper: https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/gumbel-alphazero.pdf
- **S07 — Google DeepMind mctx reference implementation.** https://github.com/google-deepmind/mctx
- **S08 — Simplifying Deep Reinforcement Learning via Self-Supervision (2021).** https://arxiv.org/abs/2106.05526
- **S09 — Burn official repository and backend notes.** https://github.com/tracel-ai/burn
- **S10 — Burn Book.** https://burn.dev/books/burn/overview.html
- **S11 — Hugging Face Candle.** https://github.com/huggingface/candle
- **S12 — tch-rs.** https://github.com/LaurentMazare/tch-rs
- **S13 — cozy-chess and its documentation.** https://github.com/analog-hors/cozy-chess ; https://docs.rs/cozy-chess/latest/cozy_chess/
- **S14 — shakmaty.** https://github.com/niklasf/shakmaty
- **S15 — NVIDIA CUDA on WSL user guide.** https://docs.nvidia.com/cuda/wsl-user-guide/index.html
- **S16 — FIDE Laws of Chess.** https://handbook.fide.com/chapter/E012023
- **S17 — Accelerating Self-Play Learning in Go (2019), optional efficiency background, not direct evidence for this chess regime.** https://arxiv.org/abs/1902.10565
