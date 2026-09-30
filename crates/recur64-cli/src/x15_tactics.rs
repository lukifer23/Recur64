//! `recur64 x15 gen-tactics | eval-tactics` - the deterministic tactical /
//! conversion suite (P6).
//!
//! Fixtures are FEN-only positions with fresh clocks and NO repetition history
//! (they say so in `history`). Labels:
//!
//! * `mate_*`: every legal move that delivers checkmate (exact under any rules:
//!   a fresh-clock position's mate is decided by the board alone).
//! * `material_gain` / `promotion`: the move(s) with the best 2-ply (own move,
//!   best opponent reply) material outcome, kept only when it beats every other
//!   move by a stated margin. This is a bounded material claim, not "best move".
//! * `hp_repetition_fen`: final positions of Train1 arena games that ended by
//!   threefold repetition, with the historical repetition state LOST. No move
//!   label; they probe whether the value head recognises a material lead.

use std::path::PathBuf;

use burn::prelude::*;
use burn::tensor::activation;
use clap::Args;
use serde::{Deserialize, Serialize};

use recur64_core::{ActionId, GameState, StandardMove, Termination};
use recur64_model::checkpoint::CheckpointMeta;
use recur64_model::config::ProbeConfig;
use recur64_runtime::model_io;
use recur64_runtime::x15_inputs::{build_x15_batch, build_x15_batch_padded, provider_for_config};

#[derive(Args, Debug)]
pub struct GenTacticsArgs {
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long, default_value_t = 20261001)]
    pub seed: u64,
    /// Fixtures per synthetic kind.
    #[arg(long, default_value_t = 8)]
    pub per_kind: usize,
    /// Arena JSON to mine repetition-draw final FENs from.
    #[arg(
        long,
        default_value = "docs/evidence/train1/final-arena/eval-arena.json"
    )]
    pub arena: PathBuf,
    /// Fail unless no fixture FEN appears in any of these fixtures files.
    #[arg(long)]
    pub disjoint_fixtures: Vec<PathBuf>,
    /// Fail unless no fixture FEN appears as a start position in these targets files.
    #[arg(long)]
    pub disjoint_targets: Vec<PathBuf>,
    /// Repetition-draw FEN fixtures to include.
    #[arg(long, default_value_t = 24)]
    pub hp_draws: usize,
}

#[derive(Args, Debug)]
pub struct EvalTacticsArgs {
    #[arg(long, default_value = "configs/x15_cuda.toml")]
    pub config: PathBuf,
    #[arg(long)]
    pub fixtures: PathBuf,
    #[arg(long)]
    pub checkpoint: Vec<PathBuf>,
    #[arg(long, default_value_t = 4)]
    pub thoughts: usize,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Fixture {
    pub id: String,
    pub kind: String,
    pub fen: String,
    /// `fen_only_fresh_clocks` or `fen_only_repetition_history_lost`.
    pub history: String,
    /// Indices into `GameState::legal_actions()` that count as correct.
    pub correct: Vec<usize>,
    /// Side-to-move material minus opponent material (pawn units).
    pub material_lead: i32,
    pub note: String,
}

#[derive(Serialize, Deserialize)]
pub struct FixtureFile {
    pub schema: String,
    pub seed: u64,
    pub fixtures: Vec<Fixture>,
}

pub(crate) fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

pub(crate) struct Rng(pub(crate) u64);
impl Rng {
    pub(crate) fn next(&mut self) -> u64 {
        self.0 = mix(self.0);
        self.0
    }
    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn value(c: char) -> i32 {
    match c.to_ascii_lowercase() {
        'p' => 1,
        'n' | 'b' => 3,
        'r' => 5,
        'q' => 9,
        _ => 0,
    }
}

/// White minus black material from a FEN.
pub(crate) fn material_diff(fen: &str) -> i32 {
    fen.split(' ')
        .next()
        .unwrap_or("")
        .chars()
        .filter(|c| c.is_ascii_alphabetic())
        .map(|c| {
            if c.is_ascii_uppercase() {
                value(c)
            } else {
                -value(c)
            }
        })
        .sum()
}

pub(crate) fn apply_id(state: &GameState, id: ActionId) -> GameState {
    let (from, to, promo) = id.to_physical(state.perspective());
    let promotion = if promo.is_none() { None } else { Some(promo) };
    let mut s = state.clone();
    s.apply(StandardMove::new(from, to, promotion))
        .expect("legal action applies");
    s
}

fn is_checkmate(s: &GameState) -> bool {
    s.termination() == Some(Termination::Checkmate)
}

/// Indices of every legal move that mates.
pub(crate) fn mating_moves(state: &GameState) -> Vec<usize> {
    state
        .legal_actions()
        .iter()
        .enumerate()
        .filter(|(_, id)| is_checkmate(&apply_id(state, **id)))
        .map(|(i, _)| i)
        .collect()
}

/// 2-ply material value of each legal move for the side to move (in the mover's
/// favour): own move, then the opponent's best reply.
fn two_ply_values(state: &GameState) -> Vec<i32> {
    let white = state.side_to_move() == recur64_core::Color::White;
    let sign = if white { 1 } else { -1 };
    state
        .legal_actions()
        .iter()
        .map(|id| {
            let s1 = apply_id(state, *id);
            if is_checkmate(&s1) {
                return 1000;
            }
            let replies = s1.legal_actions();
            if replies.is_empty() || s1.is_terminal() {
                return sign * material_diff(&s1.to_fen());
            }
            replies
                .iter()
                .map(|r| {
                    let s2 = apply_id(&s1, *r);
                    if is_checkmate(&s2) {
                        -1000
                    } else {
                        sign * material_diff(&s2.to_fen())
                    }
                })
                .min()
                .unwrap_or(0)
        })
        .collect()
}

/// Build a FEN from (square, piece char) placements; white to move, no rights.
pub(crate) fn fen_from(pieces: &[(usize, char)]) -> String {
    let mut board = [' '; 64];
    for (sq, c) in pieces {
        board[*sq] = *c;
    }
    let mut out = String::new();
    for rank in (0..8).rev() {
        let mut empty = 0;
        for file in 0..8 {
            let c = board[rank * 8 + file];
            if c == ' ' {
                empty += 1;
            } else {
                if empty > 0 {
                    out.push_str(&empty.to_string());
                    empty = 0;
                }
                out.push(c);
            }
        }
        if empty > 0 {
            out.push_str(&empty.to_string());
        }
        if rank > 0 {
            out.push('/');
        }
    }
    out.push_str(" w - - 0 1");
    out
}

/// Random placement of `pieces` on distinct squares (pawns on ranks 2-7).
pub(crate) fn place(rng: &mut Rng, pieces: &[char]) -> Vec<(usize, char)> {
    let mut used = [false; 64];
    let mut out = Vec::new();
    for &c in pieces {
        loop {
            let sq = rng.below(64);
            let rank = sq / 8;
            if used[sq] || (c.eq_ignore_ascii_case(&'p') && !(1..=6).contains(&rank)) {
                continue;
            }
            used[sq] = true;
            out.push((sq, c));
            break;
        }
    }
    out
}

pub(crate) fn try_state(fen: &str) -> Option<GameState> {
    let s = GameState::from_fen(fen).ok()?;
    // The side NOT to move must not already be in check (an illegal position
    // `from_fen` does not reject, and one in which the king could be captured).
    // A null move flips the side; if it is refused the mover is in check.
    if let Some(flipped) = s.board().null_move()
        && !flipped.checkers().is_empty()
    {
        return None;
    }
    (!s.is_terminal() && !s.legal_actions().is_empty()).then_some(s)
}

fn gen_mates(rng: &mut Rng, kind: &str, white: &[char], n: usize) -> anyhow::Result<Vec<Fixture>> {
    let mut out = Vec::new();
    let mut tries = 0u64;
    while out.len() < n {
        tries += 1;
        anyhow::ensure!(tries < 4_000_000, "could not find {n} {kind} positions");
        let mut pieces: Vec<char> = white.to_vec();
        pieces.push('k');
        let placed = place(rng, &pieces);
        if kings_adjacent(&placed) {
            continue;
        }
        let fen = fen_from(&placed);
        let Some(state) = try_state(&fen) else {
            continue;
        };
        let mates = mating_moves(&state);
        if mates.is_empty() {
            continue;
        }
        out.push(Fixture {
            id: format!("{kind}-{:02}", out.len()),
            kind: kind.into(),
            fen,
            history: "fen_only_fresh_clocks".into(),
            correct: mates,
            material_lead: material_diff(&state.to_fen()),
            note: "every mating move is correct".into(),
        });
    }
    Ok(out)
}

fn gen_material(
    rng: &mut Rng,
    kind: &str,
    promotion: bool,
    n: usize,
) -> anyhow::Result<Vec<Fixture>> {
    let mut out = Vec::new();
    let mut tries = 0u64;
    let pool = ['Q', 'R', 'B', 'N', 'P'];
    while out.len() < n {
        tries += 1;
        anyhow::ensure!(tries < 2_000_000, "could not find {n} {kind} positions");
        let mut pieces = vec!['K', 'k'];
        if promotion {
            pieces.push('P');
        } else {
            for _ in 0..1 + rng.below(2) {
                pieces.push(pool[rng.below(5)]);
            }
        }
        for _ in 0..1 + rng.below(3) {
            pieces.push(pool[rng.below(5)].to_ascii_lowercase());
        }
        let mut placed = place(rng, &pieces);
        if promotion {
            // Put the white pawn on the 7th rank.
            let file = rng.below(8);
            let pawn_sq = 6 * 8 + file;
            if placed.iter().any(|(s, _)| *s == pawn_sq) {
                continue;
            }
            for p in placed.iter_mut() {
                if p.1 == 'P' {
                    p.0 = pawn_sq;
                }
            }
        }
        if kings_adjacent(&placed) {
            continue;
        }
        let fen = fen_from(&placed);
        let Some(state) = try_state(&fen) else {
            continue;
        };
        let vals = two_ply_values(&state);
        let best = *vals.iter().max().unwrap();
        let mut sorted = vals.clone();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        let uniq_best = sorted.iter().take_while(|v| **v == best).count();
        let second = sorted.get(uniq_best).copied().unwrap_or(-1000);
        let now = material_diff(&state.to_fen());
        // Keep only clear cases: gains >= 3 over the status quo and beats every
        // non-best move by >= 3 (promotion: by >= 2).
        let margin = if promotion { 2 } else { 3 };
        if best >= 1000 || best - now < if promotion { 5 } else { 3 } || best - second < margin {
            continue;
        }
        let correct: Vec<usize> = vals
            .iter()
            .enumerate()
            .filter(|(_, v)| **v == best)
            .map(|(i, _)| i)
            .collect();
        out.push(Fixture {
            id: format!("{kind}-{:02}", out.len()),
            kind: kind.into(),
            fen,
            history: "fen_only_fresh_clocks".into(),
            correct,
            material_lead: now,
            note: format!("2-ply material {best} vs next best {second} (status quo {now})"),
        });
    }
    Ok(out)
}

fn gen_hp_draws(path: &std::path::Path, n: usize, seed: u64) -> anyhow::Result<Vec<Fixture>> {
    let text = std::fs::read_to_string(path)?;
    let v: serde_json::Value = serde_json::from_str(&text)?;
    let mut cands: Vec<(u64, Fixture)> = Vec::new();
    let mut stack = vec![&v];
    while let Some(x) = stack.pop() {
        match x {
            serde_json::Value::Object(m) => {
                if m.get("termination").and_then(|t| t.as_str()) == Some("threefold_repetition")
                    && let Some(fen) = m.get("final_fen").and_then(|f| f.as_str())
                    && let Some(state) = try_state(fen)
                {
                    let white = state.side_to_move() == recur64_core::Color::White;
                    let diff = material_diff(&state.to_fen());
                    let lead = if white { diff } else { -diff };
                    let idx = m.get("index").and_then(|i| i.as_u64()).unwrap_or(0);
                    cands.push((
                                mix(seed ^ mix(idx)),
                                Fixture {
                                    id: format!("hp-draw-{idx:03}"),
                                    kind: "hp_repetition_fen".into(),
                                    fen: fen.into(),
                                    history: "fen_only_repetition_history_lost".into(),
                                    correct: vec![],
                                    material_lead: lead,
                                    note: "final FEN of a Train1 arena game that ended by threefold repetition; repetition history NOT reconstructable".into(),
                                },
                            ));
                }
                stack.extend(m.values());
            }
            serde_json::Value::Array(a) => stack.extend(a.iter()),
            _ => {}
        }
    }
    // Prefer positions where one side clearly leads, then seeded order.
    cands.retain(|(_, f)| f.material_lead.abs() >= 3);
    cands.sort_by_key(|(k, f)| (*k, f.id.clone()));
    Ok(cands.into_iter().take(n).map(|(_, f)| f).collect())
}

pub fn run_gen(args: GenTacticsArgs) -> anyhow::Result<()> {
    let mut rng = Rng(args.seed);
    let mut fixtures = Vec::new();
    for (kind, white) in [
        ("mate_kqk", vec!['K', 'Q']),
        ("mate_krk", vec!['K', 'R']),
        ("mate_kqqk", vec!['K', 'Q', 'Q']),
        ("mate_kqrk", vec!['K', 'Q', 'R']),
        ("mate_krrk", vec!['K', 'R', 'R']),
    ] {
        fixtures.extend(gen_mates(&mut rng, kind, &white, args.per_kind)?);
    }
    fixtures.extend(gen_material(
        &mut rng,
        "material_gain",
        false,
        args.per_kind,
    )?);
    fixtures.extend(gen_material(&mut rng, "promotion", true, args.per_kind)?);
    fixtures.extend(gen_hp_draws(&args.arena, args.hp_draws, args.seed)?);
    // Hard disjointness checks (not filters): the set must share no FEN with
    // the evaluation suite or with training start positions.
    let mut forbidden: std::collections::HashSet<String> = std::collections::HashSet::new();
    for p in &args.disjoint_fixtures {
        forbidden.extend(fixture_fens(p)?);
    }
    for p in &args.disjoint_targets {
        let t = recur64_runtime::reasoning_targets::ReasoningTargetsV1::load(p)?;
        forbidden.extend(t.positions.iter().map(|q| q.start_fen.clone()));
        forbidden.extend(t.positions.iter().map(|q| q.fen.clone()));
    }
    let clashes: Vec<_> = fixtures
        .iter()
        .filter(|f| forbidden.contains(&f.fen))
        .map(|f| f.id.clone())
        .collect();
    anyhow::ensure!(
        clashes.is_empty(),
        "{} fixtures overlap the forbidden FEN set: {:?}",
        clashes.len(),
        clashes
    );
    if !forbidden.is_empty() {
        println!(
            "verified: no fixture FEN overlaps {} forbidden FENs",
            forbidden.len()
        );
    }
    let mut counts = std::collections::BTreeMap::new();
    for f in &fixtures {
        *counts.entry(f.kind.clone()).or_insert(0usize) += 1;
    }
    let file = FixtureFile {
        schema: "x15_tactics_v1".into(),
        seed: args.seed,
        fixtures,
    };
    if let Some(dir) = args.out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&args.out, serde_json::to_vec_pretty(&file)?)?;
    println!(
        "wrote {} ({} fixtures): {counts:?}",
        args.out.display(),
        file.fixtures.len()
    );
    println!(
        "history: synthetic kinds have fresh clocks and no history; hp_repetition_fen has LOST repetition history (FEN-only)"
    );
    Ok(())
}

// --- evaluation -----------------------------------------------------------------

fn eval_one<B: Backend>(
    cfg: &ProbeConfig,
    fixtures: &[Fixture],
    ck: &std::path::Path,
    tmax: usize,
    device: &B::Device,
) -> anyhow::Result<()> {
    let states: Vec<GameState> = fixtures
        .iter()
        .map(|f| GameState::from_fen(&f.fen).map_err(|e| anyhow::anyhow!("{}: {e}", f.id)))
        .collect::<anyhow::Result<_>>()?;
    // Inputs are built from the CHECKPOINT'S experimental contract, so variants
    // with and without compute / visual / candidate facts can share one command.
    let meta: CheckpointMeta = serde_json::from_slice(&std::fs::read(ck.join("meta.json"))?)?;
    let provider = provider_for_config(&meta.experimental)?;
    let batch = build_x15_batch::<B>(&states, &meta.experimental, provider.as_ref(), device)?;
    let model = model_io::load_chimera::<B>(ck, &cfg.model, &meta.experimental, device)?;
    let tmax = if meta.experimental.reasoning.enabled {
        tmax
    } else {
        1
    };
    let out = model.forward_thoughts_diagnostic(&batch.input, &batch.cands, tmax);
    let width = batch.cands.width;
    println!("checkpoint {} (T=1..{tmax})", ck.display());
    let mut kinds: Vec<String> = fixtures.iter().map(|f| f.kind.clone()).collect();
    kinds.sort();
    kinds.dedup();
    println!(
        "  {:<18} {:>3} {:>2}  {:>9} {:>9} {:>7} {:>7} {:>8} {:>10}",
        "kind", "n", "T", "top1_ok", "mass_ok", "P(win)", "P(draw)", "entropy", "lead_agree"
    );
    for kind in kinds {
        let idx: Vec<usize> = (0..fixtures.len())
            .filter(|i| fixtures[*i].kind == kind)
            .collect();
        for (k, r) in out.readouts.iter().enumerate() {
            let lp = r
                .policy
                .log_probs
                .clone()
                .into_data()
                .to_vec::<f32>()
                .unwrap_or_default();
            let wdl = activation::softmax(r.wdl_logits.clone(), 1)
                .into_data()
                .to_vec::<f32>()
                .unwrap_or_default();
            let (mut ok, mut mass, mut pw, mut pd, mut ent, mut agree) =
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            let mut labelled = 0.0;
            for &i in &idx {
                let f = &fixtures[i];
                let n_legal = states[i].legal_actions().len();
                let row = &lp[i * width..i * width + n_legal];
                if !f.correct.is_empty() {
                    labelled += 1.0;
                    let arg = row
                        .iter()
                        .enumerate()
                        .fold(0usize, |b, (j, v)| if *v > row[b] { j } else { b });
                    ok += f32::from(f.correct.contains(&arg));
                    mass += f.correct.iter().map(|c| row[*c].exp()).sum::<f32>();
                }
                let (w, d, l) = (wdl[i * 3], wdl[i * 3 + 1], wdl[i * 3 + 2]);
                pw += w;
                pd += d;
                ent += -row.iter().map(|x| x.exp() * x).sum::<f32>();
                // Value sign agrees with the side-to-move material lead.
                if f.material_lead != 0 {
                    agree += f32::from((w - l).signum() == (f.material_lead as f32).signum());
                }
            }
            let n = idx.len() as f32;
            let acc = if labelled > 0.0 {
                format!("{:.2}", ok / labelled)
            } else {
                "-".into()
            };
            let ms = if labelled > 0.0 {
                format!("{:.2}", mass / labelled)
            } else {
                "-".into()
            };
            println!(
                "  {:<18} {:>3} {:>2}  {:>9} {:>9} {:>7.3} {:>7.3} {:>8.3} {:>10.2}",
                kind,
                idx.len(),
                k + 1,
                acc,
                ms,
                pw / n,
                pd / n,
                ent / n,
                agree / n
            );
        }
    }
    Ok(())
}

pub fn run_eval(args: EvalTacticsArgs) -> anyhow::Result<()> {
    let cfg = ProbeConfig::from_toml_str(&std::fs::read_to_string(&args.config)?)?;
    cfg.experimental.validate(cfg.model.width)?;
    let file: FixtureFile = serde_json::from_slice(&std::fs::read(&args.fixtures)?)?;
    anyhow::ensure!(file.schema == "x15_tactics_v1", "unknown fixture schema");
    match cfg.device {
        recur64_model::config::DeviceKind::Cpu => {
            let device: Device<burn::backend::Flex> = Default::default();
            for ck in &args.checkpoint {
                eval_one::<burn::backend::Flex>(&cfg, &file.fixtures, ck, args.thoughts, &device)?;
            }
            Ok(())
        }
        recur64_model::config::DeviceKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                let device: Device<burn::backend::Cuda> = Default::default();
                for ck in &args.checkpoint {
                    eval_one::<burn::backend::Cuda>(
                        &cfg,
                        &file.fixtures,
                        ck,
                        args.thoughts,
                        &device,
                    )?;
                }
                Ok(())
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA support is not compiled; rebuild with --features cuda")
            }
        }
    }
}

/// Synthetic tactic positions for TRAINING data: `per_kind` of each mate set,
/// `material_gain` and `promotion`, from `seed`, skipping any FEN in `exclude`
/// (the evaluation suite) and duplicates. Returns `(kind, fen)`.
pub(crate) fn synth_fens(
    seed: u64,
    per_kind: usize,
    exclude: &std::collections::HashSet<String>,
) -> anyhow::Result<Vec<(String, String)>> {
    let mut rng = Rng(seed);
    let mut out: Vec<(String, String)> = Vec::new();
    let mut seen: std::collections::HashSet<String> = exclude.clone();
    let mut take = |fs: Vec<Fixture>, out: &mut Vec<(String, String)>| {
        let mut n = 0;
        for f in fs {
            if n < per_kind && seen.insert(f.fen.clone()) {
                out.push((f.kind.clone(), f.fen));
                n += 1;
            }
        }
        n
    };
    for (kind, white) in [
        ("mate_kqk", vec!['K', 'Q']),
        ("mate_krk", vec!['K', 'R']),
        ("mate_kqqk", vec!['K', 'Q', 'Q']),
        ("mate_kqrk", vec!['K', 'Q', 'R']),
        ("mate_krrk", vec!['K', 'R', 'R']),
    ] {
        let fs = gen_mates(&mut rng, kind, &white, per_kind + 16)?;
        anyhow::ensure!(take(fs, &mut out) == per_kind, "not enough unique {kind}");
    }
    for (kind, promo) in [("material_gain", false), ("promotion", true)] {
        let fs = gen_material(&mut rng, kind, promo, per_kind + 16)?;
        anyhow::ensure!(take(fs, &mut out) == per_kind, "not enough unique {kind}");
    }
    Ok(out)
}

/// FEN strings of an evaluation fixtures file.
pub(crate) fn fixture_fens(path: &std::path::Path) -> anyhow::Result<Vec<String>> {
    let file: FixtureFile = serde_json::from_slice(&std::fs::read(path)?)?;
    Ok(file.fixtures.into_iter().map(|f| f.fen).collect())
}

/// Adjacent kings make an illegal position that `from_fen` does not reject.
pub(crate) fn kings_adjacent(placed: &[(usize, char)]) -> bool {
    let sq = |c: char| placed.iter().find(|(_, p)| *p == c).map(|(s, _)| *s);
    match (sq('K'), sq('k')) {
        (Some(a), Some(b)) => {
            let (df, dr) = ((a % 8).abs_diff(b % 8), (a / 8).abs_diff(b / 8));
            df <= 1 && dr <= 1
        }
        _ => true,
    }
}

// --- teacher reference (probe network + deterministic PUCT) --------------------------

#[derive(Args, Debug)]
pub struct TeacherTacticsArgs {
    #[arg(long)]
    pub teacher_config: PathBuf,
    #[arg(long)]
    pub teacher_checkpoint: PathBuf,
    #[arg(long)]
    pub fixtures: PathBuf,
    /// Simulation counts to search with (deterministic PUCT, no noise).
    #[arg(long, value_delimiter = ',', default_value = "16,64,128")]
    pub sims: Vec<u32>,
    #[arg(long, default_value_t = 6)]
    pub threads: usize,
}

fn teacher_impl<B: Backend>(
    cfg: &recur64_runtime::RunConfig,
    args: &TeacherTacticsArgs,
    file: &FixtureFile,
) -> anyhow::Result<()> {
    use recur64_runtime::evaluator::SyncEvaluator;
    use recur64_search::game_tree::ChessGame;
    use recur64_search::puct::{PuctConfig, search};

    let device: B::Device = Default::default();
    let model = model_io::load::<B>(&args.teacher_checkpoint, &cfg.model, &device)?;
    let evaluator = SyncEvaluator::new(model, cfg.recurrence, device);
    println!(
        "teacher reference (probe network + deterministic PUCT, leaves_in_flight 1, no noise)"
    );
    println!(
        "  {:<18} {:>3} {:>5}  {:>9} {:>9}",
        "kind", "n", "sims", "top1_ok", "mass_ok"
    );
    let mut kinds: Vec<String> = file.fixtures.iter().map(|f| f.kind.clone()).collect();
    kinds.sort();
    kinds.dedup();
    for &sims in &args.sims {
        // (fixture index) -> (top1 correct, mass on correct)
        let chunk = file.fixtures.len().div_ceil(args.threads.max(1)).max(1);
        let results: Vec<(usize, f32, f32)> = std::thread::scope(|scope| {
            let handles: Vec<_> = file
                .fixtures
                .chunks(chunk)
                .enumerate()
                .map(|(ci, fs)| {
                    let evaluator = &evaluator;
                    scope.spawn(move || {
                        fs.iter()
                            .enumerate()
                            .filter(|(_, f)| !f.correct.is_empty())
                            .map(|(j, f)| {
                                let state = GameState::from_fen(&f.fen).expect("fen");
                                let legal = state.legal_actions();
                                let res = search(
                                    ChessGame::new(state, evaluator),
                                    &PuctConfig {
                                        c_puct: 1.0,
                                        simulations: sims,
                                        leaves_in_flight: 1,
                                    },
                                )
                                .expect("search");
                                let total = res.total_visits.max(1) as f32;
                                let mass: f32 = res
                                    .edges
                                    .iter()
                                    .filter(|e| {
                                        legal
                                            .iter()
                                            .position(|a| *a == e.action)
                                            .is_some_and(|p| f.correct.contains(&p))
                                    })
                                    .map(|e| e.visits as f32)
                                    .sum::<f32>()
                                    / total;
                                let best = res.best_action();
                                let ok = best
                                    .and_then(|b| legal.iter().position(|a| *a == b))
                                    .is_some_and(|p| f.correct.contains(&p));
                                (ci * chunk + j, f32::from(ok), mass)
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap_or_default())
                .collect()
        });
        for kind in &kinds {
            let rows: Vec<_> = results
                .iter()
                .filter(|(i, _, _)| file.fixtures[*i].kind == *kind)
                .collect();
            if rows.is_empty() {
                continue;
            }
            let n = rows.len() as f32;
            println!(
                "  {:<18} {:>3} {:>5}  {:>9.2} {:>9.2}",
                kind,
                rows.len(),
                sims,
                rows.iter().map(|r| r.1).sum::<f32>() / n,
                rows.iter().map(|r| r.2).sum::<f32>() / n
            );
        }
    }
    Ok(())
}

pub fn run_teacher(args: TeacherTacticsArgs) -> anyhow::Result<()> {
    let cfg =
        recur64_runtime::RunConfig::from_toml_str(&std::fs::read_to_string(&args.teacher_config)?)?;
    cfg.ensure_supported()?;
    let file: FixtureFile = serde_json::from_slice(&std::fs::read(&args.fixtures)?)?;
    match cfg.device.as_str() {
        "cpu" => teacher_impl::<burn::backend::Flex>(&cfg, &args, &file),
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                teacher_impl::<burn::backend::Cuda>(&cfg, &args, &file)
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA support is not compiled; rebuild with --features cuda")
            }
        }
        other => anyhow::bail!("unsupported device {other:?}"),
    }
}

/// Per-fixture top-1 correctness of one checkpoint at `t` thoughts (labelled
/// fixtures only), as `(kind, 0.0 | 1.0)`. Inputs come from the checkpoint's own
/// experimental contract.
pub(crate) fn tactic_vector<B: Backend>(
    cfg: &ProbeConfig,
    fixtures: &[Fixture],
    ck: &std::path::Path,
    t: usize,
    device: &B::Device,
) -> anyhow::Result<Vec<(String, f32)>> {
    let states: Vec<GameState> = fixtures
        .iter()
        .map(|f| GameState::from_fen(&f.fen).map_err(|e| anyhow::anyhow!("{}: {e}", f.id)))
        .collect::<anyhow::Result<_>>()?;
    let meta: CheckpointMeta = serde_json::from_slice(&std::fs::read(ck.join("meta.json"))?)?;
    let provider = provider_for_config(&meta.experimental)?;
    let batch = build_x15_batch::<B>(&states, &meta.experimental, provider.as_ref(), device)?;
    let model = model_io::load_chimera::<B>(ck, &cfg.model, &meta.experimental, device)?;
    let t = if meta.experimental.reasoning.enabled {
        t
    } else {
        1
    };
    let out = model.forward_thoughts_diagnostic(&batch.input, &batch.cands, t);
    let width = batch.cands.width;
    let lp = out.readouts[t - 1]
        .policy
        .log_probs
        .clone()
        .into_data()
        .to_vec::<f32>()
        .unwrap_or_default();
    let mut v = Vec::new();
    for (i, f) in fixtures.iter().enumerate() {
        if f.correct.is_empty() {
            continue;
        }
        let n = states[i].legal_actions().len();
        let row = &lp[i * width..i * width + n];
        let arg = row
            .iter()
            .enumerate()
            .fold(0usize, |b, (j, x)| if *x > row[b] { j } else { b });
        v.push((f.kind.clone(), f32::from(f.correct.contains(&arg))));
    }
    Ok(v)
}

// --- facts probe ------------------------------------------------------------------

#[derive(Args, Debug)]
pub struct FactsProbeArgs {
    #[arg(long, default_value = "configs/x15_cuda.toml")]
    pub config: PathBuf,
    #[arg(long)]
    pub fixtures: PathBuf,
    #[arg(long)]
    pub checkpoint: Vec<PathBuf>,
}

fn facts_probe_one<B: Backend>(
    cfg: &ProbeConfig,
    fixtures: &[Fixture],
    ck: &std::path::Path,
    device: &B::Device,
) -> anyhow::Result<()> {
    let states: Vec<GameState> = fixtures
        .iter()
        .map(|f| GameState::from_fen(&f.fen).map_err(|e| anyhow::anyhow!("{}: {e}", f.id)))
        .collect::<anyhow::Result<_>>()?;
    let meta: CheckpointMeta = serde_json::from_slice(&std::fs::read(ck.join("meta.json"))?)?;
    anyhow::ensure!(
        meta.experimental.candidate_facts.enabled,
        "{} has no candidate facts",
        ck.display()
    );
    let provider = provider_for_config(&meta.experimental)?;
    let batch = build_x15_batch::<B>(&states, &meta.experimental, provider.as_ref(), device)?;
    let model = model_io::load_chimera::<B>(ck, &cfg.model, &meta.experimental, device)?;
    let width = batch.cands.width;
    let bias = model
        .facts_bias_raw(batch.input.cand_facts.clone().expect("facts supplied"))
        .into_data()
        .to_vec::<f32>()
        .unwrap_or_default();
    let (mut on, mut off) = (Vec::new(), Vec::new());
    for (i, f) in fixtures.iter().enumerate() {
        if f.correct.is_empty() {
            continue;
        }
        let n = states[i].legal_actions().len();
        for k in 0..n {
            let b = bias[i * width + k];
            if f.correct.contains(&k) {
                on.push(b);
            } else {
                off.push(b);
            }
        }
    }
    let m = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
    println!(
        "{}: fact-bias on correct moves {:+.4} (n={}), on other legal moves {:+.4} (n={}), gap {:+.4} logits",
        ck.display(),
        m(&on),
        on.len(),
        m(&off),
        off.len(),
        m(&on) - m(&off)
    );
    Ok(())
}

pub fn run_facts_probe(args: FactsProbeArgs) -> anyhow::Result<()> {
    let cfg = ProbeConfig::from_toml_str(&std::fs::read_to_string(&args.config)?)?;
    let file: FixtureFile = serde_json::from_slice(&std::fs::read(&args.fixtures)?)?;
    #[cfg(feature = "cuda")]
    {
        let device: Device<burn::backend::Cuda> = Default::default();
        for ck in &args.checkpoint {
            facts_probe_one::<burn::backend::Cuda>(&cfg, &file.fixtures, ck, &device)?;
        }
        Ok(())
    }
    #[cfg(not(feature = "cuda"))]
    {
        let device: Device<burn::backend::Flex> = Default::default();
        for ck in &args.checkpoint {
            facts_probe_one::<burn::backend::Flex>(&cfg, &file.fixtures, ck, &device)?;
        }
        Ok(())
    }
}

// --- exact mate-in-2 (rule-exact for fresh-clock, no-history positions) -------------

use cozy_chess::{Board as CBoard, GameStatus};

fn cboard_moves(b: &CBoard) -> Vec<cozy_chess::Move> {
    let mut v = Vec::new();
    b.generate_moves(|mvs| {
        v.extend(mvs);
        false
    });
    v
}

/// The side to move can deliver checkmate this move.
fn has_mate_in_one(b: &CBoard) -> bool {
    cboard_moves(b).into_iter().any(|m| {
        let mut n = b.clone();
        n.play(m);
        n.status() == GameStatus::Won
    })
}

/// `after` (opponent to move) is a forced loss in one more move for the
/// opponent: it is not already over, the opponent has replies, and after EVERY
/// reply the original mover has a mate in one.
///
/// Exactness: the fixtures are fresh-clock, no-history positions, so a 4-ply
/// sequence cannot reach a threefold repetition or the fifty-move claim; the
/// only terminations are checkmate, stalemate and insufficient material, and
/// any reply that leads to a stalemate or dead position simply has no mate in
/// one and fails the test. This is an assumption of the FIXTURE convention, not
/// a claim about arbitrary positions with history.
fn forced_mate_after(after: &CBoard) -> bool {
    if after.status() != GameStatus::Ongoing {
        return false;
    }
    let replies = cboard_moves(after);
    if replies.is_empty() {
        return false;
    }
    replies.into_iter().all(|r| {
        let mut n = after.clone();
        n.play(r);
        n.status() == GameStatus::Ongoing && has_mate_in_one(&n)
    })
}

/// Legal-action indices of every first move that forces mate in two, for a
/// position with no mate in one.
pub(crate) fn mate_in_two_moves(state: &GameState) -> Vec<usize> {
    if has_mate_in_one(state.board()) {
        return Vec::new();
    }
    state
        .legal_actions()
        .iter()
        .enumerate()
        .filter(|(_, id)| forced_mate_after(apply_id(state, **id).board()))
        .map(|(i, _)| i)
        .collect()
}

fn gen_mate2(
    rng: &mut Rng,
    kind: &str,
    white: &[char],
    n: usize,
    seen: &mut std::collections::HashSet<String>,
) -> anyhow::Result<Vec<Fixture>> {
    let mut out = Vec::new();
    let mut tries = 0u64;
    while out.len() < n {
        tries += 1;
        anyhow::ensure!(tries < 3_000_000, "could not find {n} {kind} positions");
        let mut pieces: Vec<char> = white.to_vec();
        pieces.push('k');
        let placed = place(rng, &pieces);
        if kings_adjacent(&placed) {
            continue;
        }
        let fen = fen_from(&placed);
        let Some(state) = try_state(&fen) else {
            continue;
        };
        let firsts = mate_in_two_moves(&state);
        if firsts.is_empty() || !seen.insert(fen.clone()) {
            continue;
        }
        out.push(Fixture {
            id: format!("{kind}-{:03}", out.len()),
            kind: kind.into(),
            fen,
            history: "fen_only_fresh_clocks".into(),
            correct: firsts,
            material_lead: material_diff(&state.to_fen()),
            note: "every first move that forces mate in two (no mate in one exists)".into(),
        });
    }
    Ok(out)
}

const MATE2_KINDS: [(&str, &[char]); 5] = [
    ("mate2_kqk", &['K', 'Q']),
    ("mate2_krk", &['K', 'R']),
    ("mate2_kqqk", &['K', 'Q', 'Q']),
    ("mate2_kqrk", &['K', 'Q', 'R']),
    ("mate2_krrk", &['K', 'R', 'R']),
];

/// `per_kind` exact mate-in-2 positions of each material set as `(kind, fen)`,
/// skipping `exclude` and duplicates.
pub(crate) fn synth_mate2_fens(
    seed: u64,
    per_kind: usize,
    exclude: &std::collections::HashSet<String>,
) -> anyhow::Result<Vec<(String, String)>> {
    let mut rng = Rng(seed);
    let mut seen = exclude.clone();
    let mut out = Vec::new();
    for (kind, white) in MATE2_KINDS {
        for f in gen_mate2(&mut rng, kind, white, per_kind, &mut seen)? {
            out.push((f.kind, f.fen));
        }
    }
    Ok(out)
}

#[derive(Args, Debug)]
pub struct GenMate2Args {
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long, default_value_t = 20261010)]
    pub seed: u64,
    #[arg(long, default_value_t = 12)]
    pub per_kind: usize,
    #[arg(long)]
    pub disjoint_fixtures: Vec<PathBuf>,
    #[arg(long)]
    pub disjoint_targets: Vec<PathBuf>,
}

pub fn run_gen_mate2(args: GenMate2Args) -> anyhow::Result<()> {
    let mut forbidden: std::collections::HashSet<String> = std::collections::HashSet::new();
    for p in &args.disjoint_fixtures {
        forbidden.extend(fixture_fens(p)?);
    }
    for p in &args.disjoint_targets {
        let t = recur64_runtime::reasoning_targets::ReasoningTargetsV1::load(p)?;
        forbidden.extend(t.positions.iter().map(|q| q.start_fen.clone()));
        forbidden.extend(t.positions.iter().map(|q| q.fen.clone()));
    }
    let fens = synth_mate2_fens(args.seed, args.per_kind, &forbidden)?;
    let mut fixtures = Vec::new();
    for (i, (kind, fen)) in fens.into_iter().enumerate() {
        let state = GameState::from_fen(&fen).map_err(|e| anyhow::anyhow!("{e}"))?;
        fixtures.push(Fixture {
            id: format!("{kind}-{i:03}"),
            kind,
            history: "fen_only_fresh_clocks".into(),
            correct: mate_in_two_moves(&state),
            material_lead: material_diff(&state.to_fen()),
            note: "every first move that forces mate in two (no mate in one exists)".into(),
            fen,
        });
    }
    // Cross-check a sample against the full GameState rules path (an independent
    // code path: moves are applied through GameState and its own termination).
    let mut checked = 0usize;
    for f in fixtures.iter().step_by(7) {
        let state = GameState::from_fen(&f.fen).map_err(|e| anyhow::anyhow!("{e}"))?;
        for (i, id) in state.legal_actions().iter().enumerate() {
            let s1 = apply_id(&state, *id);
            let forced = !s1.is_terminal()
                && !s1.legal_actions().is_empty()
                && s1.legal_actions().iter().all(|r| {
                    let s2 = apply_id(&s1, *r);
                    !s2.is_terminal() && !mating_moves(&s2).is_empty()
                });
            anyhow::ensure!(
                forced == f.correct.contains(&i),
                "{}: GameState cross-check disagrees on move {i}",
                f.id
            );
        }
        checked += 1;
    }
    let file = FixtureFile {
        schema: "x15_tactics_v1".into(),
        seed: args.seed,
        fixtures,
    };
    if let Some(dir) = args.out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&args.out, serde_json::to_vec_pretty(&file)?)?;
    let mut counts = std::collections::BTreeMap::new();
    for f in &file.fixtures {
        *counts.entry(f.kind.clone()).or_insert(0usize) += 1;
    }
    println!(
        "wrote {} ({} mate-in-2 fixtures): {counts:?}; {checked} cross-checked against GameState rules; {} forbidden FENs respected",
        args.out.display(),
        file.fixtures.len(),
        forbidden.len()
    );
    Ok(())
}

// --- exact-label targets (rules search as the teacher) --------------------------------

#[derive(Args, Debug)]
pub struct GenExactArgs {
    #[arg(long)]
    pub out: PathBuf,
    /// Positions per material set (5 sets).
    #[arg(long, default_value_t = 200)]
    pub per_kind: usize,
    #[arg(long, default_value_t = 20261011)]
    pub seed: u64,
    /// Split label given to every position.
    #[arg(long, default_value = "train")]
    pub split_label: String,
    #[arg(long)]
    pub disjoint_fixtures: Vec<PathBuf>,
    #[arg(long)]
    pub disjoint_targets: Vec<PathBuf>,
}

/// Build `ReasoningTargetsV1` whose teacher is exhaustive bounded RULES search:
/// the target policy is uniform over every first move that forces mate in two,
/// the root value is +1 (a forced win for the side to move). No network, no
/// PUCT, no external data. One rung (`simulations = 0` marks "exact").
pub fn run_gen_exact(args: GenExactArgs) -> anyhow::Result<()> {
    use recur64_runtime::reasoning_targets::{
        PositionTarget, Provenance, ReasoningTargetsV1, RungTarget, TeacherContract, audit,
        observation_digest,
    };
    let mut forbidden: std::collections::HashSet<String> = std::collections::HashSet::new();
    for p in &args.disjoint_fixtures {
        forbidden.extend(fixture_fens(p)?);
    }
    for p in &args.disjoint_targets {
        let t = ReasoningTargetsV1::load(p)?;
        forbidden.extend(t.positions.iter().map(|q| q.start_fen.clone()));
        forbidden.extend(t.positions.iter().map(|q| q.fen.clone()));
    }
    let fens = synth_mate2_fens(args.seed, args.per_kind, &forbidden)?;
    anyhow::ensure!(
        fens.iter().all(|(_, f)| !forbidden.contains(f)),
        "exact positions overlap the forbidden FEN set"
    );
    let mut positions = Vec::new();
    for (i, (kind, fen)) in fens.into_iter().enumerate() {
        let state = GameState::from_fen(&fen).map_err(|e| anyhow::anyhow!("{e}"))?;
        let legal_ids = state.legal_actions();
        let correct = mate_in_two_moves(&state);
        anyhow::ensure!(!correct.is_empty(), "{fen}: no forcing move");
        let mut policy = vec![0.0f32; legal_ids.len()];
        for c in &correct {
            policy[*c] = 1.0 / correct.len() as f32;
        }
        let entropy = (correct.len() as f32).ln();
        positions.push(PositionTarget {
            id: format!("exact-{kind}-{i:05}"),
            category: kind.clone(),
            split: args.split_label.clone(),
            source_game_id: 8_000_000 + i as u64,
            ply: 0,
            start_fen: fen.clone(),
            prefix: Vec::new(),
            fen: state.to_fen(),
            observation_sha256: observation_digest(&state),
            legal: legal_ids.iter().map(|a| a.index()).collect(),
            rungs: vec![RungTarget {
                simulations: 0,
                policy,
                root_value: 1.0,
                root_network_value: 0.0,
                total_visits: 0,
                best: correct[0],
                entropy,
            }],
        });
    }
    let teacher = TeacherContract {
        checkpoint: "exact_rules_search_v1".into(),
        model_id: "none".into(),
        architecture: "exact_rules_search_v1: uniform over every first move forcing mate in two"
            .into(),
        recurrence: 0,
        c_puct: 0.0,
        leaves_in_flight: 0,
        root_noise: false,
        ladder: vec![0],
        evaluator: "exhaustive bounded rules search; fresh-clock no-history convention".into(),
    };
    let provenance = Provenance {
        git_rev: recur64_runtime::provenance::git_revision()
            .unwrap_or("unknown")
            .into(),
        created_unix_s: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    };
    let targets = ReasoningTargetsV1::new(teacher, args.seed, provenance, positions);
    let n = audit(&targets)?;
    targets.save(&args.out)?;
    println!(
        "wrote {} ({n} exact mate-in-2 positions audited); digest {}; {} forbidden FENs respected",
        args.out.display(),
        targets.digest,
        forbidden.len()
    );
    Ok(())
}

// --- conversion rollouts ----------------------------------------------------------------

#[derive(Args, Debug)]
pub struct RolloutArgs {
    #[arg(long, default_value = "configs/x15_cuda.toml")]
    pub config: PathBuf,
    /// Fixtures whose positions are played out (white to move, a forced win exists).
    #[arg(long)]
    pub fixtures: PathBuf,
    #[arg(long)]
    pub checkpoint: Vec<PathBuf>,
    /// Thoughts used by the network when it moves.
    #[arg(long, default_value_t = 1)]
    pub thoughts: usize,
    /// Plies to play before calling the game unconverted.
    #[arg(long, default_value_t = 40)]
    pub max_plies: usize,
    /// Seed for the opponent's random replies.
    #[arg(long, default_value_t = 20261020)]
    pub seed: u64,
    /// Only fixtures whose kind starts with this prefix (empty = all).
    #[arg(long, default_value = "")]
    pub kind_prefix: String,
    #[arg(long, default_value_t = 64)]
    pub width: usize,
}

/// Outcome of one rollout.
#[derive(Clone, Debug)]
struct Outcome {
    kind: String,
    /// Plies until checkmate delivered by the network's side, if it happened.
    mated_at: Option<usize>,
    /// How the game ended when it did not end in checkmate.
    ended: String,
}

fn rollout_one<B: Backend>(
    cfg: &ProbeConfig,
    fixtures: &[Fixture],
    ck: &std::path::Path,
    args: &RolloutArgs,
    device: &B::Device,
) -> anyhow::Result<Vec<Outcome>> {
    let meta: CheckpointMeta = serde_json::from_slice(&std::fs::read(ck.join("meta.json"))?)?;
    let provider = provider_for_config(&meta.experimental)?;
    let model = model_io::load_chimera::<B>(ck, &cfg.model, &meta.experimental, device)?;
    let t = if meta.experimental.reasoning.enabled {
        args.thoughts
    } else {
        1
    };
    let mut states: Vec<GameState> = Vec::new();
    let mut kinds: Vec<String> = Vec::new();
    for f in fixtures
        .iter()
        .filter(|f| f.kind.starts_with(&args.kind_prefix))
    {
        states.push(GameState::from_fen(&f.fen).map_err(|e| anyhow::anyhow!("{}: {e}", f.id))?);
        kinds.push(f.kind.clone());
    }
    let n = states.len();
    let mut done: Vec<Option<Outcome>> = vec![None; n];
    for ply in 0..args.max_plies {
        // Everyone starts with the network's side to move, so all live games
        // share the parity: even plies are the network's, odd plies are random.
        let live: Vec<usize> = (0..n).filter(|i| done[*i].is_none()).collect();
        if live.is_empty() {
            break;
        }
        if ply % 2 == 0 {
            let batch_states: Vec<GameState> = live.iter().map(|i| states[*i].clone()).collect();
            let batch = build_x15_batch_padded::<B>(
                &batch_states,
                &meta.experimental,
                provider.as_ref(),
                device,
                args.width,
            )?;
            let out = model.forward_thoughts(&batch.input, &batch.cands, t);
            let width = batch.cands.width;
            let lp = out
                .readouts
                .last()
                .expect("a readout")
                .policy
                .log_probs
                .clone()
                .into_data()
                .to_vec::<f32>()
                .unwrap_or_default();
            for (row, &i) in live.iter().enumerate() {
                let legal = states[i].legal_actions();
                let r = &lp[row * width..row * width + legal.len()];
                let arg = r
                    .iter()
                    .enumerate()
                    .fold(0usize, |b, (j, v)| if *v > r[b] { j } else { b });
                states[i] = apply_id(&states[i], legal[arg]);
            }
        } else {
            for &i in &live {
                let legal = states[i].legal_actions();
                let pick = (mix(args.seed ^ mix(i as u64) ^ mix(ply as u64)) % legal.len() as u64)
                    as usize;
                states[i] = apply_id(&states[i], legal[pick]);
            }
        }
        for &i in &live {
            if states[i].is_terminal() {
                let mated = states[i].termination() == Some(Termination::Checkmate) && ply % 2 == 0;
                done[i] = Some(Outcome {
                    kind: kinds[i].clone(),
                    mated_at: mated.then_some(ply + 1),
                    ended: format!("{:?}", states[i].termination()),
                });
            }
        }
    }
    Ok((0..n)
        .map(|i| {
            done[i].clone().unwrap_or(Outcome {
                kind: kinds[i].clone(),
                mated_at: None,
                ended: "not_converted_in_time".into(),
            })
        })
        .collect())
}

pub fn run_rollout(args: RolloutArgs) -> anyhow::Result<()> {
    let cfg = ProbeConfig::from_toml_str(&std::fs::read_to_string(&args.config)?)?;
    let file: FixtureFile = serde_json::from_slice(&std::fs::read(&args.fixtures)?)?;
    #[cfg(feature = "cuda")]
    let device: Device<burn::backend::Cuda> = Default::default();
    #[cfg(not(feature = "cuda"))]
    let device: Device<burn::backend::Flex> = Default::default();
    println!(
        "conversion rollouts: network moves (argmax, T={}), opponent replies uniformly at random (seed {}), up to {} plies; a game counts as converted only if the network's side delivers checkmate",
        args.thoughts, args.seed, args.max_plies
    );
    for ck in &args.checkpoint {
        #[cfg(feature = "cuda")]
        let outcomes =
            rollout_one::<burn::backend::Cuda>(&cfg, &file.fixtures, ck, &args, &device)?;
        #[cfg(not(feature = "cuda"))]
        let outcomes =
            rollout_one::<burn::backend::Flex>(&cfg, &file.fixtures, ck, &args, &device)?;
        let mut kinds: Vec<String> = outcomes.iter().map(|o| o.kind.clone()).collect();
        kinds.sort();
        kinds.dedup();
        println!("checkpoint {}", ck.display());
        println!(
            "  {:<14} {:>3} {:>10} {:>12}  ended (not converted)",
            "kind", "n", "converted", "mean plies"
        );
        let mut total = 0usize;
        let mut conv = 0usize;
        for k in kinds {
            let os: Vec<&Outcome> = outcomes.iter().filter(|o| o.kind == k).collect();
            let c: Vec<usize> = os.iter().filter_map(|o| o.mated_at).collect();
            let mut ends: std::collections::BTreeMap<String, usize> =
                std::collections::BTreeMap::new();
            for o in os.iter().filter(|o| o.mated_at.is_none()) {
                *ends.entry(o.ended.clone()).or_insert(0) += 1;
            }
            total += os.len();
            conv += c.len();
            println!(
                "  {:<14} {:>3} {:>10.2} {:>12.1}  {:?}",
                k,
                os.len(),
                c.len() as f32 / os.len() as f32,
                if c.is_empty() {
                    f32::NAN
                } else {
                    c.iter().sum::<usize>() as f32 / c.len() as f32
                },
                ends
            );
        }
        println!(
            "  ALL: converted {conv}/{total} = {:.3}",
            conv as f32 / total.max(1) as f32
        );
    }
    Ok(())
}

// --- exact mate-in-3 (same fixture convention as mate-in-2) ------------------------------

/// The side to move can force checkmate within two of its own moves.
fn can_force_mate_within_two(b: &CBoard) -> bool {
    has_mate_in_one(b)
        || cboard_moves(b).into_iter().any(|m| {
            let mut n = b.clone();
            n.play(m);
            forced_mate_after(&n)
        })
}

/// `after` (opponent to move): the original mover forces mate within two more
/// of its own moves whatever the opponent replies.
fn forced_mate_within_two_after(after: &CBoard) -> bool {
    if after.status() != GameStatus::Ongoing {
        return false;
    }
    let replies = cboard_moves(after);
    if replies.is_empty() {
        return false;
    }
    replies.into_iter().all(|r| {
        let mut n = after.clone();
        n.play(r);
        n.status() == GameStatus::Ongoing && can_force_mate_within_two(&n)
    })
}

/// Legal-action indices of every first move that forces mate in three, for a
/// position where the mover has neither a mate in one nor a forced mate in two
/// (so three moves are genuinely needed).
pub(crate) fn mate_in_three_moves(state: &GameState) -> Vec<usize> {
    if can_force_mate_within_two(state.board()) {
        return Vec::new();
    }
    state
        .legal_actions()
        .iter()
        .enumerate()
        .filter(|(_, id)| forced_mate_within_two_after(apply_id(state, **id).board()))
        .map(|(i, _)| i)
        .collect()
}

/// Generate `n` exact mate-in-3 positions of one material set using `threads`
/// workers with independent seeded streams. Duplicates and `seen` FENs are
/// dropped; the result is sorted by FEN so it does not depend on thread timing.
fn gen_mate3_parallel(
    max_frac: f32,
    seed: u64,
    kind: &str,
    white: &[char],
    n: usize,
    threads: usize,
    seen: &std::collections::HashSet<String>,
) -> anyhow::Result<Vec<Fixture>> {
    let threads = threads.clamp(1, 32);
    let per = n.div_ceil(threads) + 2;
    let results: Vec<anyhow::Result<Vec<Fixture>>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                scope.spawn(move || {
                    let mut rng = Rng(mix(seed ^ mix(t as u64 + 1)));
                    let mut local_seen = std::collections::HashSet::new();
                    let mut out = Vec::new();
                    let mut tries = 0u64;
                    while out.len() < per {
                        tries += 1;
                        anyhow::ensure!(tries < 4_000_000, "could not find {per} {kind} positions");
                        let mut pieces: Vec<char> = white.to_vec();
                        pieces.push('k');
                        let placed = place(&mut rng, &pieces);
                        if kings_adjacent(&placed) {
                            continue;
                        }
                        let fen = fen_from(&placed);
                        let Some(state) = try_state(&fen) else { continue };
                        let firsts = mate_in_three_moves(&state);
                        // Keep only positions where few moves work, so chance is low.
                        let legal_n = state.legal_actions().len().max(1);
                        if firsts.is_empty()
                            || firsts.len() as f32 > max_frac * legal_n as f32
                            || !local_seen.insert(fen.clone())
                        {
                            continue;
                        }
                        out.push(Fixture {
                            id: String::new(),
                            kind: kind.into(),
                            fen,
                            history: "fen_only_fresh_clocks".into(),
                            correct: firsts,
                            material_lead: material_diff(&state.to_fen()),
                            note: "every first move that forces mate in three (no forced mate in two exists)".into(),
                        });
                    }
                    Ok(out)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err(anyhow::anyhow!("worker panicked")))
            })
            .collect()
    });
    let mut all: Vec<Fixture> = Vec::new();
    for r in results {
        all.extend(r?);
    }
    all.retain(|f| !seen.contains(&f.fen));
    all.sort_by(|a, b| a.fen.cmp(&b.fen));
    all.dedup_by(|a, b| a.fen == b.fen);
    anyhow::ensure!(
        all.len() >= n,
        "only {} unique {kind} positions for {n} requested",
        all.len()
    );
    // A seeded, thread-independent choice of which n to keep.
    all.sort_by_key(|f| {
        mix(seed
            ^ f.fen
                .bytes()
                .fold(0u64, |h, b| h.wrapping_mul(131).wrapping_add(u64::from(b))))
    });
    all.truncate(n);
    for (i, f) in all.iter_mut().enumerate() {
        f.id = format!("{kind}-{i:04}");
    }
    Ok(all)
}

const MATE3_KINDS: [(&str, &[char]); 5] = [
    ("mate3_kqk", &['K', 'Q']),
    ("mate3_krk", &['K', 'R']),
    ("mate3_kqqk", &['K', 'Q', 'Q']),
    ("mate3_kqrk", &['K', 'Q', 'R']),
    ("mate3_krrk", &['K', 'R', 'R']),
];

#[derive(Args, Debug)]
pub struct GenMate3Args {
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long, default_value_t = 20261030)]
    pub seed: u64,
    #[arg(long, default_value_t = 10)]
    pub per_kind: usize,
    #[arg(long, default_value_t = 6)]
    pub threads: usize,
    /// Keep only positions where at most this fraction of the legal moves is correct.
    #[arg(long, default_value_t = 0.15)]
    pub max_correct_fraction: f32,
    #[arg(long)]
    pub disjoint_fixtures: Vec<PathBuf>,
    #[arg(long)]
    pub disjoint_targets: Vec<PathBuf>,
}

pub fn run_gen_mate3(args: GenMate3Args) -> anyhow::Result<()> {
    let mut forbidden: std::collections::HashSet<String> = std::collections::HashSet::new();
    for p in &args.disjoint_fixtures {
        forbidden.extend(fixture_fens(p)?);
    }
    for p in &args.disjoint_targets {
        let t = recur64_runtime::reasoning_targets::ReasoningTargetsV1::load(p)?;
        forbidden.extend(t.positions.iter().map(|q| q.start_fen.clone()));
        forbidden.extend(t.positions.iter().map(|q| q.fen.clone()));
    }
    let mut fixtures = Vec::new();
    for (kind, white) in MATE3_KINDS {
        let mut f = gen_mate3_parallel(
            args.max_correct_fraction,
            args.seed ^ mix(kind.len() as u64 + white.len() as u64 * 977),
            kind,
            white,
            args.per_kind,
            args.threads,
            &forbidden,
        )?;
        fixtures.append(&mut f);
    }
    // Independent cross-check of a sample through the full GameState rules path.
    let mut checked = 0usize;
    for f in fixtures.iter().step_by(11) {
        let state = GameState::from_fen(&f.fen).map_err(|e| anyhow::anyhow!("{e}"))?;
        let within2 = |s: &GameState| -> bool {
            // mate in one, or a first move after which every reply allows a mate in one
            !mating_moves(s).is_empty()
                || s.legal_actions().iter().any(|id| {
                    let s1 = apply_id(s, *id);
                    !s1.is_terminal()
                        && !s1.legal_actions().is_empty()
                        && s1.legal_actions().iter().all(|r| {
                            let s2 = apply_id(&s1, *r);
                            !s2.is_terminal() && !mating_moves(&s2).is_empty()
                        })
                })
        };
        anyhow::ensure!(!within2(&state), "{}: has a forced mate within two", f.id);
        for (i, id) in state.legal_actions().iter().enumerate() {
            let s1 = apply_id(&state, *id);
            let forced = !s1.is_terminal()
                && !s1.legal_actions().is_empty()
                && s1.legal_actions().iter().all(|r| {
                    let s2 = apply_id(&s1, *r);
                    !s2.is_terminal() && within2(&s2)
                });
            anyhow::ensure!(
                forced == f.correct.contains(&i),
                "{}: GameState cross-check disagrees on move {i}",
                f.id
            );
        }
        checked += 1;
    }
    let file = FixtureFile {
        schema: "x15_tactics_v1".into(),
        seed: args.seed,
        fixtures,
    };
    if let Some(dir) = args.out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&args.out, serde_json::to_vec_pretty(&file)?)?;
    let mut counts = std::collections::BTreeMap::new();
    for f in &file.fixtures {
        *counts.entry(f.kind.clone()).or_insert(0usize) += 1;
    }
    // Chance level of the top-1 metric: the mean share of legal moves that are correct.
    let chance: f32 = file
        .fixtures
        .iter()
        .map(|f| {
            let n = GameState::from_fen(&f.fen)
                .map(|s| s.legal_actions().len())
                .unwrap_or(1);
            f.correct.len() as f32 / n.max(1) as f32
        })
        .sum::<f32>()
        / file.fixtures.len().max(1) as f32;
    println!(
        "wrote {} ({} mate-in-3 fixtures): {counts:?}; {checked} cross-checked against GameState rules; {} forbidden FENs respected; chance level (uniform random move) {chance:.3}",
        args.out.display(),
        file.fixtures.len(),
        forbidden.len()
    );
    Ok(())
}
