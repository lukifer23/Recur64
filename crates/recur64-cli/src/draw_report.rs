//! `recur64 draw-report` — why do self-play games end in draws?
//!
//! Reconstructs every replay game and, per cycle, classifies each draw by
//! how it arose: a *failed conversion* (one side held a decisive material
//! advantage, a rook or more, at some point in the final 100 plies and still
//! drew) versus a *balanced* draw. Games from any other start (D51 endgame
//! curriculum, configured start FENs) are reported separately, so the
//! standard-start series stays comparable. Diagnostic only: CPU, no network.

use std::collections::BTreeMap;
use std::path::PathBuf;

use clap::Args;

use recur64_core::{ActionId, GameState, StandardMove, material, material_balance};
use recur64_runtime::replay::{ReplayReader, parse_shard_bytes};

#[derive(Args, Debug)]
pub struct DrawReportArgs {
    #[arg(long)]
    pub replay: PathBuf,
    /// Games per cycle (game ids are contiguous per cycle).
    #[arg(long, default_value_t = 64)]
    pub games_per_cycle: u64,
    #[arg(long)]
    pub output: PathBuf,
    /// Material advantage (pawn units) that counts as decisive.
    #[arg(long, default_value_t = 5)]
    pub decisive_advantage: i32,
    /// Final plies inspected for a decisive advantage.
    #[arg(long, default_value_t = 100)]
    pub window: usize,
}

#[derive(Debug, Default, Clone, serde::Serialize)]
struct CycleDraws {
    games: u64,
    draws: u64,
    draws_by_termination: BTreeMap<String, u64>,
    /// Draws where one side held >= `decisive_advantage` within the window.
    failed_conversions: u64,
    failed_conversions_by_termination: BTreeMap<String, u64>,
    mean_draw_plies: f64,
    /// Mean total material (both sides) at the end of drawn games.
    mean_final_material_draws: f64,
    /// Mean peak |material balance| within the window, over draws.
    mean_peak_advantage_draws: f64,
    decisive_games: u64,
    mean_decisive_plies: f64,
}

pub fn run(args: DrawReportArgs) -> anyhow::Result<()> {
    // Active shards plus shards moved to archive/ by capacity enforcement, so
    // the report covers every cycle, not only the ones still sampleable.
    let mut games = ReplayReader::open(&args.replay)?.read_all_games()?;
    let archive = args.replay.join("archive");
    if archive.exists() {
        for entry in std::fs::read_dir(&archive)? {
            let path = entry?.path();
            if path.extension().is_some_and(|e| e == "r64shard") {
                let shard = parse_shard_bytes(&std::fs::read(&path)?)
                    .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
                games.extend(shard.games);
            }
        }
    }
    games.sort_by_key(|g| g.game_id);
    games.dedup_by_key(|g| g.game_id);
    let standard_fen = GameState::startpos().to_fen();
    // Keyed by (standard start?, cycle).
    let mut cycles: BTreeMap<(bool, u64), CycleDraws> = BTreeMap::new();
    let mut sums: BTreeMap<(bool, u64), (f64, f64, f64, f64)> = BTreeMap::new();
    for game in &games {
        let cycle = (
            game.start_fen == standard_fen,
            game.game_id / args.games_per_cycle.max(1),
        );
        let mut state = GameState::from_fen(&game.start_fen)?;
        let mut balances = vec![material_balance(state.board())];
        for ply in &game.plies {
            let id = ActionId::from_index(ply.selected as u32)?;
            let (from, to, promo) = id.to_physical(state.perspective());
            let promotion = if promo.is_none() { None } else { Some(promo) };
            state.apply(StandardMove::new(from, to, promotion))?;
            balances.push(material_balance(state.board()));
        }
        let c = cycles.entry(cycle).or_default();
        let s = sums.entry(cycle).or_default();
        c.games += 1;
        let plies = game.plies.len() as f64;
        match game.outcome {
            Some(1) => {
                c.draws += 1;
                *c.draws_by_termination
                    .entry(game.termination.clone())
                    .or_default() += 1;
                let tail = &balances[balances.len().saturating_sub(args.window + 1)..];
                let peak = tail.iter().map(|b| b.abs()).max().unwrap_or(0);
                if peak >= args.decisive_advantage {
                    c.failed_conversions += 1;
                    *c.failed_conversions_by_termination
                        .entry(game.termination.clone())
                        .or_default() += 1;
                }
                let board = state.board();
                let total = material(board, recur64_core::Color::White)
                    + material(board, recur64_core::Color::Black);
                s.0 += plies;
                s.1 += total as f64;
                s.2 += peak as f64;
            }
            Some(_) => {
                c.decisive_games += 1;
                s.3 += plies;
            }
            None => {}
        }
    }
    for (key, c) in cycles.iter_mut() {
        let s = sums[key];
        let (standard, cycle) = *key;
        let d = c.draws.max(1) as f64;
        c.mean_draw_plies = s.0 / d;
        c.mean_final_material_draws = s.1 / d;
        c.mean_peak_advantage_draws = s.2 / d;
        c.mean_decisive_plies = s.3 / c.decisive_games.max(1) as f64;
        println!(
            "{} cycle {cycle}: games {} draws {} (failed conversions {} = {:.0}%) by term {:?} | draw plies {:.0} final material {:.1} peak adv {:.1} | decisive {} plies {:.0}",
            if standard { "standard" } else { "other-start" },
            c.games,
            c.draws,
            c.failed_conversions,
            100.0 * c.failed_conversions as f64 / d,
            c.draws_by_termination,
            c.mean_draw_plies,
            c.mean_final_material_draws,
            c.mean_peak_advantage_draws,
            c.decisive_games,
            c.mean_decisive_plies
        );
    }
    let split = |standard: bool| -> BTreeMap<u64, CycleDraws> {
        cycles
            .iter()
            .filter(|((s, _), _)| *s == standard)
            .map(|((_, cycle), c)| (*cycle, c.clone()))
            .collect()
    };
    std::fs::create_dir_all(&args.output)?;
    let report = serde_json::json!({
        "replay": args.replay,
        "games_per_cycle": args.games_per_cycle,
        "decisive_advantage": args.decisive_advantage,
        "window": args.window,
        "cycles": split(true),
        "other_start_cycles": split(false),
    });
    std::fs::write(
        args.output.join("draw-report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(())
}
