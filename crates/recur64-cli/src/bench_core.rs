//! `recur64 bench-core` — Phase 1 CPU baselines for the chess layer.
//!
//! Correctness first: this harness only measures, it does not optimize. Run with
//! `--release` for meaningful numbers. Writes JSON and markdown to `--output`.

use std::path::PathBuf;
use std::time::Instant;

use clap::Args;

use recur64_core::perft::perft;
use recur64_core::{Board, GameState, StandardMove, encode_observation_v1};

#[derive(Args, Debug)]
pub struct BenchCoreArgs {
    #[arg(long, default_value = "runs/bench-core")]
    pub output: PathBuf,
    #[arg(long, default_value_t = 20_000)]
    pub iters: usize,
    #[arg(long, default_value_t = 5_000)]
    pub apply_iters: usize,
    #[arg(long, default_value_t = 3)]
    pub perft_depth: u32,
}

const POSITIONS: &[(&str, &str)] = &[
    (
        "startpos",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    ),
    ("endgame_kp", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ("promotion", "8/P6k/8/8/8/8/8/K7 w - - 0 1"),
    ("castling", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
];

#[derive(serde::Serialize)]
struct PosResult {
    name: &'static str,
    legal_moves: usize,
    movegen_actions_per_sec: f64,
    apply_clone_per_sec: f64,
    encode_per_sec: f64,
    perft_nodes_per_sec: f64,
}

/// Search-relevant per-node costs as a function of game length. Search
/// clones the full `GameState` (including its whole history) for every
/// expanded node, so this cost may grow with the ply of the root position.
#[derive(serde::Serialize)]
struct PlyResult {
    ply: u32,
    legal_moves: usize,
    history_len: usize,
    clone_apply_per_sec: f64,
    clone_only_per_sec: f64,
    legal_actions_per_sec: f64,
    encode_per_sec: f64,
}

#[derive(serde::Serialize)]
struct CoreReport {
    recur64_version: String,
    profile: &'static str,
    notes: Vec<String>,
    positions: Vec<PosResult>,
    game_length_sweep: Vec<PlyResult>,
    game_state_size_bytes: usize,
    board_size_bytes: usize,
    standard_move_size_bytes: usize,
}

fn time<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed().as_secs_f64()
}

pub fn run_bench_core(args: BenchCoreArgs) -> anyhow::Result<()> {
    let mut notes = vec![
        "Release mode required for meaningful numbers.".to_string(),
        "apply_clone = clone + apply one legal move (includes history clone).".to_string(),
        "Not chess learning; CPU rules/encoding throughput only.".to_string(),
    ];
    if cfg!(debug_assertions) {
        notes.push("WARNING: built without --release; numbers are not representative.".to_string());
    }

    let mut results = Vec::new();
    for (name, fen) in POSITIONS {
        let g = GameState::from_fen(fen).unwrap();
        let moves = g.legal_standard_moves();
        let sample = moves.first().copied().unwrap_or(StandardMove::new(
            recur64_core::Square::A1,
            recur64_core::Square::A2,
            None,
        ));

        let movegen_s = time(args.iters, || {
            let _ = g.legal_action_list();
        });
        let apply_s = time(args.apply_iters, || {
            let mut s = g.clone();
            let _ = s.apply(sample);
        });
        let encode_s = time(args.iters, || {
            let _ = encode_observation_v1(&g);
        });

        let board: Board = fen.parse().unwrap();
        let perft_nodes = perft(&board, args.perft_depth) as f64;
        let perft_s = time(1, || {
            let _ = perft(&board, args.perft_depth);
        });

        let r = PosResult {
            name,
            legal_moves: moves.len(),
            movegen_actions_per_sec: args.iters as f64 / movegen_s,
            apply_clone_per_sec: args.apply_iters as f64 / apply_s,
            encode_per_sec: args.iters as f64 / encode_s,
            perft_nodes_per_sec: perft_nodes / perft_s,
        };
        println!(
            "{:<12} moves={:<3} movegen={:>10.0}/s apply={:>10.0}/s encode={:>10.0}/s perft={:>12.0} nps",
            r.name,
            r.legal_moves,
            r.movegen_actions_per_sec,
            r.apply_clone_per_sec,
            r.encode_per_sec,
            r.perft_nodes_per_sec
        );
        results.push(r);
    }

    // Game-length sweep: a deterministic pseudo-random legal game, measured
    // at increasing plies (stopping early if the game ends).
    let mut sweep = Vec::new();
    let mut g = GameState::startpos();
    let mut lcg: u64 = 0x9E37_79B9_7F4A_7C15;
    for target in [0u32, 50, 100, 200, 300, 400] {
        while g.ply() < target && g.termination().is_none() {
            let moves = g.legal_standard_moves();
            if moves.is_empty() {
                break;
            }
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let mv = moves[(lcg >> 33) as usize % moves.len()];
            g.apply(mv)?;
        }
        if g.termination().is_some() || g.ply() < target {
            println!("game-length sweep stopped at ply {} (game ended)", g.ply());
            break;
        }
        let moves = g.legal_standard_moves();
        let sample = moves[0];
        let n = args.apply_iters;
        let clone_apply_s = time(n, || {
            let mut s = g.clone();
            let _ = s.apply(sample);
        });
        let clone_s = time(n, || {
            let _ = std::hint::black_box(g.clone());
        });
        let legal_s = time(n, || {
            let _ = std::hint::black_box(g.legal_actions());
        });
        let encode_s = time(n, || {
            let _ = std::hint::black_box(encode_observation_v1(&g));
        });
        let r = PlyResult {
            ply: g.ply(),
            legal_moves: moves.len(),
            history_len: g.history().len(),
            clone_apply_per_sec: n as f64 / clone_apply_s,
            clone_only_per_sec: n as f64 / clone_s,
            legal_actions_per_sec: n as f64 / legal_s,
            encode_per_sec: n as f64 / encode_s,
        };
        println!(
            "ply {:<4} legal={:<3} history={:<4} clone+apply={:>9.0}/s clone={:>9.0}/s legal_actions={:>9.0}/s encode={:>9.0}/s",
            r.ply,
            r.legal_moves,
            r.history_len,
            r.clone_apply_per_sec,
            r.clone_only_per_sec,
            r.legal_actions_per_sec,
            r.encode_per_sec
        );
        sweep.push(r);
    }

    let report = CoreReport {
        recur64_version: recur64_model::VERSION.to_string(),
        profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        notes,
        positions: results,
        game_length_sweep: sweep,
        game_state_size_bytes: std::mem::size_of::<GameState>(),
        board_size_bytes: std::mem::size_of::<Board>(),
        standard_move_size_bytes: std::mem::size_of::<StandardMove>(),
    };

    std::fs::create_dir_all(&args.output)?;
    std::fs::write(
        args.output.join("bench-core.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;

    let mut md = String::from("# Recur64 core bench (Phase 1)\n\n");
    md.push_str(&format!(
        "- recur64 {} | profile {}\n",
        report.recur64_version, report.profile
    ));
    md.push_str(&format!(
        "- GameState {} B | Board {} B | StandardMove {} B\n\n",
        report.game_state_size_bytes, report.board_size_bytes, report.standard_move_size_bytes
    ));
    for n in &report.notes {
        md.push_str(&format!("- {n}\n"));
    }
    md.push_str("\n| position | legal | movegen/s | apply(clone)/s | encode/s | perft nps |\n");
    md.push_str("|---|---:|---:|---:|---:|---:|\n");
    for r in &report.positions {
        md.push_str(&format!(
            "| {} | {} | {:.0} | {:.0} | {:.0} | {:.0} |\n",
            r.name,
            r.legal_moves,
            r.movegen_actions_per_sec,
            r.apply_clone_per_sec,
            r.encode_per_sec,
            r.perft_nodes_per_sec
        ));
    }
    std::fs::write(args.output.join("bench-core.md"), md)?;
    println!(
        "\nwrote {}/bench-core.json and bench-core.md",
        args.output.display()
    );
    Ok(())
}
