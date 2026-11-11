//! V500 P2: baseline solve curves on the frozen TEST split.
//!
//! cargo run --release -p recur64-prove --example baseline -- <KRvK|KQvK> <per_depth> <budget> <out.json> [threads]
//!
//! For each true optimal depth, positions are taken deterministically from the TEST split (sorted by canonical key),
//! and every config is run once at the largest budget; a position counts as solved at budget b iff its CHECKED proof
//! was found with at most b nodes. Reports solve rate, median nodes, proof-length overhead and wall time.

use std::time::Instant;

use cozy_chess::{Board, Piece};
use recur64_prove::classical::Classical;
use recur64_prove::search::{Blind, Prover, verify};
use recur64_prove::split::{Part, canon_key, part_of};
use recur64_prove::tb::Tablebase;
use serde_json::json;

#[derive(Clone, Copy)]
struct Cfg {
    name: &'static str,
    classical: bool,
    width: Option<usize>,
}

const CFGS: [Cfg; 6] = [
    Cfg { name: "blind_full", classical: false, width: None },
    Cfg { name: "classical_full", classical: true, width: None },
    Cfg { name: "classical_w4", classical: true, width: Some(4) },
    Cfg { name: "classical_w3", classical: true, width: Some(3) },
    Cfg { name: "classical_w2", classical: true, width: Some(2) },
    Cfg { name: "classical_w1", classical: true, width: Some(1) },
];

struct One {
    solved: bool,
    nodes: u64,
    bound: Option<u8>,
    secs: f64,
}

fn run_one(cfg: Cfg, b: &Board, max_n: u8, budget: u64) -> One {
    let t = Instant::now();
    let solved = if cfg.classical {
        let mut o = Classical;
        Prover::new(&mut o, cfg.width, budget).solve(b, max_n)
    } else {
        let mut o = Blind;
        Prover::new(&mut o, cfg.width, budget).solve(b, max_n)
    };
    let ok = match &solved.proof {
        Some(p) => verify(b, p).is_ok(),
        None => false,
    };
    One { solved: ok, nodes: solved.nodes, bound: solved.bound, secs: t.elapsed().as_secs_f64() }
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 5, "usage: baseline <KRvK|KQvK> <per_depth> <budget> <out.json> [threads]");
    let (name, piece) = match a[1].as_str() {
        "KRvK" => ("KRvK", Piece::Rook),
        "KQvK" => ("KQvK", Piece::Queen),
        "KBNvK" => ("KBNvK", Piece::Bishop),
        other => anyhow::bail!("unknown material {other}"),
    };
    let per_depth: usize = a[2].parse()?;
    let budget: u64 = a[3].parse()?;
    let threads: usize = a.get(5).map_or(Ok(8), |s| s.parse())?;
    let extra: Vec<Piece> = if name == "KBNvK" { vec![Piece::Bishop, Piece::Knight] } else { vec![piece] };
    let tb = Tablebase::build(name, &extra)?;
    let max_depth = tb.max_plies_white.div_ceil(2);
    let mut by_depth: Vec<Vec<Board>> = vec![Vec::new(); max_depth as usize + 1];
    let mut won = tb.won_white_positions();
    won.sort_by_key(|(b, _)| canon_key(b));
    for (b, d) in won {
        if part_of(&b) == Part::Test {
            by_depth[d as usize].push(b);
        }
    }
    let mut jobs: Vec<(u8, Board)> = Vec::new();
    for d in 1..=max_depth {
        let v = &by_depth[d as usize];
        let step = (v.len() / per_depth.max(1)).max(1);
        for b in v.iter().step_by(step).take(per_depth) {
            jobs.push((d, b.clone()));
        }
    }
    eprintln!("{name}: {} test positions over depths 1..={max_depth}", jobs.len());
    let max_n = max_depth + 2;
    let mut report = Vec::new();
    for cfg in CFGS {
        let results: Vec<(u8, One)> = std::thread::scope(|s| {
            let chunk = jobs.len().div_ceil(threads);
            let handles: Vec<_> = jobs
                .chunks(chunk)
                .map(|c| s.spawn(move || c.iter().map(|(d, b)| (*d, run_one(cfg, b, max_n, budget))).collect::<Vec<_>>()))
                .collect();
            handles.into_iter().flat_map(|h| h.join().expect("worker")).collect()
        });
        let mut rows = Vec::new();
        for d in 1..=max_depth {
            let r: Vec<&One> = results.iter().filter(|(x, _)| *x == d).map(|(_, o)| o).collect();
            if r.is_empty() {
                continue;
            }
            let at = |bud: u64| r.iter().filter(|o| o.solved && o.nodes <= bud).count() as f64 / r.len() as f64;
            let mut nodes: Vec<u64> = r.iter().filter(|o| o.solved).map(|o| o.nodes).collect();
            nodes.sort_unstable();
            let over: Vec<f64> = r.iter().filter(|o| o.solved).filter_map(|o| o.bound.map(|x| f64::from(x) - f64::from(d))).collect();
            rows.push(json!({
                "depth": d, "n": r.len(),
                "solve_1e3": at(1_000), "solve_1e4": at(10_000), "solve_1e5": at(100_000), "solve_1e6": at(1_000_000),
                "median_nodes_solved": nodes.get(nodes.len() / 2),
                "mean_overhead": (!over.is_empty()).then(|| over.iter().sum::<f64>() / over.len() as f64),
                "mean_secs": r.iter().map(|o| o.secs).sum::<f64>() / r.len() as f64,
            }));
        }
        let summary: Vec<String> = rows
            .iter()
            .map(|r| format!("d{}:{:.2}", r["depth"], r["solve_1e5"].as_f64().unwrap_or(0.0)))
            .collect();
        eprintln!("{:<15} solve@1e5 {}", cfg.name, summary.join(" "));
        report.push(json!({"config": cfg.name, "width": cfg.width, "rows": rows}));
    }
    std::fs::write(
        &a[4],
        serde_json::to_vec_pretty(&json!({
            "schema": "v500_baseline_v1", "material": name, "split": "TEST", "per_depth": per_depth,
            "budget": budget, "max_n": max_n, "configs": report,
        }))?,
    )?;
    Ok(())
}
