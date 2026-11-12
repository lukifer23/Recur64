//! V500 evaluation harness: run one ordering (learned net, classical or blind) at several widths on a split, and
//! write per-position records (so gates and paired comparisons are computed from the same positions).
//!
//! cargo run --release -p recur64-prove --example eval_prover -- <orderer> <KRvK|KQvK> <val|test> <per_depth> <budget> <widths> <out.json> [threads] [min_depth]
//!   orderer: a path to net.json (learned), `classical` or `blind`; widths: comma list, `full` = no width limit.
//!
//! Position selection is identical to `baseline.rs` (sorted by canonical key, evenly strided, per depth), so a TEST run
//! reproduces the baseline's position set exactly. Every counted solve carries a proof that passed the independent checker.

use std::time::Instant;

use cozy_chess::{Board, Piece};
use recur64_prove::classical::Classical;
use recur64_prove::net::{Learned, Mlp};
use recur64_prove::search::{Blind, Orderer, Prover, verify};
use recur64_prove::split::{Part, canon_key, part_of};
use recur64_prove::tb::Tablebase;
use serde_json::json;

struct Rec {
    depth: u8,
    solved: bool,
    nodes: u64,
    bound: Option<u8>,
    secs: f64,
}

fn run_one(o: &mut dyn Orderer, width: Option<usize>, b: &Board, max_n: u8, budget: u64) -> Rec {
    let t = Instant::now();
    let s = Prover::new(o, width, budget).solve(b, max_n);
    let ok = s.proof.as_ref().is_some_and(|p| verify(b, p).is_ok());
    // A proof that fails the independent checker would be a soundness bug: refuse to continue.
    assert!(s.proof.is_none() || ok, "UNSOUND PROOF returned for {b}");
    Rec { depth: 0, solved: ok, nodes: s.nodes, bound: s.bound, secs: t.elapsed().as_secs_f64() }
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 8, "usage: eval_prover <net.json|classical|blind> <KRvK|KQvK> <val|test> <per_depth> <budget> <widths> <out.json> [threads]");
    let (name, piece) = match a[2].as_str() {
        "KRvK" => ("KRvK", Piece::Rook),
        "KQvK" => ("KQvK", Piece::Queen),
        "KBNvK" => ("KBNvK", Piece::Bishop),
        other => anyhow::bail!("unknown material {other}"),
    };
    let part = match a[3].as_str() {
        "val" => Part::Val,
        "test" => Part::Test,
        other => anyhow::bail!("split {other}"),
    };
    let per_depth: usize = a[4].parse()?;
    let budget: u64 = a[5].parse()?;
    let widths: Vec<Option<usize>> = a[6]
        .split(',')
        .map(|w| if w == "full" { Ok(None) } else { w.parse().map(Some) })
        .collect::<Result<_, _>>()?;
    let threads: usize = a.get(8).map_or(Ok(8), |s| s.parse())?;
    let min_depth: u8 = a.get(9).map_or(Ok(1), |s| s.parse())?;
    // `net.json` or `net.json@0.95` (nucleus width: keep attacker moves up to that softmax mass).
    let (net_path, top_p): (String, Option<f32>) = match a[1].split_once('@') {
        Some((p, t)) => (p.to_string(), Some(t.parse()?)),
        None => (a[1].clone(), None),
    };
    let net = match net_path.as_str() {
        "classical" | "blind" => None,
        path => Some(Mlp::load(std::path::Path::new(path))?),
    };
    let extra: Vec<Piece> = if name == "KBNvK" { vec![Piece::Bishop, Piece::Knight] } else { vec![piece] };
    let tb = Tablebase::build(name, &extra)?;
    let max_depth = tb.max_plies_white.div_ceil(2);
    let mut by_depth: Vec<Vec<Board>> = vec![Vec::new(); max_depth as usize + 1];
    let mut won = tb.won_white_positions();
    won.sort_by_key(|(b, _)| canon_key(b));
    for (b, d) in won {
        if part_of(&b) == part {
            by_depth[d as usize].push(b);
        }
    }
    let mut jobs: Vec<(u8, Board)> = Vec::new();
    for d in min_depth..=max_depth {
        let v = &by_depth[d as usize];
        let step = (v.len() / per_depth.max(1)).max(1);
        for b in v.iter().step_by(step).take(per_depth) {
            jobs.push((d, b.clone()));
        }
    }
    let max_n = max_depth + 2;
    let kind = net_path.clone();
    let mut out = Vec::new();
    for width in widths {
        let t0 = Instant::now();
        let recs: Vec<Rec> = std::thread::scope(|s| {
            let chunk = jobs.len().div_ceil(threads);
            let handles: Vec<_> = jobs
                .chunks(chunk)
                .map(|c| {
                    let net = net.clone();
                    let kind = kind.clone();
                    s.spawn(move || {
                        c.iter()
                            .map(|(d, b)| {
                                let mut r = match (&net, kind.as_str()) {
                                    (Some(n), _) => run_one(&mut Learned { net: n.clone(), top_p }, width, b, max_n, budget),
                                    (None, "classical") => run_one(&mut Classical, width, b, max_n, budget),
                                    _ => run_one(&mut Blind, width, b, max_n, budget),
                                };
                                r.depth = *d;
                                r
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles.into_iter().flat_map(|h| h.join().expect("worker")).collect()
        });
        let line: Vec<String> = (min_depth..=max_depth)
            .map(|d| {
                let r: Vec<&Rec> = recs.iter().filter(|x| x.depth == d).collect();
                format!("d{d}:{:.2}", r.iter().filter(|x| x.solved && x.nodes <= 100_000).count() as f64 / r.len().max(1) as f64)
            })
            .collect();
        eprintln!("{kind} width {width:?} solve@1e5 {} ({:.0}s)", line.join(" "), t0.elapsed().as_secs_f64());
        out.push(json!({
            "width": width,
            "records": recs.iter().map(|r| json!({"depth": r.depth, "solved": r.solved, "nodes": r.nodes, "bound": r.bound, "secs": r.secs})).collect::<Vec<_>>(),
        }));
    }
    std::fs::write(
        &a[7],
        serde_json::to_vec(&json!({"schema": "v500_eval_v1", "orderer": a[1], "material": name, "split": a[3],
                                    "per_depth": per_depth, "budget": budget, "max_n": max_n, "runs": out}))?,
    )?;
    Ok(())
}
