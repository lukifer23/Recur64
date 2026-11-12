//! FEASIBILITY DIAGNOSTIC (not a result about the learned system): how many nodes does this search structure need
//! when the attacker ordering is PERFECT (taken from the tablebase)? It bounds what any ordering can achieve with
//! width-limited AND/OR search and tells us whether a node budget is reachable at all.
//!
//! cargo run --release -p recur64-prove --example oracle_nodes -- <KRvK|KBNvK> <min_depth> <per_depth> <budget> <width> <out.json> [threads]
use cozy_chess::{Board, Color, GameStatus, Move, Piece};
use recur64_prove::search::{Orderer, Prover, verify};
use recur64_prove::split::{Part, canon_key, part_of};
use recur64_prove::tb::Tablebase;
use serde_json::json;

struct Oracle<'a> {
    tb: &'a Tablebase,
}

impl Orderer for Oracle<'_> {
    fn attacker(&mut self, b: &Board, moves: &mut Vec<Move>) {
        let key = |m: &Move| {
            let mut a = b.clone();
            a.play_unchecked(*m);
            if a.status() == GameStatus::Won { 0u32 } else { self.tb.plies_to_mate(&a).map_or(10_000, u32::from) }
        };
        moves.sort_by_key(key);
    }
    fn defender(&mut self, b: &Board, moves: &mut Vec<Move>) {
        let key = |m: &Move| {
            let mut a = b.clone();
            a.play_unchecked(*m);
            std::cmp::Reverse(self.tb.plies_to_mate(&a).map_or(10_000u32, u32::from))
        };
        moves.sort_by_key(key);
    }
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 7, "usage: oracle_nodes <material> <min_depth> <per_depth> <budget> <width> <out.json> [threads]");
    let extra: Vec<Piece> = match a[1].as_str() {
        "KRvK" => vec![Piece::Rook],
        "KBNvK" => vec![Piece::Bishop, Piece::Knight],
        o => anyhow::bail!("material {o}"),
    };
    let (min_depth, per_depth): (u8, usize) = (a[2].parse()?, a[3].parse()?);
    let (budget, width): (u64, usize) = (a[4].parse()?, a[5].parse()?);
    let threads: usize = a.get(7).map_or(Ok(8), |s| s.parse())?;
    let tb = Tablebase::build(&a[1], &extra)?;
    let max_depth = tb.max_plies_white.div_ceil(2);
    let mut by: Vec<Vec<Board>> = vec![Vec::new(); max_depth as usize + 1];
    let mut won = tb.won_white_positions();
    won.sort_by_key(|(b, _)| canon_key(b));
    for (b, d) in won {
        if part_of(&b) == Part::Test && d >= min_depth {
            by[d as usize].push(b);
        }
    }
    let mut jobs: Vec<(u8, Board)> = Vec::new();
    for d in min_depth..=max_depth {
        let v = &by[d as usize];
        let step = (v.len() / per_depth.max(1)).max(1);
        jobs.extend(v.iter().step_by(step).take(per_depth).map(|b| (d, b.clone())));
    }
    let tbr = &tb;
    let recs: Vec<(u8, u64, bool)> = std::thread::scope(|s| {
        let chunk = jobs.len().div_ceil(threads);
        let hs: Vec<_> = jobs
            .chunks(chunk)
            .map(|c| {
                s.spawn(move || {
                    c.iter()
                        .map(|(d, b)| {
                            let mut o = Oracle { tb: tbr };
                            let r = Prover::new(&mut o, Some(width), budget).solve_from(b, *d, *d + 2);
                            let ok = r.proof.as_ref().is_some_and(|p| verify(b, p).is_ok());
                            (*d, r.nodes, ok)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().expect("worker")).collect()
    });
    let mut rows = Vec::new();
    for d in min_depth..=max_depth {
        let mut n: Vec<u64> = recs.iter().filter(|r| r.0 == d && r.2).map(|r| r.1).collect();
        n.sort_unstable();
        let tot = recs.iter().filter(|r| r.0 == d).count();
        if tot > 0 {
            rows.push(json!({"depth": d, "n": tot, "solved": n.len(), "median_nodes_solved": n.get(n.len() / 2), "max_nodes_solved": n.last()}));
            eprintln!("d{d}: solved {}/{tot}, median nodes {:?}, max {:?}", n.len(), n.get(n.len() / 2), n.last());
        }
    }
    let _ = Color::White;
    std::fs::write(&a[6], serde_json::to_vec_pretty(&json!({"schema": "v500_oracle_nodes_v1", "material": a[1], "width": width, "budget": budget, "rows": rows}))?)?;
    Ok(())
}
