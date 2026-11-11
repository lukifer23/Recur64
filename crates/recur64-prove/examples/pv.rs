//! Show a proof as a game: solve one position with a learned net and print the principal line (the defender's most
//! resilient defence) with the FEN after every ply, as JSON for display.
//!
//! cargo run --release -p recur64-prove --example pv -- <net.json> <KRvK|KQvK|KBNvK> <depth> <width> <budget> <out.json> [index]
use cozy_chess::{Board, Piece};
use recur64_prove::net::{Learned, Mlp};
use recur64_prove::search::{Prover, verify};
use recur64_prove::split::{Part, canon_key, part_of};
use recur64_prove::tb::Tablebase;
use serde_json::json;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 7, "usage: pv <net.json> <material> <depth> <width> <budget> <out.json> [index]");
    let extra: Vec<Piece> = match a[2].as_str() {
        "KRvK" => vec![Piece::Rook],
        "KQvK" => vec![Piece::Queen],
        "KBNvK" => vec![Piece::Bishop, Piece::Knight],
        o => anyhow::bail!("material {o}"),
    };
    let depth: u8 = a[3].parse()?;
    let width: usize = a[4].parse()?;
    let budget: u64 = a[5].parse()?;
    let index: usize = a.get(7).map_or(Ok(0), |s| s.parse())?;
    let net = Mlp::load(std::path::Path::new(&a[1]))?;
    let tb = Tablebase::build(&a[2], &extra)?;
    let mut cands: Vec<Board> = tb
        .won_white_positions()
        .into_iter()
        .filter(|(b, d)| *d == depth && part_of(b) == Part::Test)
        .map(|(b, _)| b)
        .collect();
    cands.sort_by_key(canon_key);
    let root = cands.get(index).ok_or_else(|| anyhow::anyhow!("no TEST position of depth {depth} at index {index} ({} available)", cands.len()))?.clone();
    let mut o = Learned { net };
    let s = Prover::new(&mut o, Some(width), budget).solve(&root, depth + 4);
    let proof = s.proof.ok_or_else(|| anyhow::anyhow!("not solved within the budget ({} nodes)", s.nodes))?;
    let checked = verify(&root, &proof).map_err(|e| anyhow::anyhow!("proof failed the independent checker: {e}"))?;
    let mut b = root.clone();
    let mut plies = vec![json!({"fen": b.to_string(), "move": null})];
    for (m, r) in proof.principal_line() {
        b.play_unchecked(m);
        plies.push(json!({"fen": b.to_string(), "move": m.to_string(), "by": "attacker"}));
        if let Some(r) = r {
            b.play_unchecked(r);
            plies.push(json!({"fen": b.to_string(), "move": r.to_string(), "by": "defender"}));
        }
    }
    std::fs::write(
        &a[6],
        serde_json::to_vec_pretty(&json!({
            "schema": "v500_pv_v1", "material": a[2], "root": root.to_string(), "optimal_moves_tablebase": depth,
            "proof_checked_moves": checked, "width": width, "nodes": s.nodes, "proof_tree_nodes": proof.size(),
            "plies": plies,
        }))?,
    )?;
    eprintln!("{}: optimal {depth}, proof {checked} moves, {} nodes, proof tree {} attacker nodes", a[2], s.nodes, proof.size());
    Ok(())
}
