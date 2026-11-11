//! Build a tablebase and print its size and longest mate (validation against the known maxima).
use cozy_chess::Piece;
use recur64_prove::tb::Tablebase;

fn main() -> anyhow::Result<()> {
    let name = std::env::args().nth(1).unwrap_or_else(|| "KBNvK".into());
    let extra: Vec<Piece> = match name.as_str() {
        "KBNvK" => vec![Piece::Bishop, Piece::Knight],
        "KQvK" => vec![Piece::Queen],
        "KRvK" => vec![Piece::Rook],
        "KQRvK" => vec![Piece::Queen, Piece::Rook],
        other => anyhow::bail!("unknown material {other}"),
    };
    let t = std::time::Instant::now();
    let tb = Tablebase::build(&name, &extra)?;
    let won = tb.won_white_positions();
    let mut by_depth = vec![0usize; 80];
    for (_, d) in &won {
        by_depth[*d as usize] += 1;
    }
    eprintln!("{name}: {} legal positions, {} won white-to-move, longest mate {} plies = {} moves, built in {:.0}s",
        tb.positions.len(), won.len(), tb.max_plies_white, tb.max_plies_white.div_ceil(2), t.elapsed().as_secs_f64());
    eprintln!("white-to-move positions by optimal depth (moves): {:?}", by_depth.iter().enumerate().filter(|(_, c)| **c > 0).collect::<Vec<_>>());
    Ok(())
}
