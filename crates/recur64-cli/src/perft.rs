//! `recur64 perft` — validate legal move generation and transition.

use clap::Args;

use recur64_core::Board;
use recur64_core::perft::{perft, perft_divide};

#[derive(Args, Debug)]
pub struct PerftArgs {
    #[arg(long)]
    pub fen: String,
    #[arg(long)]
    pub depth: u32,
    /// Print a per-move node breakdown instead of only the total.
    #[arg(long, default_value_t = false)]
    pub divide: bool,
}

pub fn run_perft(args: PerftArgs) -> anyhow::Result<()> {
    let board: Board = args
        .fen
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid FEN {:?}: {e}", args.fen))?;
    println!("fen   : {}", board);
    println!("depth : {}", args.depth);
    if args.divide {
        let mut total = 0u64;
        for (mv, n) in perft_divide(&board, args.depth) {
            println!("{mv}: {n}");
            total += n;
        }
        println!("total : {total}");
    } else {
        let nodes = perft(&board, args.depth);
        println!("nodes : {nodes}");
    }
    Ok(())
}
