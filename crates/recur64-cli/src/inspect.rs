//! `recur64 validate-position` and `recur64 encode` — correctness/debug views.

use clap::Args;

use recur64_core::{GameState, PromotionCode, encode_observation_v1};

#[derive(Args, Debug)]
pub struct ValidateArgs {
    #[arg(long)]
    pub fen: String,
}

#[derive(Args, Debug)]
pub struct EncodeArgs {
    #[arg(long)]
    pub fen: String,
    /// Print every canonical legal ActionId and its UCI move.
    #[arg(long, default_value_t = false)]
    pub actions: bool,
}

pub fn run_validate(args: ValidateArgs) -> anyhow::Result<()> {
    let g = GameState::from_fen(&args.fen)?;
    let board = g.board();
    let side = g.side_to_move();
    let rights = board.castle_rights(side);

    println!("fen             : {}", g.to_fen());
    println!("side to move    : {side:?}");
    println!("termination     : {:?}", g.termination().map(|t| t.label()));
    println!("outcome         : {:?}", g.outcome());
    println!("repetition count: {}", g.repetition_count());
    println!("halfmove clock  : {}", board.halfmove_clock());
    println!(
        "castling (own)  : kingside={} queenside={}",
        rights.short.is_some(),
        rights.long.is_some()
    );
    println!("en passant file : {:?}", board.en_passant());
    println!("legal moves     : {}", g.legal_standard_moves().len());

    // Candidate invariants.
    let actions = g.legal_actions();
    let mut unique = actions.clone();
    unique.sort_unstable();
    unique.dedup();
    let unique_ok = unique.len() == actions.len();
    let count_ok = actions.len() == g.legal_standard_moves().len();
    println!("legal actions   : {}", actions.len());
    println!("actions unique  : {unique_ok}");
    println!("count matches   : {count_ok}");

    if !unique_ok || !count_ok {
        anyhow::bail!("legal candidate invariant violated for {}", args.fen);
    }

    // Action <-> move bijection check.
    let p = g.perspective();
    for id in &actions {
        let (f, t, pr) = id.to_physical(p);
        let promo = if pr.is_none() { None } else { Some(pr) };
        let mv = recur64_core::StandardMove::new(f, t, promo);
        let cozy = mv.to_cozy(board)?;
        if !board.is_legal(cozy) {
            anyhow::bail!("action {id:?} decoded to an illegal move");
        }
    }
    println!("bijection       : ok");
    Ok(())
}

pub fn run_encode(args: EncodeArgs) -> anyhow::Result<()> {
    let g = GameState::from_fen(&args.fen)?;
    let obs = encode_observation_v1(&g);
    let nonzero = obs.as_slice().iter().filter(|v| **v != 0.0).count();
    println!("fen             : {}", g.to_fen());
    println!("observation len : {}", obs.as_slice().len());
    println!("nonzero feats   : {nonzero}");
    println!(
        "castling (own)  : ks={} qs={}  opp ks={} qs={}",
        obs.get(0, recur64_core::observation::CASTLE_OFFSET),
        obs.get(0, recur64_core::observation::CASTLE_OFFSET + 1),
        obs.get(0, recur64_core::observation::CASTLE_OFFSET + 2),
        obs.get(0, recur64_core::observation::CASTLE_OFFSET + 3),
    );
    println!(
        "halfmove feat   : {}",
        obs.get(0, recur64_core::observation::HALFMOVE_OFFSET)
    );
    println!(
        "repetition feat : {}",
        obs.get(0, recur64_core::observation::REPETITION_OFFSET)
    );

    let actions = g.legal_actions();
    println!("legal actions   : {}", actions.len());
    if args.actions {
        let p = g.perspective();
        for id in &actions {
            let (f, t, pr) = id.to_physical(p);
            let promo = if pr.is_none() { None } else { Some(pr) };
            let mv = recur64_core::StandardMove::new(f, t, promo);
            println!(
                "  {:<6} id={:<6} promo={}",
                mv.to_uci(),
                id.index(),
                pr.code()
            );
        }
        // Silence unused import warning when actions are printed.
        let _ = PromotionCode::NONE;
    }
    Ok(())
}
