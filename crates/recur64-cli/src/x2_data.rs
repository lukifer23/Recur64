//! `recur64 x2 gen` - the exact tool-necessity mate-in-2 dataset (V2.8).
//!
//! Every position: white to move, pawnless, fresh clocks / no history
//! (`fresh_no_history_v1`), NO mate in one, at least one first move forcing mate in
//! two, correct fraction <= 15% of the legal moves, and at least one correct root
//! move whose CandidateFactsV1 vector equals that of an incorrect root move (so the
//! one-ply fact vector alone cannot identify the answer).
//!
//! No PUCT, no network, no external engine: exhaustive rules search. Splits are
//! generated from separate seeds and are hard-disjoint by exact FEN AND by
//! symmetry-canonical key (the 8 board symmetries preserve the rules of these
//! pawnless, castling-free, white-to-move families). The confirmation and tuning
//! sets get a 100% independent label audit through `GameState`.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use clap::Args;
use serde::{Deserialize, Serialize};

use recur64_coproc::world::HISTORY_CONTRACT;
use recur64_core::GameState;
use recur64_runtime::candidate_facts::facts_for;

use crate::x15_tactics::{
    Rng, apply_id, fen_from, fixture_fens, kings_adjacent, mate_in_two_moves, mating_moves, mix,
    place, try_state,
};

pub const DATA_SCHEMA: &str = "v2_mate2_exact_v1";

/// Material families (white pieces besides the king; black has only a king).
pub const KINDS: [(&str, &[char]); 5] = [
    ("kqk", &['K', 'Q']),
    ("krk", &['K', 'R']),
    ("kqqk", &['K', 'Q', 'Q']),
    ("kqrk", &['K', 'Q', 'R']),
    ("krrk", &['K', 'R', 'R']),
];

/// Maximum fraction of the legal moves that may be correct.
pub const MAX_CORRECT_FRACTION: f32 = 0.15;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct V2Pos {
    pub id: String,
    pub kind: String,
    pub fen: String,
    /// Symmetry-canonical key (minimum over the 8 board symmetries).
    pub canon: String,
    pub legal_n: usize,
    /// Legal-action indices of every first move forcing mate in two.
    pub correct: Vec<usize>,
    /// `correct / legal`: the top-1 accuracy of a uniformly random legal move.
    pub chance: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Audit {
    pub positions: usize,
    pub max_legal: usize,
    pub legal_p50: usize,
    pub legal_p90: usize,
    pub legal_p99: usize,
    pub max_replies: usize,
    pub replies_p50: usize,
    pub replies_p90: usize,
    pub replies_p99: usize,
    pub w_cap: usize,
    pub r_cap: usize,
    /// Positions with more legal moves than `w_cap`.
    pub legal_overflow: usize,
    /// (position, candidate) pairs with more replies than `r_cap`.
    pub reply_overflow: usize,
    /// Positions that violate `fresh_no_history_v1` (nonzero clock, terminal, ...).
    pub non_fresh: usize,
    pub mean_chance: f32,
    pub min_correct: usize,
    pub max_correct: usize,
    pub per_kind: BTreeMap<String, usize>,
}

impl Audit {
    /// The experiment refuses to start on any overflow or history violation.
    pub fn enforce(&self, w_cap: usize, r_cap: usize) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.max_legal <= w_cap && self.max_replies <= r_cap,
            "capacity audit: max legal {} / max replies {} exceed the configured w_cap {w_cap} / r_cap {r_cap}",
            self.max_legal,
            self.max_replies
        );
        anyhow::ensure!(
            self.legal_overflow == 0 && self.reply_overflow == 0,
            "capacity overflow: {} positions and {} candidates exceed the caps",
            self.legal_overflow,
            self.reply_overflow
        );
        anyhow::ensure!(
            self.non_fresh == 0,
            "{} positions violate the {HISTORY_CONTRACT} contract",
            self.non_fresh
        );
        Ok(())
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct V2Data {
    pub schema: String,
    pub history_contract: String,
    pub split: String,
    pub seed: u64,
    pub filters: serde_json::Value,
    pub audit: Audit,
    pub positions: Vec<V2Pos>,
}

impl V2Data {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let d: V2Data = serde_json::from_slice(&std::fs::read(path)?)?;
        anyhow::ensure!(
            d.schema == DATA_SCHEMA && d.history_contract == HISTORY_CONTRACT,
            "{}: schema/history contract mismatch ({} / {})",
            path.display(),
            d.schema,
            d.history_contract
        );
        Ok(d)
    }

    pub fn states(&self) -> anyhow::Result<Vec<GameState>> {
        self.positions
            .iter()
            .map(|p| GameState::from_fen(&p.fen).map_err(|e| anyhow::anyhow!("{}: {e}", p.id)))
            .collect()
    }

    /// Recompute the capacity / history audit from the positions themselves (never
    /// trusting the stored numbers) and enforce it against the caps.
    pub fn audit_and_enforce(&self, w_cap: usize, r_cap: usize) -> anyhow::Result<Audit> {
        let a = audit_positions(&self.positions, w_cap, r_cap)?;
        a.enforce(w_cap, r_cap)?;
        Ok(a)
    }

    /// Uniform target over the correct set, `[legal_n]` for position `i`.
    pub fn target(&self, i: usize) -> Vec<f32> {
        let p = &self.positions[i];
        let mut t = vec![0.0f32; p.legal_n];
        for c in &p.correct {
            t[*c] = 1.0 / p.correct.len() as f32;
        }
        t
    }
}

fn pct(sorted: &[usize], q: f64) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

/// The capacity and history-contract audit.
pub fn audit_positions(pos: &[V2Pos], w_cap: usize, r_cap: usize) -> anyhow::Result<Audit> {
    let mut a = Audit {
        positions: pos.len(),
        w_cap,
        r_cap,
        min_correct: usize::MAX,
        ..Default::default()
    };
    let (mut legal, mut replies) = (Vec::new(), Vec::new());
    for p in pos {
        let s = GameState::from_fen(&p.fen).map_err(|e| anyhow::anyhow!("{}: {e}", p.id))?;
        // fresh_no_history_v1: a FEN-only state carries no repetition history by
        // construction; what must still hold is a zero halfmove clock and a live position.
        let clock_zero = s.to_fen().split(' ').nth(4) == Some("0");
        if !clock_zero || s.is_terminal() {
            a.non_fresh += 1;
        }
        let l = s.legal_actions();
        legal.push(l.len());
        if l.len() > w_cap {
            a.legal_overflow += 1;
        }
        for id in &l {
            let c = apply_id(&s, *id);
            let n = if c.is_terminal() {
                0
            } else {
                c.legal_actions().len()
            };
            replies.push(n);
            if n > r_cap {
                a.reply_overflow += 1;
            }
        }
        a.mean_chance += p.chance;
        a.min_correct = a.min_correct.min(p.correct.len());
        a.max_correct = a.max_correct.max(p.correct.len());
        *a.per_kind.entry(p.kind.clone()).or_insert(0) += 1;
    }
    a.mean_chance /= pos.len().max(1) as f32;
    legal.sort_unstable();
    replies.sort_unstable();
    a.max_legal = legal.last().copied().unwrap_or(0);
    a.max_replies = replies.last().copied().unwrap_or(0);
    (a.legal_p50, a.legal_p90, a.legal_p99) =
        (pct(&legal, 0.5), pct(&legal, 0.9), pct(&legal, 0.99));
    (a.replies_p50, a.replies_p90, a.replies_p99) =
        (pct(&replies, 0.5), pct(&replies, 0.9), pct(&replies, 0.99));
    if pos.is_empty() {
        a.min_correct = 0;
    }
    Ok(a)
}

/// Board-symmetry-canonical key of a pawnless, castling-free, white-to-move FEN.
pub fn canonical_key(fen: &str) -> String {
    let mut grid = ['.'; 64];
    for (row, rank) in fen.split(' ').next().unwrap_or("").split('/').enumerate() {
        let r = 7 - row;
        let mut f = 0usize;
        for ch in rank.chars() {
            match ch.to_digit(10) {
                Some(d) => f += d as usize,
                None => {
                    grid[r * 8 + f] = ch;
                    f += 1;
                }
            }
        }
    }
    let mut best: Option<String> = None;
    for t in 0..8 {
        let mut key = String::with_capacity(64);
        for r in 0..8 {
            for f in 0..8 {
                let (mut rr, mut ff) = (r, f);
                if t & 1 != 0 {
                    ff = 7 - ff;
                }
                if t & 2 != 0 {
                    rr = 7 - rr;
                }
                if t & 4 != 0 {
                    std::mem::swap(&mut rr, &mut ff);
                }
                key.push(grid[rr * 8 + ff]);
            }
        }
        if best.as_ref().is_none_or(|b| key < *b) {
            best = Some(key);
        }
    }
    best.unwrap_or_default()
}

/// Some correct move shares its CandidateFactsV1 row with an incorrect move.
fn root_fact_ambiguous(state: &GameState, correct: &[usize]) -> bool {
    let facts = facts_for(state);
    let n = facts.len() / 8;
    let row = |i: usize| &facts[i * 8..(i + 1) * 8];
    correct
        .iter()
        .any(|&c| (0..n).any(|w| !correct.contains(&w) && row(w) == row(c)))
}

#[derive(Default, Clone, Serialize, Debug)]
pub struct KindStats {
    pub tried: u64,
    pub mate2: u64,
    pub rejected_fraction: u64,
    pub rejected_ambiguity: u64,
    pub accepted_raw: u64,
}

fn gen_kind(
    seed: u64,
    kind: &str,
    white: &[char],
    n: usize,
    threads: usize,
    forbid_fen: &HashSet<String>,
    forbid_canon: &HashSet<String>,
) -> anyhow::Result<(Vec<V2Pos>, KindStats)> {
    let per = n.div_ceil(threads) + n / 8 + 2;
    let results: Vec<anyhow::Result<(Vec<V2Pos>, KindStats)>> = std::thread::scope(|scope| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                scope.spawn(move || {
                    let mut rng = Rng(mix(seed ^ mix(t as u64 + 101)));
                    let (mut out, mut st) = (Vec::new(), KindStats::default());
                    let mut local = HashSet::new();
                    while out.len() < per {
                        st.tried += 1;
                        anyhow::ensure!(
                            st.tried < 40_000_000,
                            "{kind}: could not find {per} positions ({st:?})"
                        );
                        let mut pieces: Vec<char> = white.to_vec();
                        pieces.push('k');
                        let placed = place(&mut rng, &pieces);
                        if kings_adjacent(&placed) {
                            continue;
                        }
                        let fen = fen_from(&placed);
                        let Some(state) = try_state(&fen) else {
                            continue;
                        };
                        let correct = mate_in_two_moves(&state); // empty if a mate in one exists
                        if correct.is_empty() {
                            continue;
                        }
                        st.mate2 += 1;
                        let legal_n = state.legal_actions().len();
                        if correct.len() as f32 > MAX_CORRECT_FRACTION * legal_n as f32 {
                            st.rejected_fraction += 1;
                            continue;
                        }
                        if !root_fact_ambiguous(&state, &correct) {
                            st.rejected_ambiguity += 1;
                            continue;
                        }
                        let canon = canonical_key(&fen);
                        if forbid_fen.contains(&fen)
                            || forbid_canon.contains(&canon)
                            || !local.insert(canon.clone())
                        {
                            continue;
                        }
                        st.accepted_raw += 1;
                        out.push(V2Pos {
                            id: String::new(),
                            kind: kind.into(),
                            chance: correct.len() as f32 / legal_n as f32,
                            fen,
                            canon,
                            legal_n,
                            correct,
                        });
                    }
                    Ok((out, st))
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err(anyhow::anyhow!("worker panicked")))
            })
            .collect()
    });
    let (mut all, mut stats) = (Vec::new(), KindStats::default());
    for r in results {
        let (v, s) = r?;
        all.extend(v);
        stats.tried += s.tried;
        stats.mate2 += s.mate2;
        stats.rejected_fraction += s.rejected_fraction;
        stats.rejected_ambiguity += s.rejected_ambiguity;
        stats.accepted_raw += s.accepted_raw;
    }
    // Thread-independent: dedupe by canonical key, order by a seeded hash, take n.
    all.sort_by(|a, b| a.canon.cmp(&b.canon));
    all.dedup_by(|a, b| a.canon == b.canon);
    anyhow::ensure!(
        all.len() >= n,
        "{kind}: only {} unique positions for {n} requested ({stats:?})",
        all.len()
    );
    all.sort_by_key(|p| {
        mix(seed
            ^ p.canon
                .bytes()
                .fold(0u64, |h, b| h.wrapping_mul(131).wrapping_add(u64::from(b))))
    });
    all.truncate(n);
    Ok((all, stats))
}

/// The independent 100% label audit: recompute, through `GameState` and its own
/// termination, both "no mate in one" and every candidate's forced-mate status.
pub fn independent_label_audit(d: &V2Data) -> anyhow::Result<usize> {
    for p in &d.positions {
        let state = GameState::from_fen(&p.fen).map_err(|e| anyhow::anyhow!("{}: {e}", p.id))?;
        anyhow::ensure!(
            mating_moves(&state).is_empty(),
            "{}: a mate in one exists",
            p.id
        );
        let legal = state.legal_actions();
        anyhow::ensure!(legal.len() == p.legal_n, "{}: legal count changed", p.id);
        for (i, id) in legal.iter().enumerate() {
            let s1 = apply_id(&state, *id);
            let forced = !s1.is_terminal()
                && !s1.legal_actions().is_empty()
                && s1.legal_actions().iter().all(|r| {
                    let s2 = apply_id(&s1, *r);
                    !s2.is_terminal() && !mating_moves(&s2).is_empty()
                });
            anyhow::ensure!(
                forced == p.correct.contains(&i),
                "{}: GameState label check disagrees on move {i}",
                p.id
            );
        }
    }
    Ok(d.positions.len())
}

#[derive(Args, Debug)]
pub struct GenArgs {
    #[arg(long)]
    pub out_dir: PathBuf,
    #[arg(long, default_value_t = 2400)]
    pub train_per_kind: usize,
    /// The KQK and KRK families have only a few hundred distinct symmetry classes that
    /// satisfy the filters (measured: ~560 for KQK); their TRAIN count is capped here
    /// and the shortfall is reported, never silently relaxed.
    #[arg(long, default_value_t = 350)]
    pub train_small_family_cap: usize,
    #[arg(long, default_value_t = 52)]
    pub tune_per_kind: usize,
    #[arg(long, default_value_t = 52)]
    pub confirm_per_kind: usize,
    #[arg(long, default_value_t = 20262001)]
    pub train_seed: u64,
    #[arg(long, default_value_t = 20262002)]
    pub tune_seed: u64,
    #[arg(long, default_value_t = 20262003)]
    pub confirm_seed: u64,
    #[arg(long, default_value_t = 6)]
    pub threads: usize,
    #[arg(long, default_value_t = 64)]
    pub w_cap: usize,
    #[arg(long, default_value_t = 16)]
    pub r_cap: usize,
    /// V1 fixture files (tactics / mate-in-2 / mate-in-3 JSON) to be disjoint from.
    #[arg(long)]
    pub disjoint_fixtures: Vec<PathBuf>,
    /// V1 ReasoningTargetsV1 files to be disjoint from.
    #[arg(long)]
    pub disjoint_targets: Vec<PathBuf>,
}

pub fn run_gen(args: GenArgs) -> anyhow::Result<()> {
    let mut forbid_fen: HashSet<String> = HashSet::new();
    for p in &args.disjoint_fixtures {
        forbid_fen.extend(fixture_fens(p)?);
    }
    for p in &args.disjoint_targets {
        let t = recur64_runtime::reasoning_targets::ReasoningTargetsV1::load(p)?;
        forbid_fen.extend(t.positions.iter().map(|q| q.start_fen.clone()));
        forbid_fen.extend(t.positions.iter().map(|q| q.fen.clone()));
    }
    // Symmetry keys are only defined for pawnless white-to-move placements; a V1 FEN
    // outside that family simply has no equivalent in the V2 families.
    let mut forbid_canon: HashSet<String> = forbid_fen
        .iter()
        .filter(|f| {
            f.split(' ').nth(1) == Some("w")
                && !f.split(' ').next().unwrap_or("").contains(['p', 'P'])
        })
        .map(|f| canonical_key(f))
        .collect();
    println!(
        "V1 data forbidden: {} FENs, {} symmetry classes",
        forbid_fen.len(),
        forbid_canon.len()
    );
    std::fs::create_dir_all(&args.out_dir)?;
    let mut summary = serde_json::Map::new();
    // Confirm and tune first, so train can never take a position from either.
    for (split, seed, per_kind) in [
        ("confirm", args.confirm_seed, args.confirm_per_kind),
        ("tune", args.tune_seed, args.tune_per_kind),
        ("train", args.train_seed, args.train_per_kind),
    ] {
        let started = std::time::Instant::now();
        let mut positions = Vec::new();
        let mut stats = BTreeMap::new();
        for (kind, white) in KINDS {
            let want = if split == "train" && matches!(kind, "kqk" | "krk") {
                per_kind.min(args.train_small_family_cap)
            } else {
                per_kind
            };
            if want < per_kind {
                println!("  {split} {kind}: capped at {want} of the requested {per_kind} (small pool)");
            }
            let (v, st) = gen_kind(
                seed,
                kind,
                white,
                want,
                args.threads.max(1),
                &forbid_fen,
                &forbid_canon,
            )?;
            println!(
                "  {split:<7} {kind:<5} {:>5} positions  (tried {}, mate2 {}, over-fraction {}, non-ambiguous {})",
                v.len(),
                st.tried,
                st.mate2,
                st.rejected_fraction,
                st.rejected_ambiguity
            );
            positions.extend(v);
            stats.insert(kind.to_string(), st);
        }
        for (i, p) in positions.iter_mut().enumerate() {
            p.id = format!("v2-{split}-{}-{i:05}", p.kind);
        }
        forbid_fen.extend(positions.iter().map(|p| p.fen.clone()));
        forbid_canon.extend(positions.iter().map(|p| p.canon.clone()));
        let audit = audit_positions(&positions, args.w_cap, args.r_cap)?;
        audit.enforce(args.w_cap, args.r_cap)?;
        let data = V2Data {
            schema: DATA_SCHEMA.into(),
            history_contract: HISTORY_CONTRACT.into(),
            split: split.into(),
            seed,
            filters: serde_json::json!({
                "no_mate_in_one": true,
                "max_correct_fraction": MAX_CORRECT_FRACTION,
                "root_fact_ambiguity": "some correct move shares its CandidateFactsV1 vector with an incorrect move",
                "symmetry_canonical_disjointness": "8 board symmetries (pawnless, castling-free, white to move)",
                "generation": stats,
            }),
            audit: audit.clone(),
            positions,
        };
        // 100% independent GameState validation of every label (all splits: it is cheap).
        let checked = independent_label_audit(&data)?;
        let path = args.out_dir.join(format!("{split}.json"));
        std::fs::write(&path, serde_json::to_vec(&data)?)?;
        println!(
            "{split}: {} positions, {checked}/{checked} labels independently verified via GameState; mean chance top-1 {:.4}; correct/pos {}..{}; legal max {} (p50 {} p90 {} p99 {}); replies max {} (p50 {} p90 {} p99 {}); overflow legal {} reply {}; non-fresh {}; {:.0}s -> {}",
            audit.positions,
            audit.mean_chance,
            audit.min_correct,
            audit.max_correct,
            audit.max_legal,
            audit.legal_p50,
            audit.legal_p90,
            audit.legal_p99,
            audit.max_replies,
            audit.replies_p50,
            audit.replies_p90,
            audit.replies_p99,
            audit.legal_overflow,
            audit.reply_overflow,
            audit.non_fresh,
            started.elapsed().as_secs_f64(),
            path.display()
        );
        summary.insert(split.into(), serde_json::to_value(&audit)?);
    }
    // Final cross-split disjointness proof, recomputed from the written files.
    let mut all: Vec<V2Data> = Vec::new();
    for s in ["train", "tune", "confirm"] {
        all.push(V2Data::load(&args.out_dir.join(format!("{s}.json")))?);
    }
    let mut seen_fen = HashSet::new();
    let mut seen_canon = HashSet::new();
    for d in &all {
        for p in &d.positions {
            anyhow::ensure!(
                seen_fen.insert(p.fen.clone()),
                "{}: FEN in two splits",
                p.id
            );
            anyhow::ensure!(
                seen_canon.insert(p.canon.clone()),
                "{}: symmetry class in two splits (or duplicated)",
                p.id
            );
            anyhow::ensure!(!forbidden_v1(&args, p)?, "{}: overlaps V1 data", p.id);
        }
    }
    println!(
        "disjointness verified from the written files: {} FENs, {} symmetry classes, no overlap between train/tune/confirm or with V1 data",
        seen_fen.len(),
        seen_canon.len()
    );
    std::fs::write(
        args.out_dir.join("audit.json"),
        serde_json::to_vec_pretty(&serde_json::Value::Object(summary))?,
    )?;
    Ok(())
}

fn forbidden_v1(args: &GenArgs, p: &V2Pos) -> anyhow::Result<bool> {
    thread_local! {
        static CACHE: std::cell::RefCell<Option<(HashSet<String>, HashSet<String>)>> = const { std::cell::RefCell::new(None) };
    }
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if c.is_none() {
            let mut fens: HashSet<String> = HashSet::new();
            for f in &args.disjoint_fixtures {
                fens.extend(fixture_fens(f)?);
            }
            for f in &args.disjoint_targets {
                let t = recur64_runtime::reasoning_targets::ReasoningTargetsV1::load(f)?;
                fens.extend(t.positions.iter().map(|q| q.start_fen.clone()));
                fens.extend(t.positions.iter().map(|q| q.fen.clone()));
            }
            let canon = fens
                .iter()
                .filter(|f| {
                    f.split(' ').nth(1) == Some("w")
                        && !f.split(' ').next().unwrap_or("").contains(['p', 'P'])
                })
                .map(|f| canonical_key(f))
                .collect();
            *c = Some((fens, canon));
        }
        let (fens, canon) = c.as_ref().expect("initialised above");
        Ok(fens.contains(&p.fen) || canon.contains(&p.canon))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_board_and_its_reflections_share_a_canonical_key() {
        let a = "4k3/8/8/8/8/8/8/K3Q3 w - - 0 1";
        let mirrored = "3k4/8/8/8/8/8/8/3Q3K w - - 0 1"; // file reflection
        let flipped = "K3Q3/8/8/8/8/8/8/4k3 w - - 0 1"; // rank reflection
        let transposed = "8/8/8/8/8/8/8/8 w - - 0 1";
        assert_eq!(canonical_key(a), canonical_key(mirrored));
        assert_eq!(canonical_key(a), canonical_key(flipped));
        assert_ne!(canonical_key(a), canonical_key(transposed));
        assert_ne!(
            canonical_key(a),
            canonical_key("4k3/8/8/8/8/8/8/K2Q4 w - - 0 1")
        );
    }

    #[test]
    fn a_transposed_position_shares_a_canonical_key() {
        // Transposition (a1-h8 diagonal reflection) of one placement.
        let a = "8/8/8/8/8/8/1k6/K1Q5 w - - 0 1";
        // K a1->a1, Q c1->a3, k b2->b2 after transposition (file<->rank).
        let t = "8/8/8/8/8/Q7/1k6/K7 w - - 0 1";
        assert_eq!(canonical_key(a), canonical_key(t));
    }

    #[test]
    fn a_known_mate_in_two_passes_the_independent_audit_and_chance_is_correct_over_legal() {
        // Ka1 Kc3 Qh1? use a position built by the generator itself.
        let mut rng = Rng(7);
        let mut found = None;
        for _ in 0..2_000_000 {
            let placed = place(&mut rng, &['K', 'R', 'k']);
            if kings_adjacent(&placed) {
                continue;
            }
            let fen = fen_from(&placed);
            let Some(s) = try_state(&fen) else { continue };
            let c = mate_in_two_moves(&s);
            if !c.is_empty() {
                found = Some((fen, s, c));
                break;
            }
        }
        let (fen, s, c) = found.expect("a KRK mate-in-2 position exists");
        let legal_n = s.legal_actions().len();
        let d = V2Data {
            schema: DATA_SCHEMA.into(),
            history_contract: HISTORY_CONTRACT.into(),
            split: "t".into(),
            seed: 0,
            filters: serde_json::Value::Null,
            audit: Audit::default(),
            positions: vec![V2Pos {
                id: "t".into(),
                kind: "krk".into(),
                canon: canonical_key(&fen),
                fen,
                legal_n,
                chance: c.len() as f32 / legal_n as f32,
                correct: c,
            }],
        };
        assert_eq!(independent_label_audit(&d).unwrap(), 1);
        let a = audit_positions(&d.positions, 64, 16).unwrap();
        assert_eq!(a.non_fresh, 0);
        assert!(a.enforce(64, 16).is_ok());
        assert!(a.enforce(4, 16).is_err(), "a too-small cap must be refused");
    }
}
