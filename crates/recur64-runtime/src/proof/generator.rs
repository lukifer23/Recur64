//! Exact proof-position generator (`ProofTargetsV1`).
//!
//! Positions are white-to-move, pawnless, castling-free, with a bare black king
//! (the families in [`super::targets::FAMILIES`]), fresh clocks and no history.
//! Each sampled position is classified once by its exact mate depth; it is routed
//! to that depth's quota. No network, no PUCT, no heuristic.
//!
//! Filters (pre-registered): for depth >= 2 the correct moves must be at most
//! [`MAX_CORRECT_FRACTION`] of the legal moves, and at least one correct root
//! move must share its `CandidateFactsV1` vector with an incorrect root move (so
//! the one-ply facts alone cannot identify the answer). Depth 1 is exposed by the
//! `mate` fact by construction and is exempt. A cell that cannot be filled is an
//! error carrying the measured pool composition; the filters are never relaxed.
//!
//! Determinism: every worker thread draws from its own seeded stream and fills
//! fixed per-thread quotas, so the accepted set does not depend on scheduling.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Instant;

use cozy_chess::Board;
use serde::Serialize;

use recur64_core::{ActionId, GameState, StandardMove, candidate_facts};

use super::mate::MateSolver;
use super::targets::{FAMILIES, ProofPosition, ProofTargets, Split};

/// Maximum fraction of the legal moves that may be correct (depth >= 2).
pub const MAX_CORRECT_FRACTION: f32 = 0.15;

/// Generator request for one split.
#[derive(Debug, Clone)]
pub struct GenSpec {
    pub split: Split,
    pub seed: u64,
    /// Positions per (family, depth band).
    pub per_cell: usize,
    /// Depth bands to generate (each in 1..=5).
    pub depths: Vec<u8>,
    pub threads: usize,
    /// Give up on a worker after this many sampled positions (reports the pool).
    pub max_tries_per_thread: u64,
    pub forbid_fen: HashSet<String>,
    pub forbid_canon: HashSet<String>,
}

/// Per-(family, depth) generation statistics.
#[derive(Debug, Clone, Default, Serialize)]
pub struct CellReport {
    pub family: String,
    pub depth: u8,
    pub accepted: usize,
    /// Positions of this depth found before filters.
    pub found: u64,
    pub rejected_fraction: u64,
    pub rejected_ambiguity: u64,
    pub rejected_duplicate_or_forbidden: u64,
}

/// Whole-run report, including the measured pool composition.
#[derive(Debug, Clone, Default, Serialize)]
pub struct GenReport {
    pub cells: Vec<CellReport>,
    /// Sampled positions by exact depth ("none" = deeper than the max band).
    pub pool_by_family: BTreeMap<String, BTreeMap<String, u64>>,
    pub sampled: u64,
    pub solver_nodes: u64,
    pub max_legal_root: usize,
    pub wall_s: f64,
}

pub struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

pub fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Board-symmetry-canonical key of a pawnless, castling-free, white-to-move FEN
/// (minimum over the 8 dihedral symmetries; these preserve the rules of the
/// families above).
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

/// The canonical representative of a symmetry class, as a FEN. The canonical
/// key is itself a `rank * 8 + file` grid, so this is independent of which
/// symmetric placement the enumeration happened to visit first.
pub fn fen_from_canon(canon: &str) -> String {
    let mut grid = ['.'; 64];
    for (slot, c) in grid.iter_mut().zip(canon.chars()) {
        *slot = c;
    }
    fen_from_grid(&grid)
}

/// FEN of white pieces (uppercase, includes 'K') plus the black king on the
/// given distinct squares, white to move, fresh clocks.
fn fen_for(white: &[char], squares: &[usize]) -> String {
    let mut grid = ['.'; 64];
    for (c, &sq) in white.iter().zip(squares) {
        grid[sq] = *c;
    }
    grid[squares[white.len()]] = 'k';
    fen_from_grid(&grid)
}

/// The white-to-move, fresh-clock FEN of a `rank * 8 + file` piece grid.
fn fen_from_grid(grid: &[char; 64]) -> String {
    let mut out = String::new();
    for rank in (0..8).rev() {
        let mut empty = 0;
        for file in 0..8 {
            let c = grid[rank * 8 + file];
            if c == '.' {
                empty += 1;
            } else {
                if empty > 0 {
                    out.push_str(&empty.to_string());
                    empty = 0;
                }
                out.push(c);
            }
        }
        if empty > 0 {
            out.push_str(&empty.to_string());
        }
        if rank > 0 {
            out.push('/');
        }
    }
    out.push_str(" w - - 0 1");
    out
}

fn kings_adjacent(wk: usize, bk: usize) -> bool {
    let (wr, wf) = (wk / 8, wk % 8);
    let (br, bf) = (bk / 8, bk % 8);
    wr.abs_diff(br) <= 1 && wf.abs_diff(bf) <= 1
}

/// A random legal, live position of the family, or `None` if this draw is illegal.
pub(crate) fn sample_position(rng: &mut Rng, white: &[char]) -> Option<(String, GameState)> {
    let n = white.len() + 1;
    let mut squares = Vec::with_capacity(n);
    while squares.len() < n {
        let s = rng.below(64) as usize;
        if !squares.contains(&s) {
            squares.push(s);
        }
    }
    // white[0] is the king; the black king is the last square.
    if kings_adjacent(squares[0], squares[n - 1]) {
        return None;
    }
    let fen = fen_for(white, &squares);
    let state = GameState::from_fen(&fen).ok()?; // rejects illegal positions
    if state.is_terminal() || state.legal_actions().is_empty() {
        return None;
    }
    Some((fen, state))
}

/// The legal moves of `state` as cozy moves, in `legal_actions()` order.
pub fn legal_cozy_moves(state: &GameState) -> Vec<cozy_chess::Move> {
    let p = state.perspective();
    state
        .legal_actions()
        .into_iter()
        .map(|id: ActionId| {
            let (f, t, promo) = id.to_physical(p);
            let promo = (!promo.is_none()).then_some(promo);
            StandardMove::new(f, t, promo)
                .to_cozy(state.board())
                .expect("a legal action converts to a cozy move")
        })
        .collect()
}

/// A correct move shares its CandidateFactsV1 row with an incorrect move.
fn root_fact_ambiguous(state: &GameState, correct: &[usize]) -> bool {
    let facts = candidate_facts(state);
    correct
        .iter()
        .any(|&c| (0..facts.len()).any(|w| !correct.contains(&w) && facts[w] == facts[c]))
}

struct WorkerOut {
    /// (family index, depth) -> accepted positions.
    accepted: BTreeMap<(usize, u8), Vec<ProofPosition>>,
    cells: BTreeMap<(usize, u8), CellReport>,
    pool: BTreeMap<usize, BTreeMap<String, u64>>,
    sampled: u64,
    nodes: u64,
    max_legal: usize,
}

#[allow(clippy::too_many_arguments)]
fn worker(spec: &GenSpec, thread: usize, quota: usize, max_depth: u8) -> anyhow::Result<WorkerOut> {
    let mut out = WorkerOut {
        accepted: BTreeMap::new(),
        cells: BTreeMap::new(),
        pool: BTreeMap::new(),
        sampled: 0,
        nodes: 0,
        max_legal: 0,
    };
    let mut solver = MateSolver::new();
    for (fi, (family, white)) in FAMILIES.iter().enumerate() {
        let stream = mix(spec.seed ^ mix(1000 + fi as u64) ^ mix(thread as u64 + 1));
        let mut rng = Rng(stream);
        let mut local: HashSet<String> = HashSet::new();
        let mut tries = 0u64;
        let full = |acc: &BTreeMap<(usize, u8), Vec<ProofPosition>>| {
            spec.depths
                .iter()
                .all(|d| acc.get(&(fi, *d)).is_some_and(|v| v.len() >= quota))
        };
        while !full(&out.accepted) {
            tries += 1;
            if tries > spec.max_tries_per_thread {
                let pool = out.pool.get(&fi).cloned().unwrap_or_default();
                let cells: Vec<_> = out.cells.iter().filter(|((f, _), _)| *f == fi).collect();
                anyhow::bail!(
                    "{family}: could not fill the requested quotas within {tries} samples; \
                     measured pool {pool:?}, cells {cells:?}. The filters are not relaxed."
                );
            }
            let Some((fen, state)) = sample_position(&mut rng, white) else {
                continue;
            };
            out.sampled += 1;
            if solver.table_size() > 3_000_000 {
                out.nodes += solver.nodes;
                solver = MateSolver::new();
            }
            let depth = solver.mate_depth(state.board(), max_depth);
            let label = depth.map_or("none".to_string(), |d| d.to_string());
            *out.pool.entry(fi).or_default().entry(label).or_insert(0) += 1;
            let Some(d) = depth else { continue };
            if !spec.depths.contains(&d) {
                continue;
            }
            if out.accepted.get(&(fi, d)).is_some_and(|v| v.len() >= quota) {
                continue;
            }
            let cell = out.cells.entry((fi, d)).or_insert_with(|| CellReport {
                family: family.to_string(),
                depth: d,
                ..Default::default()
            });
            cell.found += 1;

            let moves = legal_cozy_moves(&state);
            let correct = solver.correct_moves(state.board(), d, &moves);
            if correct.is_empty() {
                anyhow::bail!("{fen}: depth {d} but no correct move (solver inconsistency)");
            }
            out.max_legal = out.max_legal.max(moves.len());
            if d >= 2 {
                if correct.len() as f32 > MAX_CORRECT_FRACTION * moves.len() as f32 {
                    cell.rejected_fraction += 1;
                    continue;
                }
                if !root_fact_ambiguous(&state, &correct) {
                    cell.rejected_ambiguity += 1;
                    continue;
                }
            }
            let canon = canonical_key(&fen);
            if spec.forbid_fen.contains(&fen)
                || spec.forbid_canon.contains(&canon)
                || !local.insert(canon.clone())
            {
                cell.rejected_duplicate_or_forbidden += 1;
                continue;
            }
            cell.accepted += 1;
            let legal: Vec<u16> = state
                .legal_actions()
                .iter()
                .map(|a| a.index() as u16)
                .collect();
            out.accepted
                .entry((fi, d))
                .or_default()
                .push(ProofPosition {
                    id: String::new(),
                    fen,
                    split: spec.split,
                    family: family.to_string(),
                    mate_depth: d,
                    canon,
                    chance_top1: correct.len() as f32 / legal.len() as f32,
                    legal,
                    correct: correct.iter().map(|&i| i as u32).collect(),
                    generator_seed: spec.seed,
                });
        }
    }
    out.nodes += solver.nodes;
    Ok(out)
}

/// Generate one split. Returns the dataset and the measured report.
pub fn generate(spec: &GenSpec) -> anyhow::Result<(ProofTargets, GenReport)> {
    anyhow::ensure!(spec.per_cell > 0 && !spec.depths.is_empty() && spec.threads > 0);
    let max_depth = *spec.depths.iter().max().unwrap();
    let t0 = Instant::now();
    // Slack so cross-thread duplicates cannot leave a cell short.
    let quota = spec.per_cell.div_ceil(spec.threads) + spec.per_cell / 8 + 2;
    let results: Vec<anyhow::Result<WorkerOut>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..spec.threads)
            .map(|t| scope.spawn(move || worker(spec, t, quota, max_depth)))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("generator worker panicked"))
            .collect()
    });
    let mut report = GenReport::default();
    let mut merged: BTreeMap<(usize, u8), Vec<ProofPosition>> = BTreeMap::new();
    let mut cells: BTreeMap<(usize, u8), CellReport> = BTreeMap::new();
    for r in results {
        let w = r?;
        report.sampled += w.sampled;
        report.solver_nodes += w.nodes;
        report.max_legal_root = report.max_legal_root.max(w.max_legal);
        for (fi, pool) in w.pool {
            let dst = report
                .pool_by_family
                .entry(FAMILIES[fi].0.to_string())
                .or_default();
            for (k, v) in pool {
                *dst.entry(k).or_insert(0) += v;
            }
        }
        for (k, v) in w.accepted {
            merged.entry(k).or_default().extend(v);
        }
        for (k, c) in w.cells {
            let e = cells.entry(k).or_default();
            e.family = c.family;
            e.depth = c.depth;
            e.found += c.found;
            e.rejected_fraction += c.rejected_fraction;
            e.rejected_ambiguity += c.rejected_ambiguity;
            e.rejected_duplicate_or_forbidden += c.rejected_duplicate_or_forbidden;
        }
    }
    // Deterministic selection: cross-thread dedup by canonical key, then the
    // first `per_cell` by (canon, fen).
    let mut positions = Vec::new();
    for ((fi, d), mut v) in merged {
        v.sort_by(|a, b| (&a.canon, &a.fen).cmp(&(&b.canon, &b.fen)));
        v.dedup_by(|a, b| a.canon == b.canon);
        anyhow::ensure!(
            v.len() >= spec.per_cell,
            "{} depth {d}: only {} unique positions after cross-thread dedup, need {}",
            FAMILIES[fi].0,
            v.len(),
            spec.per_cell
        );
        v.truncate(spec.per_cell);
        for (i, mut p) in v.into_iter().enumerate() {
            p.id = format!(
                "{}-{}-m{d}-{i:05}",
                spec.split.label(),
                FAMILIES[fi].0.to_lowercase()
            );
            positions.push(p);
        }
        let c = cells.entry((fi, d)).or_default();
        c.accepted = spec.per_cell;
    }
    report.cells = cells.into_values().collect();
    report.wall_s = t0.elapsed().as_secs_f64();
    let filters = serde_json::json!({
        "max_correct_fraction": MAX_CORRECT_FRACTION,
        "fact_ambiguity_required_for_depth_at_least": 2,
        "per_cell": spec.per_cell,
        "depths": spec.depths,
        "families": FAMILIES.iter().map(|f| f.0).collect::<Vec<_>>(),
    });
    let targets = ProofTargets::new(spec.split, spec.seed, filters, positions);
    Ok((targets, report))
}

/// Board of a stored position (used by audits and tools).
pub fn board_of(p: &ProofPosition) -> anyhow::Result<Board> {
    p.fen
        .parse::<Board>()
        .map_err(|e| anyhow::anyhow!("{}: {e:?}", p.id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_key_is_symmetry_invariant() {
        // A position and its file-mirror share a key; a different one does not.
        let a = "7k/8/6K1/8/8/8/8/1Q6 w - - 0 1";
        let b = "k7/8/1K6/8/8/8/8/6Q1 w - - 0 1"; // mirrored files
        let c = "7k/8/6K1/8/8/8/8/2Q5 w - - 0 1";
        assert_eq!(canonical_key(a), canonical_key(b));
        assert_ne!(canonical_key(a), canonical_key(c));
    }

    #[test]
    fn sampled_positions_are_legal_live_and_in_family() {
        let mut rng = Rng(5);
        let mut seen = 0;
        while seen < 200 {
            if let Some((fen, s)) = sample_position(&mut rng, &['K', 'Q', 'R']) {
                assert!(!s.is_terminal());
                assert_eq!(s.to_fen(), fen);
                seen += 1;
            }
        }
    }

    #[test]
    fn enumeration_is_independent_of_the_thread_count() {
        let (ra, a) = enumerate_pool(1, 3, 2).unwrap();
        let (rb, b) = enumerate_pool(1, 3, 7).unwrap();
        assert_eq!(ra.canonical_classes, rb.canonical_classes);
        assert_eq!(ra.eligible_by_depth, rb.eligible_by_depth);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert_eq!((&x.canon, &x.fen, x.depth), (&y.canon, &y.fen, y.depth));
        }
        // The stored FEN is the canonical representative itself.
        for c in a.iter().take(50) {
            assert_eq!(c.canon, canonical_key(&c.fen));
            assert_eq!(c.fen, fen_from_canon(&c.canon));
        }
    }

    #[test]
    fn scale_rule_uses_targets_when_the_pool_allows_and_splits_80_10_10_otherwise() {
        assert_eq!(
            plan_cell(5000, 1000, 100, 100),
            CellPlan {
                train: 1000,
                tune: 100,
                confirm: 100
            }
        );
        assert_eq!(
            plan_cell(1200, 1000, 100, 100),
            CellPlan {
                train: 1000,
                tune: 100,
                confirm: 100
            }
        );
        assert_eq!(
            plan_cell(1076, 1000, 100, 100),
            CellPlan {
                train: 862,
                tune: 107,
                confirm: 107
            }
        );
        assert_eq!(
            plan_cell(189, 1000, 100, 100),
            CellPlan {
                train: 153,
                tune: 18,
                confirm: 18
            }
        );
    }

    #[test]
    fn exhaustive_pool_enumerates_the_whole_kqk_space() {
        let r = exhaustive_pool(0, 2, 4).unwrap();
        assert_eq!(r.family, "KQvK");
        assert_eq!(r.raw_placements, 64 * 63 * 62);
        assert!(r.legal_live < r.raw_placements && r.legal_live > 0);
        // Canonical classes: about legal/8 (fixed points make it slightly more).
        assert!(r.canonical_classes * 8 >= r.legal_live);
        assert!(r.canonical_classes * 8 < r.legal_live + r.legal_live / 4);
        let total: u64 = r.classes_by_depth.values().sum();
        assert_eq!(total, r.canonical_classes);
        assert!(r.eligible_by_depth.get("1").copied().unwrap_or(0) > 0);
    }

    #[test]
    fn small_generation_is_deterministic_and_filtered() {
        let spec = |threads| GenSpec {
            split: Split::Train,
            seed: 11,
            per_cell: 3,
            depths: vec![1, 2],
            threads,
            max_tries_per_thread: 5_000_000,
            forbid_fen: HashSet::new(),
            forbid_canon: HashSet::new(),
        };
        let (a, ra) = generate(&spec(2)).unwrap();
        let (b, _) = generate(&spec(2)).unwrap();
        assert_eq!(
            a.digest, b.digest,
            "same seed and thread count reproduce exactly"
        );
        a.validate().unwrap();
        assert_eq!(a.positions.len(), 5 * 2 * 3);
        for p in &a.positions {
            if p.mate_depth >= 2 {
                assert!(p.chance_top1 <= MAX_CORRECT_FRACTION + 1e-6, "{}", p.id);
            }
        }
        assert!(ra.sampled > 0 && ra.solver_nodes > 0);
    }
}

/// Exact pool of one family: every legal, live placement, deduplicated by
/// symmetry-canonical key, classified by exact mate depth, with the filters
/// applied. This is the ceiling on what hard-disjoint splits can draw from.
#[derive(Debug, Clone, Default, Serialize)]
pub struct PoolReport {
    pub family: String,
    /// Raw placements enumerated (distinct squares, kings not adjacent).
    pub raw_placements: u64,
    /// Legal, live placements.
    pub legal_live: u64,
    /// Symmetry-canonical classes among legal, live placements.
    pub canonical_classes: u64,
    /// Canonical classes by exact depth (1..=max_depth, "none" = deeper).
    pub classes_by_depth: BTreeMap<String, u64>,
    /// Canonical classes surviving the filters (fraction and ambiguity), by depth.
    pub eligible_by_depth: BTreeMap<String, u64>,
    pub wall_s: f64,
}

/// An eligible position of the exact pool (labels are built later).
#[derive(Debug, Clone)]
pub struct Candidate {
    pub fen: String,
    pub canon: String,
    pub depth: u8,
}

/// Pool sizes only (no candidates kept).
pub fn exhaustive_pool(fi: usize, max_depth: u8, threads: usize) -> anyhow::Result<PoolReport> {
    enumerate_pool(fi, max_depth, threads).map(|(r, _)| r)
}

/// Enumerate the whole placement space of family `fi`. Exact but exhaustive:
/// cost grows as 64^pieces (tens of seconds per four-piece family). Returns the
/// pool report and every ELIGIBLE canonical position, sorted by canonical key so
/// the result does not depend on thread scheduling.
pub fn enumerate_pool(
    fi: usize,
    max_depth: u8,
    threads: usize,
) -> anyhow::Result<(PoolReport, Vec<Candidate>)> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let (family, white) = FAMILIES[fi];
    let n = white.len() + 1;
    let t0 = Instant::now();
    let next_first = AtomicUsize::new(0);
    type Classes = HashMap<String, (Option<u8>, bool, String)>;
    let parts: Vec<(u64, u64, Classes)> = std::thread::scope(|scope| {
        let hs: Vec<_> = (0..threads.max(1))
            .map(|_| {
                scope.spawn(|| {
                    let mut solver = MateSolver::new();
                    let (mut raw, mut live) = (0u64, 0u64);
                    let mut classes: Classes = HashMap::new();
                    loop {
                        let first = next_first.fetch_add(1, Ordering::Relaxed);
                        if first >= 64 {
                            break;
                        }
                        let mut sq = vec![0usize; n];
                        sq[0] = first; // the white king's square
                        // Depth-first over the remaining distinct squares.
                        let mut stack = vec![0usize; n];
                        let mut level = 1usize;
                        stack[1] = 0;
                        loop {
                            if level == n {
                                // complete placement
                                raw += 1;
                                if !kings_adjacent(sq[0], sq[n - 1]) {
                                    let fen = fen_for(white, &sq);
                                    if let Ok(state) = GameState::from_fen(&fen)
                                        && !state.is_terminal()
                                        && !state.legal_actions().is_empty()
                                    {
                                        live += 1;
                                        let canon = canonical_key(&fen);
                                        if let std::collections::hash_map::Entry::Vacant(slot) =
                                            classes.entry(canon)
                                        {
                                            if solver.table_size() > 3_000_000 {
                                                solver = MateSolver::new();
                                            }
                                            let depth = solver.mate_depth(state.board(), max_depth);
                                            let mut eligible = false;
                                            if let Some(d) = depth {
                                                let moves = legal_cozy_moves(&state);
                                                let correct =
                                                    solver.correct_moves(state.board(), d, &moves);
                                                eligible = d == 1
                                                    || (correct.len() as f32
                                                        <= MAX_CORRECT_FRACTION
                                                            * moves.len() as f32
                                                        && root_fact_ambiguous(&state, &correct));
                                            }
                                            let keep =
                                                if eligible { fen.clone() } else { String::new() };
                                            slot.insert((depth, eligible, String::new()));
                                        }
                                    }
                                }
                                level -= 1;
                                stack[level] += 1;
                                continue;
                            }
                            // next candidate square for `level`
                            let mut s = stack[level];
                            while s < 64 && sq[..level].contains(&s) {
                                s += 1;
                            }
                            if s >= 64 {
                                if level == 1 {
                                    break;
                                }
                                level -= 1;
                                stack[level] += 1;
                                continue;
                            }
                            sq[level] = s;
                            stack[level] = s;
                            level += 1;
                            if level < n {
                                stack[level] = 0;
                            }
                        }
                    }
                    (raw, live, classes)
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("pool worker panicked"))
            .collect()
    });
    let mut rep = PoolReport {
        family: family.to_string(),
        ..Default::default()
    };
    let mut merged: HashMap<String, (Option<u8>, bool, String)> = HashMap::new();
    for (raw, live, classes) in parts {
        rep.raw_placements += raw;
        rep.legal_live += live;
        merged.extend(classes);
    }
    rep.canonical_classes = merged.len() as u64;
    for (depth, eligible, _) in merged.values() {
        let label = depth.map_or("none".to_string(), |d| d.to_string());
        *rep.classes_by_depth.entry(label.clone()).or_insert(0) += 1;
        if *eligible {
            *rep.eligible_by_depth.entry(label).or_insert(0) += 1;
        }
    }
    let mut cands: Vec<Candidate> = merged
        .into_iter()
        .filter_map(|(canon, (depth, eligible, _))| match (depth, eligible) {
            (Some(depth), true) => Some(Candidate {
                fen: fen_from_canon(&canon),
                canon,
                depth,
            }),
            _ => None,
        })
        .collect();
    cands.sort_by(|a, b| a.canon.cmp(&b.canon));
    rep.wall_s = t0.elapsed().as_secs_f64();
    Ok((rep, cands))
}

/// Split sizes for one (family, depth) cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CellPlan {
    pub train: usize,
    pub tune: usize,
    pub confirm: usize,
}

/// The pre-registered scale rule (docs/WORKSTATION_V25_EXPERIMENTS.md, E-DATA-1).
///
/// A cell whose eligible pool `pool` holds at least `train + tune + confirm`
/// targets uses exactly those targets. A smaller cell splits its whole pool
/// 80/10/10: TUNE = CONFIRM = floor(pool / 10), TRAIN = the rest. Filters are
/// never relaxed to grow a pool.
pub fn plan_cell(pool: usize, train: usize, tune: usize, confirm: usize) -> CellPlan {
    if pool >= train + tune + confirm {
        CellPlan {
            train,
            tune,
            confirm,
        }
    } else {
        let eval = pool / 10;
        CellPlan {
            train: pool - 2 * eval,
            tune: eval,
            confirm: eval,
        }
    }
}

/// Deterministic seeded Fisher-Yates over `items`.
fn shuffled<T: Clone>(items: &[T], seed: u64) -> Vec<T> {
    let mut v = items.to_vec();
    let mut rng = Rng(seed);
    for i in (1..v.len()).rev() {
        let j = rng.below(i as u64 + 1) as usize;
        v.swap(i, j);
    }
    v
}

/// Build the labelled positions of `chosen` for one split.
fn label_positions(
    chosen: &[Candidate],
    family: &str,
    split: Split,
    seed: u64,
    counter: &mut usize,
) -> anyhow::Result<Vec<ProofPosition>> {
    let mut solver = MateSolver::new();
    let mut out = Vec::with_capacity(chosen.len());
    for c in chosen {
        let state = GameState::from_fen(&c.fen)?;
        if solver.table_size() > 3_000_000 {
            solver = MateSolver::new();
        }
        let d = solver
            .mate_depth(state.board(), c.depth)
            .filter(|d| *d == c.depth)
            .ok_or_else(|| anyhow::anyhow!("{}: pool depth {} not reproduced", c.fen, c.depth))?;
        let moves = legal_cozy_moves(&state);
        let correct = solver.correct_moves(state.board(), d, &moves);
        let legal: Vec<u16> = state
            .legal_actions()
            .iter()
            .map(|a| a.index() as u16)
            .collect();
        out.push(ProofPosition {
            id: format!(
                "{}-{}-m{d}-{:05}",
                split.label(),
                family.to_lowercase(),
                *counter
            ),
            fen: c.fen.clone(),
            split,
            family: family.to_string(),
            mate_depth: d,
            canon: c.canon.clone(),
            chance_top1: correct.len() as f32 / legal.len() as f32,
            legal,
            correct: correct.iter().map(|&i| i as u32).collect(),
            generator_seed: seed,
        });
        *counter += 1;
    }
    Ok(out)
}

/// Targets and seeds for [`generate_exhaustive`].
#[derive(Debug, Clone)]
pub struct ExhaustiveSpec {
    pub depths: Vec<u8>,
    pub train: usize,
    pub tune: usize,
    pub confirm: usize,
    pub seed_train: u64,
    pub seed_tune: u64,
    pub seed_confirm: u64,
    pub threads: usize,
}

/// Result of exhaustive generation: the three splits, the per-cell plan and the
/// exact pool reports.
pub struct ExhaustiveOutput {
    pub train: ProofTargets,
    pub tune: ProofTargets,
    pub confirm: ProofTargets,
    pub plan: Vec<(String, u8, usize, CellPlan)>,
    pub pools: Vec<PoolReport>,
}

/// Generate TRAIN/TUNE/CONFIRM from the exact pools under the pre-registered
/// scale rule. CONFIRM is drawn first (by `seed_confirm`), then TUNE from the
/// remainder (by `seed_tune`), then TRAIN from the remainder (by `seed_train`),
/// each by an independent seeded shuffle of the canonical pool, so the splits
/// are disjoint by exact FEN and canonical class by construction.
pub fn generate_exhaustive(spec: &ExhaustiveSpec) -> anyhow::Result<ExhaustiveOutput> {
    anyhow::ensure!(
        spec.seed_train != spec.seed_tune
            && spec.seed_tune != spec.seed_confirm
            && spec.seed_train != spec.seed_confirm,
        "split seeds must be independent"
    );
    let max_depth = *spec.depths.iter().max().unwrap_or(&1);
    let (mut tr, mut tu, mut co) = (Vec::new(), Vec::new(), Vec::new());
    let (mut c_tr, mut c_tu, mut c_co) = (0usize, 0usize, 0usize);
    let mut plan = Vec::new();
    let mut pools = Vec::new();
    for (fi, (family, _)) in FAMILIES.iter().enumerate() {
        let (report, cands) = enumerate_pool(fi, max_depth, spec.threads)?;
        pools.push(report);
        for &d in &spec.depths {
            let cell: Vec<Candidate> = cands.iter().filter(|c| c.depth == d).cloned().collect();
            let cp = plan_cell(cell.len(), spec.train, spec.tune, spec.confirm);
            plan.push((family.to_string(), d, cell.len(), cp));
            let tag = mix(fi as u64 * 16 + d as u64);
            let take = |from: &[Candidate], seed: u64, n: usize| -> Vec<Candidate> {
                shuffled(from, mix(seed ^ tag))
                    .into_iter()
                    .take(n)
                    .collect()
            };
            let confirm = take(&cell, spec.seed_confirm, cp.confirm);
            let used: HashSet<&str> = confirm.iter().map(|c| c.canon.as_str()).collect();
            let rest: Vec<Candidate> = cell
                .iter()
                .filter(|c| !used.contains(c.canon.as_str()))
                .cloned()
                .collect();
            let tune = take(&rest, spec.seed_tune, cp.tune);
            let used2: HashSet<&str> = tune.iter().map(|c| c.canon.as_str()).collect();
            let rest2: Vec<Candidate> = rest
                .iter()
                .filter(|c| !used2.contains(c.canon.as_str()))
                .cloned()
                .collect();
            let train = take(&rest2, spec.seed_train, cp.train);
            anyhow::ensure!(
                confirm.len() == cp.confirm && tune.len() == cp.tune && train.len() == cp.train,
                "{family} M{d}: pool {} cannot supply the plan {cp:?}",
                cell.len()
            );
            co.extend(label_positions(
                &confirm,
                family,
                Split::Confirm,
                spec.seed_confirm,
                &mut c_co,
            )?);
            tu.extend(label_positions(
                &tune,
                family,
                Split::Tune,
                spec.seed_tune,
                &mut c_tu,
            )?);
            tr.extend(label_positions(
                &train,
                family,
                Split::Train,
                spec.seed_train,
                &mut c_tr,
            )?);
        }
    }
    let filters = |split: Split| {
        serde_json::json!({
            "source": "exhaustive exact pool, independent seeded shuffle per split",
            "scale_rule": "pool >= train+tune+confirm targets: use targets; else 80/10/10 of the pool",
            "targets": { "train": spec.train, "tune": spec.tune, "confirm": spec.confirm },
            "max_correct_fraction": MAX_CORRECT_FRACTION,
            "fact_ambiguity_required_for_depth_at_least": 2,
            "depths": spec.depths,
            "split": split.label(),
        })
    };
    Ok(ExhaustiveOutput {
        train: ProofTargets::new(Split::Train, spec.seed_train, filters(Split::Train), tr),
        tune: ProofTargets::new(Split::Tune, spec.seed_tune, filters(Split::Tune), tu),
        confirm: ProofTargets::new(
            Split::Confirm,
            spec.seed_confirm,
            filters(Split::Confirm),
            co,
        ),
        plan,
        pools,
    })
}
