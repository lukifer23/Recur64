//! The evaluator boundary carries `CandidateFactsV1` only to evaluators that
//! ask for it, aligned to the legal-action list, on both the single-leaf and the
//! multi-leaf (batched) paths.

use std::sync::Mutex;

use recur64_core::{ActionId, GameState, candidate_facts};
use recur64_search::evaluator::{EvalError, EvalRequest, EvalResult, Evaluator};
use recur64_search::game_tree::ChessGame;
use recur64_search::puct::PuctGame;

struct Probe {
    wants: bool,
    /// (legal len, facts len or None, facts equal to the authoritative rows)
    seen: Mutex<Vec<(usize, Option<usize>, bool)>>,
    expected: Mutex<Vec<Vec<[f32; 8]>>>,
}

impl Probe {
    fn new(wants: bool) -> Self {
        Self {
            wants,
            seen: Mutex::new(Vec::new()),
            expected: Mutex::new(Vec::new()),
        }
    }
}

impl Evaluator for Probe {
    fn needs_candidate_facts(&self) -> bool {
        self.wants
    }
    fn evaluate(&self, request: EvalRequest<'_>) -> Result<EvalResult, EvalError> {
        let mut seen = self.seen.lock().unwrap();
        let exp = self.expected.lock().unwrap();
        let idx = seen.len();
        let equal = match request.facts {
            Some(f) => exp.get(idx).is_some_and(|e| e.as_slice() == f),
            None => false,
        };
        seen.push((request.legal.len(), request.facts.map(<[_]>::len), equal));
        Ok(EvalResult::uniform(request.legal.len(), 0.0))
    }
}

fn states() -> Vec<GameState> {
    let mut a = GameState::startpos();
    a.apply_uci("e2e4").unwrap();
    vec![
        GameState::startpos(),
        a,
        GameState::from_fen("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1").unwrap(),
    ]
}

#[test]
fn facts_are_supplied_aligned_when_requested_on_both_paths() {
    let probe = Probe::new(true);
    let sts = states();
    *probe.expected.lock().unwrap() = sts.iter().map(candidate_facts).collect();

    // Single-leaf path, one state.
    let g = ChessGame::new(sts[0].clone(), &probe);
    let legal = sts[0].legal_actions();
    g.evaluate(&legal).unwrap();
    // Multi-leaf path, all three states in one submission.
    probe.seen.lock().unwrap().clear();
    let games: Vec<ChessGame> = sts
        .iter()
        .map(|s| ChessGame::new(s.clone(), &probe))
        .collect();
    let refs: Vec<&ChessGame> = games.iter().collect();
    let legals: Vec<Vec<ActionId>> = sts.iter().map(GameState::legal_actions).collect();
    for r in ChessGame::evaluate_many(&refs, &legals) {
        r.unwrap();
    }
    let seen = probe.seen.lock().unwrap();
    assert_eq!(seen.len(), 3);
    for (i, (legal_n, facts_n, equal)) in seen.iter().enumerate() {
        assert_eq!(
            *facts_n,
            Some(*legal_n),
            "state {i}: facts rows == legal rows"
        );
        assert!(equal, "state {i}: facts equal the authoritative rows");
    }
}

#[test]
fn legacy_evaluators_receive_no_facts() {
    let probe = Probe::new(false);
    let sts = states();
    let games: Vec<ChessGame> = sts
        .iter()
        .map(|s| ChessGame::new(s.clone(), &probe))
        .collect();
    let refs: Vec<&ChessGame> = games.iter().collect();
    let legals: Vec<Vec<ActionId>> = sts.iter().map(GameState::legal_actions).collect();
    for r in ChessGame::evaluate_many(&refs, &legals) {
        r.unwrap();
    }
    games[0].evaluate(&legals[0]).unwrap();
    assert!(
        probe
            .seen
            .lock()
            .unwrap()
            .iter()
            .all(|(_, f, _)| f.is_none())
    );
}
