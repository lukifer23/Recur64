//! PUCT control search, generic over a small game abstraction.
//!
//! The search is deliberately independent of chess and of Burn. It operates on
//! [`PuctGame`], which supplies terminal values, legal actions, transitions, and
//! (for non-terminal positions) an evaluation. This makes the algorithm testable
//! on exact synthetic trees before it is ever pointed at real chess.
//!
//! # Value perspective
//!
//! Every value is from the **side-to-move perspective** of the position it is
//! attached to. A child's value is negated before being added to the parent
//! edge, because the child's side to move is the parent's opponent.
//!
//! # Budget
//!
//! [`search`] performs exactly `cfg.simulations` traversals for a non-terminal
//! root. Traversal 0 expands the root (one neural evaluation); each later
//! traversal descends to a leaf, expanding it (one neural evaluation) or
//! stopping at a terminal leaf (no neural evaluation). A terminal root performs
//! zero traversals and produces no policy target.
//!
//! # MCTS-solver (D50, optional)
//!
//! With `PuctConfig::solver`, exact game-theoretic results propagate up the
//! tree: a node is a proven win when some move leads to a proven loss for the
//! opponent, and is proven once every move is proven (a draw if any move
//! draws, else a loss). A proven node is never expanded further; its exact
//! value is backed up like a terminal. Selection always takes a proven
//! winning move (the shortest) and never takes a proven losing move while an
//! alternative exists. Proofs come from the rules profile's terminals only,
//! never from the network, and are exact for the path-dependent tree (the
//! game state carries its repetition history).

use crate::evaluator::{EvalError, EvalResult};

/// A game the search can traverse. Values are side-to-move perspective.
pub trait PuctGame: Sized {
    type Action: Copy + Eq + Ord + std::fmt::Debug;

    /// Terminal value from the side-to-move perspective, or `None` if ongoing.
    fn terminal_value(&self) -> Option<f32>;
    /// Legal actions in a deterministic (sorted) order.
    fn legal_actions(&self) -> Vec<Self::Action>;
    /// Apply an action, producing the child state.
    fn apply(&self, action: Self::Action) -> Self;
    /// Evaluate a non-terminal position. `policy` must align to `legal`.
    fn evaluate(&self, legal: &[Self::Action]) -> Result<EvalResult, EvalError>;
    /// Evaluate several positions of one search tree together (multi-leaf
    /// search, D47). Results are in input order. Default: one by one.
    fn evaluate_many(
        games: &[&Self],
        legal: &[Vec<Self::Action>],
    ) -> Vec<Result<EvalResult, EvalError>> {
        games
            .iter()
            .zip(legal)
            .map(|(g, l)| g.evaluate(l))
            .collect()
    }
}

/// Search configuration. `c_puct` is explicit and configurable; the Phase 2
/// default is a pilot value, not a tuned constant.
#[derive(Debug, Clone, Copy)]
pub struct PuctConfig {
    pub c_puct: f32,
    pub simulations: u32,
    /// Leaves selected per round with virtual loss and evaluated together
    /// (D47). `1` is the original one-leaf-at-a-time search, unchanged.
    pub leaves_in_flight: u32,
    /// Propagate proven results (D50 MCTS-solver). `false` is the original
    /// search, unchanged.
    pub solver: bool,
}

/// An exact game-theoretic result, from the side-to-move perspective of the
/// node it is attached to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Proof {
    Win,
    Draw,
    Loss,
}

impl Proof {
    fn from_terminal(value: f32) -> Self {
        if value > 0.0 {
            Proof::Win
        } else if value < 0.0 {
            Proof::Loss
        } else {
            Proof::Draw
        }
    }

    /// The exact value this result backs up.
    pub fn value(self) -> f32 {
        match self {
            Proof::Win => 1.0,
            Proof::Draw => 0.0,
            Proof::Loss => -1.0,
        }
    }

    /// The same result seen by the opponent.
    pub fn flip(self) -> Self {
        match self {
            Proof::Win => Proof::Loss,
            Proof::Draw => Proof::Draw,
            Proof::Loss => Proof::Win,
        }
    }
}

impl Default for PuctConfig {
    fn default() -> Self {
        Self {
            c_puct: 1.0,
            simulations: 16,
            leaves_in_flight: 1,
            solver: false,
        }
    }
}

/// One root edge in the search result.
#[derive(Debug, Clone)]
pub struct RootEdge<A> {
    pub action: A,
    pub prior: f32,
    pub visits: u32,
    /// Proven result of playing this move, for the root's side to move
    /// (D50; always `None` without the solver).
    pub proof: Option<Proof>,
    /// Plies to the proving terminal along the proof, counting this move.
    pub proof_plies: u32,
}

/// Result of a search.
#[derive(Debug, Clone)]
pub struct SearchResult<A> {
    pub edges: Vec<RootEdge<A>>,
    pub total_visits: u32,
    /// Number of traversals actually performed.
    pub traversals: u32,
    /// Mean value estimate from the root's side-to-move perspective.
    pub root_value: f32,
    /// The network's own value at the root (before search), side-to-move
    /// perspective; 0 for terminal/empty roots.
    pub root_network_value: f32,
    /// Proven result of the root (D50; always `None` without the solver).
    pub root_proof: Option<Proof>,
}

impl<A: Copy + Ord> SearchResult<A> {
    /// Visit distribution over root edges. Falls back to priors when there were
    /// no visits (e.g. `simulations <= 1`), and to uniform if priors are
    /// degenerate. Empty only when the root had no legal moves.
    pub fn policy(&self) -> Vec<f32> {
        if self.edges.is_empty() {
            return Vec::new();
        }
        if self.total_visits > 0 {
            let total = self.total_visits as f32;
            return self.edges.iter().map(|e| e.visits as f32 / total).collect();
        }
        let prior_sum: f32 = self.edges.iter().map(|e| e.prior).sum();
        if prior_sum > 0.0 {
            self.edges.iter().map(|e| e.prior / prior_sum).collect()
        } else {
            vec![1.0 / self.edges.len() as f32; self.edges.len()]
        }
    }

    /// Highest-visit action; ties broken by lower action id. `None` if empty.
    pub fn best_action(&self) -> Option<A> {
        self.edges
            .iter()
            .max_by(|a, b| {
                a.visits
                    .cmp(&b.visits)
                    .then_with(|| b.action.cmp(&a.action)) // lower action wins ties
            })
            .map(|e| e.action)
    }
}

struct Edge<G: PuctGame> {
    action: G::Action,
    prior: f32,
    n: u32,
    w: f32,
    child: Option<Box<Node<G>>>,
}

struct Node<G: PuctGame> {
    game: G,
    terminal: Option<f32>,
    expanded: bool,
    edges: Vec<Edge<G>>,
    visits_total: u32,
    eval_value: f32,
    /// Exact result (terminals always; interior nodes only via the solver).
    proof: Option<Proof>,
    /// Plies from this node to the proving terminal.
    proof_plies: u32,
}

impl<G: PuctGame> Node<G> {
    fn new(game: G) -> Self {
        let terminal = game.terminal_value();
        Self {
            game,
            terminal,
            expanded: false,
            edges: Vec::new(),
            visits_total: 0,
            eval_value: 0.0,
            proof: terminal.map(Proof::from_terminal),
            proof_plies: 0,
        }
    }

    /// The value a traversal reaching this node backs up without expanding
    /// it: the terminal value, or (with the solver) the proven value.
    fn known_value(&self, cfg: &PuctConfig) -> Option<f32> {
        if cfg.solver {
            self.proof.map(Proof::value)
        } else {
            self.terminal
        }
    }

    /// D50: derive this node's proof from its children's, if now determined.
    fn resolve(&mut self) {
        if self.proof.is_some() || self.edges.is_empty() {
            return;
        }
        let mut win: Option<u32> = None;
        let mut draw: Option<u32> = None;
        let mut loss: Option<u32> = None;
        let mut all_proven = true;
        for e in &self.edges {
            let Some((p, plies)) = edge_proof(e) else {
                all_proven = false;
                continue;
            };
            match p {
                Proof::Win => win = Some(win.map_or(plies, |w| w.min(plies))),
                Proof::Draw => draw = Some(draw.map_or(plies, |d| d.min(plies))),
                // The loser delays the loss as long as possible.
                Proof::Loss => loss = Some(loss.map_or(plies, |l| l.max(plies))),
            }
        }
        let proven = match (win, all_proven, draw, loss) {
            (Some(w), _, _, _) => Some((Proof::Win, w)),
            (None, true, Some(d), _) => Some((Proof::Draw, d)),
            (None, true, None, Some(l)) => Some((Proof::Loss, l)),
            _ => None,
        };
        if let Some((p, plies)) = proven {
            self.proof = Some(p);
            self.proof_plies = plies;
        }
    }

    fn expand(&mut self) -> Result<(), EvalError> {
        let legal = self.game.legal_actions();
        let eval = self.game.evaluate(&legal)?;
        self.install(legal, eval)
    }

    /// Attach an evaluation (priors and value) to this node's legal edges.
    fn install(&mut self, legal: Vec<G::Action>, eval: EvalResult) -> Result<(), EvalError> {
        if eval.policy.len() != legal.len() {
            return Err(EvalError::Invalid(format!(
                "policy len {} != legal len {}",
                eval.policy.len(),
                legal.len()
            )));
        }
        self.edges = legal
            .into_iter()
            .zip(eval.policy)
            .map(|(action, prior)| Edge {
                action,
                prior,
                n: 0,
                w: 0.0,
                child: None,
            })
            .collect();
        self.eval_value = eval.value;
        self.expanded = true;
        Ok(())
    }
}

/// Proven result of taking edge `e`, for the side choosing it, with the
/// plies to the proving terminal (counting the move itself).
fn edge_proof<G: PuctGame>(e: &Edge<G>) -> Option<(Proof, u32)> {
    let child = e.child.as_deref()?;
    child.proof.map(|p| (p.flip(), child.proof_plies + 1))
}

fn select<G: PuctGame>(node: &Node<G>, cfg: &PuctConfig) -> usize {
    // D50: always take the shortest proven win; skip proven losses unless
    // every move loses.
    let mut skip_losses = false;
    if cfg.solver {
        let mut shortest_win: Option<(u32, usize)> = None;
        let mut all_lose = true;
        for (i, e) in node.edges.iter().enumerate() {
            match edge_proof(e) {
                Some((Proof::Win, plies)) if shortest_win.is_none_or(|(w, _)| plies < w) => {
                    shortest_win = Some((plies, i));
                }
                Some((Proof::Win | Proof::Loss, _)) => {}
                _ => all_lose = false,
            }
        }
        if let Some((_, i)) = shortest_win {
            return i;
        }
        skip_losses = !all_lose;
    }
    let sqrt_total = (node.visits_total as f32).sqrt();
    let mut best = usize::MAX;
    let mut best_score = f32::NEG_INFINITY;
    for (i, e) in node.edges.iter().enumerate() {
        if skip_losses && matches!(edge_proof(e), Some((Proof::Loss, _))) {
            continue;
        }
        let q = if e.n > 0 { e.w / e.n as f32 } else { 0.0 };
        let u = cfg.c_puct * e.prior * sqrt_total / (1.0 + e.n as f32);
        let score = q + u;
        let better = best == usize::MAX
            || score > best_score
            || (score == best_score && e.prior > node.edges[best].prior);
        if better {
            best = i;
            best_score = score;
        }
    }
    best
}

/// One simulation below an expanded, non-terminal node. Returns the value from
/// this node's side-to-move perspective.
///
/// Exactly one new node is expanded per call: if the selected edge has no child,
/// that child is created, evaluated (if non-terminal), and its value returned;
/// otherwise the simulation descends into the existing child and backs its value
/// up on the way out.
fn simulate<G: PuctGame>(node: &mut Node<G>, cfg: &PuctConfig) -> Result<f32, EvalError> {
    if node.edges.is_empty() {
        return Ok(0.0);
    }
    let idx = select(node, cfg);

    let child_value = if node.edges[idx].child.is_none() {
        let child_game = node.game.apply(node.edges[idx].action);
        let mut child = Node::new(child_game);
        let value = match child.known_value(cfg) {
            Some(t) => t,
            None => {
                child.expand()?;
                child.eval_value
            }
        };
        node.edges[idx].child = Some(Box::new(child));
        value
    } else {
        let child = node.edges[idx]
            .child
            .as_mut()
            .expect("checked is_some above");
        match child.known_value(cfg) {
            Some(t) => t,
            None => simulate(child, cfg)?,
        }
    };
    if cfg.solver {
        node.resolve();
    }

    // Child value is from the child's perspective; from this node's perspective
    // it is negated.
    let edge_value = -child_value;
    node.edges[idx].w += edge_value;
    node.edges[idx].n += 1;
    node.visits_total += 1;
    Ok(edge_value)
}

/// Virtual loss per pending traversal (D47): each edge on a pending path
/// counts one provisional visit with value -1 for the side choosing it, so
/// later selections in the same round prefer other lines. Removed on backup.
const VIRTUAL_LOSS: f32 = 1.0;

/// How one selected path ended.
enum PathEnd<G: PuctGame> {
    /// Known value, from the perspective of the node at the end of the path.
    Value(f32),
    /// A new non-terminal leaf to evaluate.
    Leaf(G),
    /// The selected edge is already pending in this round.
    Collision,
}

/// The outcome of one selection in a multi-leaf round.
enum Descent<G: PuctGame> {
    /// A new leaf to evaluate; its path carries virtual loss.
    Leaf(Vec<usize>, G),
    /// The traversal reached a known value and was already backed up.
    Resolved,
    /// Collision with a pending leaf; the partial path was reverted.
    Collision,
}

/// Select one path from `root`, applying virtual loss to every chosen edge.
fn descend<G: PuctGame>(root: &mut Node<G>, cfg: &PuctConfig) -> Descent<G> {
    let mut path = Vec::new();
    let end = {
        let mut node: &mut Node<G> = root;
        loop {
            if node.edges.is_empty() {
                break PathEnd::Value(0.0);
            }
            let idx = select(node, cfg);
            // A visited edge without a child can only be a leaf that is
            // pending in this round (sequential search always creates it).
            if node.edges[idx].child.is_none() && node.edges[idx].n > 0 {
                break PathEnd::Collision;
            }
            node.edges[idx].n += 1;
            node.edges[idx].w -= VIRTUAL_LOSS;
            node.visits_total += 1;
            path.push(idx);
            if node.edges[idx].child.is_none() {
                let child = Node::new(node.game.apply(node.edges[idx].action));
                match child.known_value(cfg) {
                    Some(t) => {
                        node.edges[idx].child = Some(Box::new(child));
                        break PathEnd::Value(t);
                    }
                    None => break PathEnd::Leaf(child.game),
                }
            }
            let child = node.edges[idx].child.as_deref_mut().expect("checked");
            if let Some(t) = child.known_value(cfg) {
                break PathEnd::Value(t);
            }
            node = child;
        }
    };
    match end {
        PathEnd::Value(v) => {
            backup(root, &path, v);
            if cfg.solver {
                propagate(root, &path);
            }
            Descent::Resolved
        }
        PathEnd::Leaf(game) => Descent::Leaf(path, game),
        PathEnd::Collision => {
            revert(root, &path);
            Descent::Collision
        }
    }
}

/// Back up `leaf_value` (from the perspective of the node at the end of
/// `path`) along `path`, turning each virtual visit into a real one.
fn backup<G: PuctGame>(root: &mut Node<G>, path: &[usize], leaf_value: f32) {
    let len = path.len();
    let mut node: &mut Node<G> = root;
    for (d, &idx) in path.iter().enumerate() {
        // Each ply flips the perspective: an edge value is the negated value
        // of the child it leads to.
        let edge_value = if (len - d) % 2 == 1 {
            -leaf_value
        } else {
            leaf_value
        };
        node.edges[idx].w += edge_value + VIRTUAL_LOSS;
        if d + 1 < len {
            node = node.edges[idx]
                .child
                .as_deref_mut()
                .expect("path child exists");
        }
    }
}

/// D50: re-derive proofs bottom-up along `path` after a known value reached
/// its end (only a proven end can prove its ancestors).
fn propagate<G: PuctGame>(node: &mut Node<G>, path: &[usize]) {
    if let Some((&idx, rest)) = path.split_first() {
        if let Some(child) = node.edges[idx].child.as_deref_mut() {
            propagate(child, rest);
        }
        node.resolve();
    }
}

/// Undo the virtual visits of a partial path (collision).
fn revert<G: PuctGame>(root: &mut Node<G>, path: &[usize]) {
    let mut node: &mut Node<G> = root;
    for (d, &idx) in path.iter().enumerate() {
        node.edges[idx].n -= 1;
        node.edges[idx].w += VIRTUAL_LOSS;
        node.visits_total -= 1;
        if d + 1 < path.len() {
            node = node.edges[idx]
                .child
                .as_deref_mut()
                .expect("path child exists");
        }
    }
}

/// Attach an evaluated leaf at the end of `path`.
fn insert<G: PuctGame>(root: &mut Node<G>, path: &[usize], leaf: Node<G>) {
    let (last, prefix) = path.split_last().expect("non-empty leaf path");
    let mut node: &mut Node<G> = root;
    for &idx in prefix {
        node = node.edges[idx]
            .child
            .as_deref_mut()
            .expect("path child exists");
    }
    node.edges[*last].child = Some(Box::new(leaf));
}

/// Multi-leaf search (D47): repeatedly select up to `leaves_in_flight`
/// leaves with virtual loss, evaluate them in one submission, then expand and
/// back up. Performs exactly `simulations - 1` traversals after the root
/// expansion, like the sequential search.
fn search_rounds<G: PuctGame>(root: &mut Node<G>, cfg: &PuctConfig) -> Result<(), EvalError> {
    let mut done = 1u32;
    while done < cfg.simulations {
        let mut leaves: Vec<(Vec<usize>, G)> = Vec::new();
        while (leaves.len() as u32) < cfg.leaves_in_flight
            && done + (leaves.len() as u32) < cfg.simulations
        {
            match descend(root, cfg) {
                Descent::Leaf(path, game) => leaves.push((path, game)),
                Descent::Resolved => done += 1,
                Descent::Collision => break,
            }
        }
        if leaves.is_empty() {
            continue;
        }
        let legal: Vec<Vec<G::Action>> = leaves.iter().map(|(_, g)| g.legal_actions()).collect();
        let evals = {
            let games: Vec<&G> = leaves.iter().map(|(_, g)| g).collect();
            G::evaluate_many(&games, &legal)
        };
        for (((path, game), legal), eval) in leaves.into_iter().zip(legal).zip(evals) {
            let mut leaf = Node::new(game);
            leaf.install(legal, eval?)?;
            let value = leaf.eval_value;
            insert(root, &path, leaf);
            backup(root, &path, value);
            done += 1;
        }
    }
    Ok(())
}

/// Exploration noise mixed into the root priors only:
/// `prior' = (1 - epsilon) * prior + epsilon * noise[i]`, with `noise` aligned
/// to the root's legal actions (their deterministic order).
#[derive(Debug, Clone)]
pub struct RootNoise {
    pub epsilon: f32,
    pub noise: Vec<f32>,
}

/// Run PUCT from `root`. Performs exactly `cfg.simulations` traversals for a
/// non-terminal root; zero for a terminal root.
pub fn search<G: PuctGame>(
    root: G,
    cfg: &PuctConfig,
) -> Result<SearchResult<G::Action>, EvalError> {
    search_with_root_noise(root, cfg, None)
}

/// [`search`] with optional root exploration noise (self-play only). The
/// reported `RootEdge::prior` is the mixed prior the search used.
pub fn search_with_root_noise<G: PuctGame>(
    root: G,
    cfg: &PuctConfig,
    root_noise: Option<&RootNoise>,
) -> Result<SearchResult<G::Action>, EvalError> {
    let mut node = Node::new(root);
    if let Some(t) = node.terminal {
        return Ok(SearchResult {
            edges: Vec::new(),
            total_visits: 0,
            traversals: 0,
            root_value: t,
            root_network_value: 0.0,
            root_proof: None,
        });
    }
    if cfg.simulations == 0 {
        return Ok(SearchResult {
            edges: Vec::new(),
            total_visits: 0,
            traversals: 0,
            root_value: 0.0,
            root_network_value: 0.0,
            root_proof: None,
        });
    }

    node.expand()?; // traversal 0
    if let Some(n) = root_noise {
        if n.noise.len() != node.edges.len() {
            return Err(EvalError::Invalid(format!(
                "root noise len {} != root edges {}",
                n.noise.len(),
                node.edges.len()
            )));
        }
        for (e, x) in node.edges.iter_mut().zip(&n.noise) {
            e.prior = (1.0 - n.epsilon) * e.prior + n.epsilon * x;
        }
    }
    if cfg.leaves_in_flight <= 1 {
        for _ in 1..cfg.simulations {
            simulate(&mut node, cfg)?;
        }
    } else {
        search_rounds(&mut node, cfg)?;
    }

    let total = node.visits_total;
    let root_value = if total > 0 {
        node.edges.iter().map(|e| e.w).sum::<f32>() / total as f32
    } else {
        node.eval_value
    };
    let edges = node
        .edges
        .iter()
        .map(|e| {
            let proven = if cfg.solver { edge_proof(e) } else { None };
            RootEdge {
                action: e.action,
                prior: e.prior,
                visits: e.n,
                proof: proven.map(|(p, _)| p),
                proof_plies: proven.map_or(0, |(_, n)| n),
            }
        })
        .collect();

    Ok(SearchResult {
        edges,
        total_visits: total,
        traversals: cfg.simulations,
        root_value,
        root_network_value: node.eval_value,
        root_proof: if cfg.solver { node.proof } else { None },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    /// A synthetic mathematical tree. Action = index into `children`.
    #[derive(Clone)]
    struct TreeNode {
        children: Vec<usize>,
        terminal: Option<f32>,
        value: f32,
        policy: Option<Vec<f32>>,
    }

    #[derive(Clone)]
    struct TreeGame {
        nodes: Rc<Vec<TreeNode>>,
        index: usize,
        calls: Rc<Cell<u32>>,
    }

    impl TreeGame {
        fn new(nodes: Vec<TreeNode>) -> Self {
            Self {
                nodes: Rc::new(nodes),
                index: 0,
                calls: Rc::new(Cell::new(0)),
            }
        }
        fn node(&self) -> &TreeNode {
            &self.nodes[self.index]
        }
    }

    impl PuctGame for TreeGame {
        type Action = usize;
        fn terminal_value(&self) -> Option<f32> {
            self.node().terminal
        }
        fn legal_actions(&self) -> Vec<usize> {
            (0..self.node().children.len()).collect()
        }
        fn apply(&self, action: usize) -> Self {
            Self {
                nodes: self.nodes.clone(),
                index: self.node().children[action],
                calls: self.calls.clone(),
            }
        }
        fn evaluate(&self, _legal: &[usize]) -> Result<EvalResult, EvalError> {
            self.calls.set(self.calls.get() + 1);
            let n = self.node().children.len();
            let policy = self
                .node()
                .policy
                .clone()
                .unwrap_or_else(|| vec![1.0 / n as f32; n]);
            Ok(EvalResult {
                policy,
                value: self.node().value,
                wdl: crate::evaluator::value_to_wdl(self.node().value),
            })
        }
    }

    fn leaf() -> TreeNode {
        TreeNode {
            children: vec![],
            terminal: Some(0.0),
            value: 0.0,
            policy: None,
        }
    }

    #[test]
    fn one_legal_move_target_is_that_move() {
        // Root -> one child (non-terminal leaf).
        let nodes = vec![
            TreeNode {
                children: vec![1],
                terminal: None,
                value: 0.0,
                policy: None,
            },
            TreeNode {
                children: vec![],
                terminal: Some(0.0),
                value: 0.0,
                policy: None,
            },
        ];
        let g = TreeGame::new(nodes);
        let r = search(
            g,
            &PuctConfig {
                c_puct: 1.0,
                simulations: 8,
                leaves_in_flight: 1,
                solver: false,
            },
        )
        .unwrap();
        assert_eq!(r.edges.len(), 1);
        assert_eq!(r.total_visits, 7); // 8 traversals - 1 root expansion
        let p = r.policy();
        assert_eq!(p.len(), 1);
        assert!((p[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn higher_prior_gets_more_visits_when_values_equal() {
        // Two terminal children with equal values but unequal priors.
        let nodes = vec![
            TreeNode {
                children: vec![1, 2],
                terminal: None,
                value: 0.0,
                policy: Some(vec![0.8, 0.2]),
            },
            leaf(),
            leaf(),
        ];
        let g = TreeGame::new(nodes);
        let r = search(
            g,
            &PuctConfig {
                c_puct: 1.0,
                simulations: 32,
                leaves_in_flight: 1,
                solver: false,
            },
        )
        .unwrap();
        assert!(r.edges[0].visits > r.edges[1].visits);
    }

    #[test]
    fn root_noise_mixes_root_priors_only_and_checks_length() {
        let nodes = vec![
            TreeNode {
                children: vec![1, 2],
                terminal: None,
                value: 0.0,
                policy: Some(vec![0.8, 0.2]),
            },
            leaf(),
            leaf(),
        ];
        let cfg = PuctConfig {
            c_puct: 1.0,
            simulations: 32,
            leaves_in_flight: 1,
            solver: false,
        };
        let noise = RootNoise {
            epsilon: 0.25,
            noise: vec![0.0, 1.0],
        };
        let r = search_with_root_noise(TreeGame::new(nodes.clone()), &cfg, Some(&noise)).unwrap();
        // (1 - 0.25) * 0.8 + 0.25 * 0 = 0.6; (1 - 0.25) * 0.2 + 0.25 * 1 = 0.4
        assert!((r.edges[0].prior - 0.6).abs() < 1e-6);
        assert!((r.edges[1].prior - 0.4).abs() < 1e-6);
        let clean = search(TreeGame::new(nodes.clone()), &cfg).unwrap();
        assert!(
            (clean.edges[0].prior - 0.8).abs() < 1e-6,
            "search() stays noise-free"
        );
        let bad = RootNoise {
            epsilon: 0.25,
            noise: vec![1.0],
        };
        assert!(search_with_root_noise(TreeGame::new(nodes), &cfg, Some(&bad)).is_err());
    }

    #[test]
    fn value_sign_inversion_across_plies() {
        // The child is terminal with value +1 from the *child's* perspective, so
        // the root edge must accumulate -(+1) = -1 from the root's perspective.
        let nodes = vec![
            TreeNode {
                children: vec![1],
                terminal: None,
                value: 0.0,
                policy: None,
            },
            TreeNode {
                children: vec![],
                terminal: Some(1.0),
                value: 0.0,
                policy: None,
            },
        ];
        let g = TreeGame::new(nodes);
        let r = search(
            g,
            &PuctConfig {
                c_puct: 1.0,
                simulations: 4,
                leaves_in_flight: 1,
                solver: false,
            },
        )
        .unwrap();
        // Edge value from root perspective must be -1.
        assert!(r.root_value < 0.0, "root_value={}", r.root_value);
        assert!((r.root_value + 1.0).abs() < 1e-6);
    }

    #[test]
    fn terminal_win_and_loss_children() {
        // Child A is a terminal win for the child (+1) => -1 for root.
        // Child B is a terminal loss for the child (-1) => +1 for root.
        let nodes = vec![
            TreeNode {
                children: vec![1, 2],
                terminal: None,
                value: 0.0,
                policy: Some(vec![0.5, 0.5]),
            },
            TreeNode {
                children: vec![],
                terminal: Some(1.0),
                value: 0.0,
                policy: None,
            },
            TreeNode {
                children: vec![],
                terminal: Some(-1.0),
                value: 0.0,
                policy: None,
            },
        ];
        let g = TreeGame::new(nodes);
        let r = search(
            g,
            &PuctConfig {
                c_puct: 1.0,
                simulations: 32,
                leaves_in_flight: 1,
                solver: false,
            },
        )
        .unwrap();
        // The root prefers the child that is bad for the child (loss => +1 root).
        assert_eq!(r.best_action(), Some(1));
        assert!(r.root_value > 0.0);
    }

    #[test]
    fn terminal_root_has_no_target_and_no_eval() {
        let nodes = vec![TreeNode {
            children: vec![],
            terminal: Some(-1.0),
            value: 0.0,
            policy: None,
        }];
        let g = TreeGame::new(nodes);
        let calls = g.calls.clone();
        let r = search(
            g,
            &PuctConfig {
                c_puct: 1.0,
                simulations: 16,
                leaves_in_flight: 1,
                solver: false,
            },
        )
        .unwrap();
        assert!(r.edges.is_empty());
        assert_eq!(r.traversals, 0);
        assert_eq!(calls.get(), 0, "terminal root must not call the evaluator");
        assert_eq!(r.root_value, -1.0);
    }

    #[test]
    fn terminal_leaves_consume_traversals_without_eval() {
        // Root with two terminal children.
        let nodes = vec![
            TreeNode {
                children: vec![1, 2],
                terminal: None,
                value: 0.0,
                policy: Some(vec![0.5, 0.5]),
            },
            leaf(),
            leaf(),
        ];
        let g = TreeGame::new(nodes);
        let calls = g.calls.clone();
        let r = search(
            g,
            &PuctConfig {
                c_puct: 1.0,
                simulations: 10,
                leaves_in_flight: 1,
                solver: false,
            },
        )
        .unwrap();
        assert_eq!(r.traversals, 10);
        // Only the root was expanded (1 eval); terminal children never evaluate.
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn budget_off_by_one_contract() {
        // A deep chain: each traversal expands exactly one new non-terminal node.
        let nodes = vec![
            TreeNode {
                children: vec![1],
                terminal: None,
                value: 0.0,
                policy: None,
            },
            TreeNode {
                children: vec![2],
                terminal: None,
                value: 0.0,
                policy: None,
            },
            TreeNode {
                children: vec![3],
                terminal: None,
                value: 0.0,
                policy: None,
            },
            TreeNode {
                children: vec![4],
                terminal: None,
                value: 0.0,
                policy: None,
            },
            TreeNode {
                children: vec![],
                terminal: Some(0.0),
                value: 0.0,
                policy: None,
            },
        ];
        let g = TreeGame::new(nodes);
        let calls = g.calls.clone();
        let sims = 5u32;
        let r = search(
            g,
            &PuctConfig {
                c_puct: 1.0,
                simulations: sims,
                leaves_in_flight: 1,
                solver: false,
            },
        )
        .unwrap();
        assert_eq!(r.traversals, sims);
        // Root + up to 3 non-terminal nodes expanded; terminal leaf adds none.
        assert!(calls.get() <= sims);
        assert_eq!(calls.get(), 4);
    }

    #[test]
    fn deterministic_tie_breaks_lowest_action() {
        let nodes = vec![
            TreeNode {
                children: vec![1, 2],
                terminal: None,
                value: 0.0,
                policy: Some(vec![0.5, 0.5]),
            },
            leaf(),
            leaf(),
        ];
        let g = TreeGame::new(nodes);
        let r = search(
            g,
            &PuctConfig {
                c_puct: 1.0,
                simulations: 9,
                leaves_in_flight: 1,
                solver: false,
            },
        )
        .unwrap();
        // With identical priors and values, visits alternate; the first selected
        // is the lowest action.
        assert_eq!(r.edges.len(), 2);
        // Deterministic: repeated search gives identical visits.
        let g2 = TreeGame::new(vec![
            TreeNode {
                children: vec![1, 2],
                terminal: None,
                value: 0.0,
                policy: Some(vec![0.5, 0.5]),
            },
            leaf(),
            leaf(),
        ]);
        let r2 = search(
            g2,
            &PuctConfig {
                c_puct: 1.0,
                simulations: 9,
                leaves_in_flight: 1,
                solver: false,
            },
        )
        .unwrap();
        assert_eq!(
            r.edges.iter().map(|e| e.visits).collect::<Vec<_>>(),
            r2.edges.iter().map(|e| e.visits).collect::<Vec<_>>()
        );
    }

    fn inner(children: Vec<usize>, value: f32, policy: Option<Vec<f32>>) -> TreeNode {
        TreeNode {
            children,
            terminal: None,
            value,
            policy,
        }
    }

    fn terminal(value: f32) -> TreeNode {
        TreeNode {
            children: vec![],
            terminal: Some(value),
            value: 0.0,
            policy: None,
        }
    }

    fn solver_cfg(simulations: u32, leaves_in_flight: u32) -> PuctConfig {
        PuctConfig {
            c_puct: 1.0,
            simulations,
            leaves_in_flight,
            solver: true,
        }
    }

    /// Root move 0 walks into a mate in one for the opponent; the network
    /// likes it (prior 0.9, the opponent's node looks bad for the opponent).
    fn trap_tree() -> Vec<TreeNode> {
        vec![
            inner(vec![1, 2], 0.0, Some(vec![0.9, 0.1])),
            // Opponent to move: move 0 mates the root side, move 1 is quiet.
            inner(vec![3, 4], -0.8, Some(vec![0.05, 0.95])),
            inner(vec![5], 0.0, None),
            terminal(-1.0), // root side to move and checkmated
            inner(vec![6], 0.0, None),
            terminal(0.0),
            terminal(0.0),
        ]
    }

    #[test]
    fn solver_proves_a_losing_move_and_stops_choosing_it() {
        for k in [1, 4] {
            let r = search(TreeGame::new(trap_tree()), &solver_cfg(64, k)).unwrap();
            assert_eq!(r.edges[0].proof, Some(Proof::Loss), "K={k}");
            assert_eq!(r.edges[0].proof_plies, 2, "K={k}");
            assert_eq!(r.best_action(), Some(1), "K={k}: {:?}", r.edges);
            // Once proven, the losing move receives no further visits.
            let longer = search(TreeGame::new(trap_tree()), &solver_cfg(256, k)).unwrap();
            assert_eq!(longer.edges[0].visits, r.edges[0].visits, "K={k}");
        }
        let off = search(
            TreeGame::new(trap_tree()),
            &PuctConfig {
                c_puct: 1.0,
                simulations: 64,
                leaves_in_flight: 1,
                solver: false,
            },
        )
        .unwrap();
        assert!(off.edges.iter().all(|e| e.proof.is_none()));
        assert_eq!(off.root_proof, None);
    }

    #[test]
    fn solver_proves_a_win_and_concentrates_visits_on_it() {
        let nodes = vec![
            inner(vec![1, 2], 0.0, Some(vec![0.2, 0.8])),
            terminal(-1.0), // opponent checkmated
            inner(vec![3], 0.0, None),
            terminal(0.0),
        ];
        for k in [1, 4] {
            let r = search(TreeGame::new(nodes.clone()), &solver_cfg(64, k)).unwrap();
            assert_eq!(r.root_proof, Some(Proof::Win), "K={k}");
            assert_eq!(r.edges[0].proof, Some(Proof::Win));
            assert_eq!(r.edges[0].proof_plies, 1);
            assert_eq!(r.best_action(), Some(0));
            assert!(r.edges[0].visits > 55, "K={k}: {:?}", r.edges);
        }
    }

    #[test]
    fn solver_proves_draws_and_losses_when_every_move_is_proven() {
        // Every move reaches a terminal: one draws, one loses -> draw.
        let draw = vec![
            inner(vec![1, 2], 0.0, None),
            terminal(0.0),
            inner(vec![3], 0.0, None),
            terminal(-1.0), // after move 1 the opponent mates us next
        ];
        // `2` is an opponent node with one move (to 3) where the root side is
        // checkmated: from the root that move is a proven loss.
        let draw = {
            let mut d = draw;
            d[3] = terminal(-1.0);
            d
        };
        let r = search(TreeGame::new(draw), &solver_cfg(64, 1)).unwrap();
        assert_eq!(r.edges[1].proof, Some(Proof::Loss));
        assert_eq!(r.edges[0].proof, Some(Proof::Draw));
        assert_eq!(r.root_proof, Some(Proof::Draw));
        assert_eq!(r.best_action(), Some(0));

        let lost = vec![
            inner(vec![1, 2], 0.0, None),
            inner(vec![3], 0.0, None),
            inner(vec![4], 0.0, None),
            terminal(-1.0),
            terminal(-1.0),
        ];
        let r = search(TreeGame::new(lost), &solver_cfg(64, 1)).unwrap();
        assert_eq!(r.root_proof, Some(Proof::Loss));
        assert!(
            r.root_value < -0.9,
            "exact losses back up: {}",
            r.root_value
        );
    }

    #[test]
    fn all_values_finite_no_nan() {
        let nodes = vec![
            TreeNode {
                children: vec![1, 2],
                terminal: None,
                value: 0.0,
                policy: Some(vec![1.0, 0.0]),
            },
            leaf(),
            leaf(),
        ];
        let g = TreeGame::new(nodes);
        let r = search(
            g,
            &PuctConfig {
                c_puct: 1.0,
                simulations: 16,
                leaves_in_flight: 1,
                solver: false,
            },
        )
        .unwrap();
        assert!(r.root_value.is_finite());
        for e in &r.edges {
            assert!(e.prior.is_finite());
        }
        for p in r.policy() {
            assert!(p.is_finite());
        }
    }
}
