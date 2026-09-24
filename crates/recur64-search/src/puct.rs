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
}

/// Search configuration. `c_puct` is explicit and configurable; the Phase 2
/// default is a pilot value, not a tuned constant.
#[derive(Debug, Clone, Copy)]
pub struct PuctConfig {
    pub c_puct: f32,
    pub simulations: u32,
}

impl Default for PuctConfig {
    fn default() -> Self {
        Self {
            c_puct: 1.0,
            simulations: 16,
        }
    }
}

/// One root edge in the search result.
#[derive(Debug, Clone)]
pub struct RootEdge<A> {
    pub action: A,
    pub prior: f32,
    pub visits: u32,
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
        }
    }

    fn expand(&mut self) -> Result<(), EvalError> {
        let legal = self.game.legal_actions();
        let eval = self.game.evaluate(&legal)?;
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

fn select<G: PuctGame>(node: &Node<G>, cfg: &PuctConfig) -> usize {
    let sqrt_total = (node.visits_total as f32).sqrt();
    let mut best = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (i, e) in node.edges.iter().enumerate() {
        let q = if e.n > 0 { e.w / e.n as f32 } else { 0.0 };
        let u = cfg.c_puct * e.prior * sqrt_total / (1.0 + e.n as f32);
        let score = q + u;
        let better =
            score > best_score || (score == best_score && e.prior > node.edges[best].prior);
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
        let value = match child.terminal {
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
        match child.terminal {
            Some(t) => t,
            None => simulate(child, cfg)?,
        }
    };

    // Child value is from the child's perspective; from this node's perspective
    // it is negated.
    let edge_value = -child_value;
    node.edges[idx].w += edge_value;
    node.edges[idx].n += 1;
    node.visits_total += 1;
    Ok(edge_value)
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
        });
    }
    if cfg.simulations == 0 {
        return Ok(SearchResult {
            edges: Vec::new(),
            total_visits: 0,
            traversals: 0,
            root_value: 0.0,
            root_network_value: 0.0,
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
    for _ in 1..cfg.simulations {
        simulate(&mut node, cfg)?;
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
        .map(|e| RootEdge {
            action: e.action,
            prior: e.prior,
            visits: e.n,
        })
        .collect();

    Ok(SearchResult {
        edges,
        total_visits: total,
        traversals: cfg.simulations,
        root_value,
        root_network_value: node.eval_value,
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
            },
        )
        .unwrap();
        assert_eq!(
            r.edges.iter().map(|e| e.visits).collect::<Vec<_>>(),
            r2.edges.iter().map(|e| e.visits).collect::<Vec<_>>()
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
