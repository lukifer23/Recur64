//! The V4 run loop as an explicit, inspectable session.
//!
//! belief -> predicted utility -> exact query -> evidence message -> ledger -> belief.
//!
//! Examples run in lockstep (one query per step each, like V3). Every successor state enters
//! only through [`QueryManager::query`]. The session owns the real managers, trees and ledger;
//! counterfactual probes fork an isolated manager and never mutate any of them.

use burn::prelude::*;
use burn::tensor::{Bool, Int, TensorData};
use recur64_core::{ActionId, CandidateFactsV1, GameState, ObservationV1};
use recur64_model::active::features::{EDGE_FEATS, edge_features};
use recur64_model::active::modules::gather_rows;
use recur64_model::active::{EdgeRef, QueryScript, Tree};
use recur64_model::candidate::CandidateInputs;
use recur64_statequery::{QueryManager, StatePacketV1};

use crate::accounting::Accounting;
use crate::base::BaseStage;
use crate::belief::BeliefOutput;
use crate::content::ContentBatch;
use crate::ledger::{EvidenceLedger, EvidenceMessage};
use crate::model::EvidenceBeliefModel;
use crate::util::SplitMix;
use crate::utility::{BELIEF_FEATS, FrontierEdgeView};
use crate::{IN_FEATURES, MASKED_LOGIT, SQUARES};

/// Largest budget a V4 run accepts (B0..B16; larger is a different experiment).
pub const MAX_BUDGET: usize = 16;

/// How the next edge is chosen.
pub enum Selection<'a> {
    /// `argmax` predicted utility (lowest frontier index on ties).
    Utility,
    /// `fixed_bfs_actionid_v1`: minimum of `(parent depth, parent slot, ActionId)`.
    Fixed,
    /// Uniform over the frontier, seeded per example.
    Random(u64),
    /// An external schedule (replays, counterfactual studies, ablations).
    Script(&'a mut dyn QueryScript),
}

/// Treatment of the content that enters the evidence encoder. `Zero` and `Shuffled` are
/// EVALUATION-ONLY diagnostics: they can only be built through
/// [`RunOptions::zero_content_eval`] / [`RunOptions::shuffled_content_eval`], and `run`
/// refuses them unless the selection is an external replay script.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentMode {
    Normal,
    Zero,
    Shuffled,
}

/// Which parameter groups may receive gradient from this run (gradient scope). A frozen group
/// is detached, so under an autodiff backend its parameters get no gradient and AdamW leaves
/// them untouched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Freeze {
    /// Detach hypothesis tokens and `z0` (the immutable base).
    pub base: bool,
    /// Detach the evidence delta (encoder + belief update).
    pub evidence: bool,
}

impl Freeze {
    pub const NONE: Freeze = Freeze {
        base: false,
        evidence: false,
    };
    /// Base frozen, evidence trainable (Stage B / D).
    pub const BASE: Freeze = Freeze {
        base: true,
        evidence: false,
    };
    /// Base and evidence frozen (Stage C utility training, probes).
    pub const BASE_AND_EVIDENCE: Freeze = Freeze {
        base: true,
        evidence: true,
    };
}

#[derive(Debug, Clone)]
pub struct RunOptions {
    pub budget: usize,
    pub health_checks: bool,
    /// Read message norms / trust back to the host (synchronises).
    pub diagnostics: bool,
    pub freeze: Freeze,
    /// Encode every discovered state with the parent/action state encoder (needed by the utility
    /// head). Off by default: evidence-only runs never read it, and it dominates their memory.
    pub state_path: bool,
    content: ContentMode,
}

impl RunOptions {
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            health_checks: true,
            diagnostics: false,
            freeze: Freeze::NONE,
            state_path: false,
            content: ContentMode::Normal,
        }
    }

    /// Enable the utility path's state encoding (required for `Selection::Utility`, utilities and probes).
    pub fn with_state(mut self) -> Self {
        self.state_path = true;
        self
    }

    pub fn with_freeze(mut self, freeze: Freeze) -> Self {
        self.freeze = freeze;
        self
    }

    pub fn content(&self) -> ContentMode {
        self.content
    }

    /// Evaluation-only: replay the queried path with every content tensor zeroed.
    pub fn zero_content_eval(budget: usize) -> Self {
        Self {
            content: ContentMode::Zero,
            ..Self::new(budget)
        }
    }

    /// Evaluation-only: replay the queried path with the content of another example.
    pub fn shuffled_content_eval(budget: usize) -> Self {
        Self {
            content: ContentMode::Shuffled,
            ..Self::new(budget)
        }
    }
}

/// One executed query (host-side trace).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryTrace {
    pub step: usize,
    pub parent_slot: usize,
    pub action: u16,
    pub branch: usize,
    pub depth: u32,
    pub terminal: bool,
    pub frontier_size: usize,
}

/// Current belief and its decomposition.
pub struct Beliefs<B: Backend> {
    /// `[b, w]` `z_t = z0 + delta`; padding is `MASKED_LOGIT`.
    pub z: Tensor<B, 2>,
    /// `[b, w]` the immutable base logits.
    pub z0: Tensor<B, 2>,
    pub update: BeliefOutput<B>,
}

/// Everything a finished run produces.
pub struct V4Output<B: Backend> {
    pub z0: Tensor<B, 2>,
    pub logits: Tensor<B, 2>,
    pub delta: Tensor<B, 2>,
    /// `[b, w]` masked log-probabilities of `z_t` (padding exactly 0).
    pub log_probs: Tensor<B, 2>,
    /// `[b, w]` masked log-probabilities of `z0`.
    pub base_log_probs: Tensor<B, 2>,
    pub mask: Tensor<B, 2, Bool>,
    pub chosen: Vec<Vec<usize>>,
    pub traces: Vec<Vec<QueryTrace>>,
    pub messages: Vec<EvidenceMessage>,
    pub accounting: Accounting,
}

/// A hypothetical belief after one counterfactual query (host side; labels only).
#[derive(Debug, Clone)]
pub struct ProbeBelief {
    /// Index into the frontier handed to `probe_edges`.
    pub frontier_index: usize,
    /// `z_after` logits over the root candidates, padding stripped to the valid width.
    pub logits: Vec<f32>,
    /// Digest of the child state the isolated fork returned.
    pub child_digest: String,
}

fn int1<B: Backend>(data: Vec<i32>, device: &B::Device) -> Tensor<B, 1, Int> {
    let n = data.len();
    Tensor::from_data(TensorData::new(data, [n]), device)
}

fn int2<B: Backend>(data: Vec<i32>, shape: [usize; 2], device: &B::Device) -> Tensor<B, 2, Int> {
    Tensor::from_data(TensorData::new(data, shape), device)
}

fn host<B: Backend, const D: usize>(t: Tensor<B, D>, what: &str) -> anyhow::Result<Vec<f32>> {
    t.into_data()
        .to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("reading {what} to host: {e:?}"))
}

fn obs_vec(o: &ObservationV1) -> Vec<f32> {
    o.as_slice().to_vec()
}

/// Per-edge scalars of the current belief (host), in the order of `BELIEF_FEATS`.
fn belief_features(z: &[f32], valid: usize, branch: usize, n_msgs: usize, delta: &[f32]) -> [f32; BELIEF_FEATS] {
    let zs = &z[..valid];
    let mx = zs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let den: f32 = zs.iter().map(|v| (v - mx).exp()).sum();
    let p: Vec<f32> = zs.iter().map(|v| (v - mx).exp() / den).collect();
    let entropy: f32 = p.iter().map(|&q| if q > 0.0 { -q * q.ln() } else { 0.0 }).sum();
    let (mut best, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &v in zs {
        if v > best {
            second = best;
            best = v;
        } else if v > second {
            second = v;
        }
    }
    let margin = if second.is_finite() { best - second } else { 0.0 };
    let dnorm = delta[..valid].iter().map(|d| d * d).sum::<f32>().sqrt();
    [
        zs[branch] - mx,
        p[branch],
        entropy,
        p.iter().cloned().fold(0.0, f32::max),
        margin,
        n_msgs as f32 / 16.0,
        dnorm,
        delta[branch],
    ]
}

/// Computes the base stage from host inputs (observations, legal lists, candidate facts).
pub type BaseProvider<B> = dyn Fn(
    &[&ObservationV1],
    &[Vec<ActionId>],
    &[&[CandidateFactsV1]],
) -> anyhow::Result<BaseStage<B>>;

pub struct Session<'m, B: Backend> {
    model: &'m EvidenceBeliefModel<B>,
    device: B::Device,
    opts: RunOptions,
    b: usize,
    w: usize,
    mask: Tensor<B, 2, Bool>,
    /// Hypothesis tokens `[b, w, cd]` (detached when the base is frozen).
    tokens: Tensor<B, 3>,
    /// Base logits `[b, w]` (detached when the base is frozen).
    z0: Tensor<B, 2>,
    valid_width: Vec<usize>,
    managers: Vec<QueryManager>,
    trees: Vec<Tree>,
    /// `[example][slot]` observation of every known node (parent/root content).
    obs: Vec<Vec<Vec<f32>>>,
    ledger: EvidenceLedger<B>,
    node_pool: Vec<Tensor<B, 3>>,
    edge_embs: Vec<Tensor<B, 3>>,
    widths: Vec<usize>,
    rngs: Vec<SplitMix>,
    step: usize,
    chosen_log: Vec<Vec<usize>>,
    traces: Vec<Vec<QueryTrace>>,
    messages: Vec<EvidenceMessage>,
    acct: Accounting,
}

impl<'m, B: Backend> Session<'m, B> {
    pub fn new(
        model: &'m EvidenceBeliefModel<B>,
        roots: &[GameState],
        opts: RunOptions,
        random_seed: u64,
        device: &B::Device,
    ) -> anyhow::Result<Self> {
        Self::new_with(model, roots, opts, random_seed, device, None)
    }

    /// As [`Session::new`], with an optional provider of the base stage. A provider lets a frozen
    /// base be computed on a graph-free backend (see [`Session::new_frozen`]).
    pub fn new_with(
        model: &'m EvidenceBeliefModel<B>,
        roots: &[GameState],
        opts: RunOptions,
        random_seed: u64,
        device: &B::Device,
        base: Option<&BaseProvider<B>>,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(!roots.is_empty(), "empty batch");
        anyhow::ensure!(
            opts.budget <= MAX_BUDGET,
            "budget {} exceeds the V4 maximum {MAX_BUDGET}",
            opts.budget
        );
        for (i, r) in roots.iter().enumerate() {
            anyhow::ensure!(
                !r.is_terminal(),
                "root {i} is terminal; terminal roots bypass neural evaluation"
            );
        }
        let b = roots.len();
        let facts: Vec<Vec<CandidateFactsV1>> =
            roots.iter().map(recur64_core::candidate_facts).collect();
        let obs: Vec<ObservationV1> = roots
            .iter()
            .map(recur64_core::encode_observation_v1)
            .collect();
        let legal: Vec<Vec<ActionId>> = roots.iter().map(|s| s.legal_actions()).collect();
        for (i, l) in legal.iter().enumerate() {
            anyhow::ensure!(!l.is_empty(), "root {i} has no legal moves");
        }
        let obs_refs: Vec<&ObservationV1> = obs.iter().collect();
        let fact_refs: Vec<&[CandidateFactsV1]> = facts.iter().map(Vec::as_slice).collect();
        let inputs = CandidateInputs::<B>::from_parts(&obs_refs, &legal, &fact_refs, device)?;

        let mut managers = Vec::with_capacity(b);
        let mut trees = Vec::with_capacity(b);
        for (i, s) in roots.iter().enumerate() {
            let m = QueryManager::new(s.clone())?;
            let p = m.packet(0)?;
            let want: Vec<u16> = legal[i].iter().map(|x| x.index() as u16).collect();
            anyhow::ensure!(
                p.legal_actions == want,
                "root {i}: tool legal list differs from the candidate batch"
            );
            trees.push(Tree::new(&p)?);
            managers.push(m);
        }

        let stage = match base {
            Some(provide) => provide(&obs_refs, &legal, &fact_refs)?,
            None => model.base_stage(inputs.board.clone(), &inputs.cands, inputs.facts.clone()),
        };
        let w = inputs.cands.width;
        let (tokens, z0, root_node) = if opts.freeze.base {
            (
                stage.tokens.clone().detach(),
                stage.z0.clone().detach(),
                stage.root_node.clone().detach(),
            )
        } else {
            (
                stage.tokens.clone(),
                stage.z0.clone(),
                stage.root_node.clone(),
            )
        };
        let acct = Accounting {
            requested_budget: opts.budget,
            batch: b,
            successful_queries: vec![0; b],
            root_encoder_runs: 1,
            ..Accounting::default()
        };
        let m_dim = model.evidence().message_dim;
        Ok(Self {
            model,
            device: device.clone(),
            b,
            w,
            mask: inputs.cands.mask.clone(),
            tokens: tokens.clone(),
            z0,
            valid_width: legal.iter().map(Vec::len).collect(),
            managers,
            trees,
            obs: obs.iter().map(|o| vec![obs_vec(o)]).collect(),
            ledger: EvidenceLedger::new(b, m_dim),
            node_pool: vec![root_node.unsqueeze_dim::<3>(1)],
            edge_embs: vec![tokens],
            widths: vec![w],
            rngs: (0..b)
                .map(|i| {
                    SplitMix(
                        random_seed
                            ^ (i as u64)
                                .wrapping_mul(0xD6E8_FEB8_6659_FD93)
                                .wrapping_add(1),
                    )
                })
                .collect(),
            step: 0,
            chosen_log: vec![Vec::new(); b],
            traces: vec![Vec::new(); b],
            messages: Vec::new(),
            acct,
            opts,
        })
    }

    pub fn batch(&self) -> usize {
        self.b
    }

    pub fn width(&self) -> usize {
        self.w
    }

    pub fn step(&self) -> usize {
        self.step
    }

    pub fn mask(&self) -> &Tensor<B, 2, Bool> {
        &self.mask
    }

    pub fn ledger(&self) -> &EvidenceLedger<B> {
        &self.ledger
    }

    pub fn trees(&self) -> &[Tree] {
        &self.trees
    }

    pub fn managers(&self) -> &[QueryManager] {
        &self.managers
    }

    pub fn accounting(&self) -> &Accounting {
        &self.acct
    }

    pub fn z0(&self) -> &Tensor<B, 2> {
        &self.z0
    }

    pub fn frontiers(&self) -> Vec<Vec<EdgeRef>> {
        self.trees.iter().map(Tree::frontier).collect()
    }

    fn delta_of(&self, update: &BeliefOutput<B>) -> Tensor<B, 2> {
        if self.opts.freeze.evidence {
            update.delta.clone().detach()
        } else {
            update.delta.clone()
        }
    }

    /// `z_t = z0 + delta_z_t` for the current ledger. `z0` is only read.
    pub fn belief(&self) -> Beliefs<B> {
        let update = self
            .model
            .update
            .forward(self.tokens.clone(), self.mask.clone(), &self.ledger);
        let z = self.z0.clone() + self.delta_of(&update);
        Beliefs {
            z,
            z0: self.z0.clone(),
            update,
        }
    }

    /// Predicted utility of every frontier edge, computed from parent-known information only.
    /// Performs no query. Returns `(scores [b, fmax], view)`.
    pub fn utilities(
        &self,
        fronts: &[Vec<EdgeRef>],
    ) -> anyhow::Result<(Tensor<B, 2>, FrontierEdgeView<B>)> {
        anyhow::ensure!(
            self.opts.state_path,
            "utility scoring needs RunOptions::with_state (the parent/action state path is off)"
        );
        let device = &self.device;
        let (b, w) = (self.b, self.w);
        let qd = self.model.evidence().token_dim();
        let m_dim = self.model.evidence().message_dim;
        let remaining = self.opts.budget.saturating_sub(self.step);
        let fmax = fronts.iter().map(Vec::len).max().unwrap_or(1).max(1);
        let mut offsets = Vec::with_capacity(self.widths.len());
        let mut acc = 0usize;
        for &wd in &self.widths {
            offsets.push(acc);
            acc += wd;
        }
        let beliefs = self.belief();
        let zh = host(beliefs.z.clone(), "belief logits")?;
        let dh = host(beliefs.update.delta.clone(), "belief delta")?;

        let mut node_idx = vec![0i32; b * fmax];
        let mut flat_idx = vec![0i32; b * fmax];
        let mut branch_idx = vec![0i32; b * fmax];
        let mut feats = vec![0.0f32; b * fmax * EDGE_FEATS];
        let mut bel = vec![0.0f32; b * fmax * BELIEF_FEATS];
        let mut mask = vec![false; b * fmax];
        for (e, front) in fronts.iter().enumerate() {
            let zr = &zh[e * w..(e + 1) * w];
            let dr = &dh[e * w..(e + 1) * w];
            let nm = self.ledger.count(e);
            for (i, edge) in front.iter().enumerate() {
                let at = e * fmax + i;
                node_idx[at] = edge.node_slot as i32;
                flat_idx[at] = (offsets[edge.node_slot] + edge.pos) as i32;
                branch_idx[at] = edge.branch as i32;
                let f = edge_features(edge.parent_depth + 1, remaining, edge.parent_depth % 2)?;
                feats[at * EDGE_FEATS..(at + 1) * EDGE_FEATS].copy_from_slice(&f);
                let bf = belief_features(zr, self.valid_width[e], edge.branch, nm, dr);
                bel[at * BELIEF_FEATS..(at + 1) * BELIEF_FEATS].copy_from_slice(&bf);
                mask[at] = true;
            }
        }
        let node_t = int2::<B>(node_idx, [b, fmax], device);
        let flat_t = int2::<B>(flat_idx, [b, fmax], device);
        let branch_t = int2::<B>(branch_idx, [b, fmax], device);
        let e_cat = Tensor::cat(self.edge_embs.clone(), 1);
        let p_cat = Tensor::cat(self.node_pool.clone(), 1);
        let summary = self
            .ledger
            .mean_message(device)
            .unsqueeze_dim::<3>(1)
            .expand([b, fmax, m_dim]);
        let view = FrontierEdgeView {
            edge_emb: gather_rows(e_cat, flat_t),
            parent_pool: gather_rows(p_cat, node_t),
            branch_token: gather_rows(self.tokens.clone().detach(), branch_t),
            belief: Tensor::from_data(TensorData::new(bel, [b, fmax, BELIEF_FEATS]), device),
            ledger_summary: summary.detach(),
            edge_feats: Tensor::from_data(TensorData::new(feats, [b, fmax, EDGE_FEATS]), device),
            mask: Tensor::from_data(TensorData::new(mask, [b, fmax]), device),
        };
        debug_assert_eq!(view.edge_emb.dims()[2], qd);
        let scores = self.model.utility.score(&view);
        Ok((scores, view))
    }

    fn choose(
        &mut self,
        selection: &mut Selection<'_>,
        fronts: &[Vec<EdgeRef>],
        exhausted: &[bool],
    ) -> anyhow::Result<Vec<usize>> {
        let mut chosen = vec![usize::MAX; self.b];
        let util: Option<(Vec<f32>, usize)> = if matches!(selection, Selection::Utility) {
            let fmax = fronts.iter().map(Vec::len).max().unwrap_or(1).max(1);
            let (scores, _) = self.utilities(fronts)?;
            self.acct.utility_rows_executed += self.b;
            Some((host(scores, "utility scores")?, fmax))
        } else {
            None
        };
        for e in 0..self.b {
            if exhausted[e] {
                continue;
            }
            let front = &fronts[e];
            chosen[e] = match selection {
                Selection::Utility => {
                    let (s, fmax) = util.as_ref().expect("utility scores");
                    let row = &s[e * fmax..e * fmax + front.len()];
                    anyhow::ensure!(
                        row.iter().all(|v| v.is_finite()),
                        "non-finite utility at step {}, example {e}",
                        self.step
                    );
                    let mut best = 0usize;
                    for (i, &v) in row.iter().enumerate() {
                        if v > row[best] {
                            best = i;
                        }
                    }
                    best
                }
                Selection::Fixed => front
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, edge)| edge.bfs_key())
                    .map(|(i, _)| i)
                    .expect("non-empty frontier"),
                Selection::Random(_) => (self.rngs[e].next() % front.len() as u64) as usize,
                Selection::Script(script) => {
                    let s = script.next(e, self.step, front, &self.trees[e])?;
                    anyhow::ensure!(
                        s.follow < front.len(),
                        "script chose frontier index {} of {}",
                        s.follow,
                        front.len()
                    );
                    s.follow
                }
            };
        }
        Ok(chosen)
    }

    /// Execute one lockstep step. Returns `false` once every example is exhausted or the
    /// budget is spent.
    pub fn advance(&mut self, selection: &mut Selection<'_>) -> anyhow::Result<bool> {
        if self.step >= self.opts.budget {
            return Ok(false);
        }
        anyhow::ensure!(
            self.opts.content == ContentMode::Normal || matches!(selection, Selection::Script(_)),
            "zero/shuffled content is an evaluation-only diagnostic and needs an external replay \
             script (it is refused for utility, fixed or random selection)"
        );
        let device = self.device.clone();
        let b = self.b;
        let fronts = self.frontiers();
        let exhausted: Vec<bool> = fronts.iter().map(Vec::is_empty).collect();
        if exhausted.iter().all(|&e| e) {
            return Ok(false);
        }
        let chosen = self.choose(selection, &fronts, &exhausted)?;
        for e in 0..b {
            if !exhausted[e] {
                self.chosen_log[e].push(chosen[e]);
            }
        }

        // ---- exact queries: the ONLY way a successor reaches the model ----
        let mut children: Vec<Option<(usize, StatePacketV1, EdgeRef)>> =
            (0..b).map(|_| None).collect();
        for e in 0..b {
            if exhausted[e] {
                continue;
            }
            let edge = fronts[e][chosen[e]].clone();
            let parent_id = self.trees[e].node(edge.node_slot).id;
            let pkt = self.managers[e].query(parent_id, edge.action)?;
            let slot = self.trees[e].add_child(&edge, &pkt)?;
            anyhow::ensure!(
                slot == self.step + 1,
                "example {e}: child slot {slot} at step {} (lockstep violated)",
                self.step
            );
            self.acct.successful_queries[e] += 1;
            self.traces[e].push(QueryTrace {
                step: self.step,
                parent_slot: edge.node_slot,
                action: edge.action,
                branch: edge.branch,
                depth: pkt.ply_from_root,
                terminal: pkt.terminal,
                frontier_size: fronts[e].len(),
            });
            self.obs[e].push(obs_vec(&pkt.observation));
            children[e] = Some((slot, pkt, edge));
        }
        let active: Vec<usize> = (0..b).filter(|&e| children[e].is_some()).collect();
        let n = active.len();

        // ---- evidence message (content only) ----
        let mut root_o: Vec<&[f32]> = Vec::with_capacity(n);
        let mut par_o: Vec<&[f32]> = Vec::with_capacity(n);
        let mut chi_o: Vec<&[f32]> = Vec::with_capacity(n);
        let mut acts = Vec::with_capacity(n);
        let mut flags = Vec::with_capacity(n);
        for &e in &active {
            let (slot, pkt, edge) = children[e].as_ref().expect("active child");
            root_o.push(&self.obs[e][0]);
            par_o.push(&self.obs[e][edge.node_slot]);
            chi_o.push(&self.obs[e][*slot]);
            acts.push(edge.action);
            flags.push([f32::from(u8::from(pkt.in_check)), f32::from(u8::from(pkt.terminal))]);
        }
        let mut content =
            ContentBatch::<B>::from_host(&root_o, &par_o, &chi_o, &acts, &flags, &device)?;
        match self.opts.content {
            ContentMode::Normal => {}
            ContentMode::Zero => content = content.zeroed(),
            ContentMode::Shuffled => {
                anyhow::ensure!(
                    n >= 2,
                    "shuffled content needs at least two examples that received a state"
                );
                content = content.rolled(1);
            }
        }
        let msg = self.model.encoder.forward(&content);
        self.acct.evidence_encoder_calls += 1;
        self.acct.evidence_encoder_examples += n;
        let msg = if self.opts.freeze.evidence { msg.detach() } else { msg };
        let branches: Vec<usize> = active
            .iter()
            .map(|&e| children[e].as_ref().expect("child").2.branch)
            .collect();
        let depths: Vec<u32> = active
            .iter()
            .map(|&e| children[e].as_ref().expect("child").1.ply_from_root)
            .collect();
        if self.opts.diagnostics {
            let mh = host(msg.clone(), "message")?;
            let m_dim = self.ledger.dim();
            for (row, &e) in active.iter().enumerate() {
                let norm = mh[row * m_dim..(row + 1) * m_dim]
                    .iter()
                    .map(|v| v * v)
                    .sum::<f32>()
                    .sqrt();
                self.messages.push(EvidenceMessage {
                    example: e,
                    step: self.step,
                    branch: branches[row],
                    depth: depths[row],
                    norm: Some(norm),
                    trust: None,
                });
            }
        } else {
            for (row, &e) in active.iter().enumerate() {
                self.messages.push(EvidenceMessage {
                    example: e,
                    step: self.step,
                    branch: branches[row],
                    depth: depths[row],
                    norm: None,
                    trust: None,
                });
            }
        }
        self.ledger.append(&device, &active, msg, &branches, &depths)?;

        // ---- parent/action state encoding for the utility path only ----
        if self.opts.state_path {
        let a_w = active
            .iter()
            .map(|&e| children[e].as_ref().map_or(0, |(_, p, _)| p.legal_actions.len()))
            .max()
            .unwrap_or(0)
            .max(1);
        let mut obs_buf = vec![0.0f32; n * SQUARES * IN_FEATURES];
        let (mut from_v, mut to_v, mut promo_v) = (
            vec![0i32; n * a_w],
            vec![0i32; n * a_w],
            vec![0i32; n * a_w],
        );
        for (row, &e) in active.iter().enumerate() {
            let (_, p, _) = children[e].as_ref().expect("child");
            obs_buf[row * SQUARES * IN_FEATURES..(row + 1) * SQUARES * IN_FEATURES]
                .copy_from_slice(p.observation.as_slice());
            for (i, &act) in p.legal_actions.iter().enumerate() {
                let (f, t, pr) = ActionId::from_index(u32::from(act))?.decode();
                from_v[row * a_w + i] = f as i32;
                to_v[row * a_w + i] = t as i32;
                promo_v[row * a_w + i] = i32::from(pr.code());
            }
        }
        let obs_t = Tensor::<B, 3>::from_data(
            TensorData::new(obs_buf, [n, SQUARES, IN_FEATURES]),
            &device,
        );
        let (sq, pool_c) = self.model.state.forward(obs_t);
        let emb_c = self.model.state.action_embeddings(
            &sq,
            int2::<B>(from_v, [n, a_w], &device),
            int2::<B>(to_v, [n, a_w], &device),
            int2::<B>(promo_v, [n, a_w], &device),
        );
        self.acct.state_encoder_calls += 1;
        let qd = self.model.evidence().token_dim();
        let mut zero_map = vec![n as i32; b];
        for (row, &e) in active.iter().enumerate() {
            zero_map[e] = row as i32;
        }
        let zero_t = int1::<B>(zero_map, &device);
        let pool_full = Tensor::cat(vec![pool_c, Tensor::<B, 2>::zeros([1, qd], &device)], 0)
            .select(0, zero_t.clone());
        let emb_full = Tensor::cat(
            vec![emb_c, Tensor::<B, 3>::zeros([1, a_w, qd], &device)],
            0,
        )
        .select(0, zero_t);
        self.node_pool.push(pool_full.unsqueeze_dim::<3>(1));
        self.edge_embs.push(emb_full);
        self.widths.push(a_w);
        }

        self.step += 1;
        Ok(true)
    }

    /// Spend the whole budget.
    pub fn run_all(&mut self, selection: &mut Selection<'_>) -> anyhow::Result<()> {
        while self.advance(selection)? {}
        Ok(())
    }

    /// Reconcile accounting and return the final belief.
    pub fn finish(mut self) -> anyhow::Result<V4Output<B>> {
        let beliefs = self.belief();
        let delta = self.delta_of(&beliefs.update);
        let z = beliefs.z.clone();
        if self.opts.health_checks {
            let zh = host(z.clone(), "final logits")?;
            anyhow::ensure!(
                zh.iter().all(|v| v.is_finite()),
                "non-finite final logits; refusing to continue with a substituted policy"
            );
            let dh = host(delta.clone(), "final delta")?;
            let bound = self.model.evidence().delta_bound as f32;
            anyhow::ensure!(
                dh.iter().all(|v| v.is_finite() && v.abs() <= bound),
                "evidence delta is non-finite or exceeds its bound {bound}"
            );
        }
        if self.opts.diagnostics {
            if let Some(trust) = &beliefs.update.trust {
                let th = host(trust.clone(), "trust")?;
                let j = self.ledger.slots();
                for m in &mut self.messages {
                    // slot index equals the step that produced the message (lockstep).
                    m.trust = Some(th[m.example * j + m.step]);
                }
            }
        }
        self.acct.total_successful_queries = self.acct.successful_queries.iter().sum();
        self.acct.state_transitions = self
            .managers
            .iter()
            .map(|m| u64::from(m.successful_queries()))
            .sum();
        self.acct.legal_moves_generated = self.managers.iter().map(|m| m.legal_moves_generated()).sum();
        self.acct.ledger_messages = (0..self.b).map(|e| self.ledger.count(e)).collect();
        self.acct.exhausted_examples = (0..self.b)
            .filter(|&e| self.acct.successful_queries[e] < self.opts.budget)
            .count();
        self.acct
            .check_invariants()
            .map_err(|e| anyhow::anyhow!("accounting invariant violated: {e}"))?;
        let log_probs = EvidenceBeliefModel::<B>::log_probs(z.clone(), &self.mask);
        let base_log_probs = EvidenceBeliefModel::<B>::log_probs(self.z0.clone(), &self.mask);
        Ok(V4Output {
            z0: self.z0,
            logits: z,
            delta,
            log_probs,
            base_log_probs,
            mask: self.mask,
            chosen: self.chosen_log,
            traces: self.traces,
            messages: self.messages,
            accounting: self.acct,
        })
    }

    /// Counterfactual probes for one example: for each edge, execute it in an ISOLATED fork
    /// (`QueryManager::new(parent_state.clone())`, its own counters), encode the exact child it
    /// returns and report the hypothetical belief. The real managers, trees, ledger and counters
    /// are not touched; fork queries are counted in `probe_queries` only. Labels only: nothing
    /// returned here is ever fed to the real trajectory.
    pub fn probe_edges(
        &mut self,
        example: usize,
        fronts: &[EdgeRef],
        picks: &[usize],
    ) -> anyhow::Result<Vec<ProbeBelief>> {
        anyhow::ensure!(example < self.b && !picks.is_empty(), "bad probe request");
        let device = self.device.clone();
        let k = picks.len();
        let mut root_o: Vec<&[f32]> = Vec::with_capacity(k);
        let mut par_o: Vec<&[f32]> = Vec::with_capacity(k);
        let mut chi_store: Vec<Vec<f32>> = Vec::with_capacity(k);
        let mut acts = Vec::with_capacity(k);
        let mut flags = Vec::with_capacity(k);
        let mut branches = Vec::with_capacity(k);
        let mut depths = Vec::with_capacity(k);
        let mut digests = Vec::with_capacity(k);
        for &pi in picks {
            anyhow::ensure!(pi < fronts.len(), "probe index outside the frontier");
            let edge = &fronts[pi];
            let parent_id = self.trees[example].node(edge.node_slot).id;
            let parent_state = self.managers[example].state(parent_id)?.clone();
            let mut fork = QueryManager::new(parent_state)?;
            let pkt = fork.query(0, edge.action)?;
            self.acct.probe_queries += u64::from(fork.successful_queries());
            digests.push(pkt.state_digest());
            chi_store.push(obs_vec(&pkt.observation));
            flags.push([f32::from(u8::from(pkt.in_check)), f32::from(u8::from(pkt.terminal))]);
            acts.push(edge.action);
            branches.push(edge.branch);
            depths.push(edge.parent_depth + 1);
            root_o.push(&self.obs[example][0]);
            par_o.push(&self.obs[example][edge.node_slot]);
        }
        let chi_o: Vec<&[f32]> = chi_store.iter().map(Vec::as_slice).collect();
        let content = ContentBatch::<B>::from_host(&root_o, &par_o, &chi_o, &acts, &flags, &device)?;
        let msg = self.model.encoder.forward(&content).detach();
        let fork_ledger = self.ledger.fork_with(&device, example, msg, &branches, &depths)?;
        let idx = int1::<B>(vec![example as i32; k], &device);
        let h = self.tokens.clone().select(0, idx.clone()).detach();
        let vw0 = self.valid_width[example];
        let mask_host: Vec<bool> = (0..k)
            .flat_map(|_| (0..self.w).map(move |c| c < vw0))
            .collect();
        let mask = Tensor::<B, 2, Bool>::from_data(TensorData::new(mask_host, [k, self.w]), &device);
        let update = self.model.update.forward(h, mask, &fork_ledger);
        let z0 = self.z0.clone().select(0, idx).detach();
        let z = host(z0 + update.delta.detach(), "probe logits")?;
        let w = self.w;
        let vw = self.valid_width[example];
        Ok((0..k)
            .map(|i| ProbeBelief {
                frontier_index: picks[i],
                logits: z[i * w..i * w + vw].to_vec(),
                child_digest: digests[i].clone(),
            })
            .collect())
    }

    /// Current belief logits of one example on its valid candidates (host).
    pub fn belief_logits(&self, example: usize) -> anyhow::Result<Vec<f32>> {
        let z = host(self.belief().z, "belief logits")?;
        Ok(z[example * self.w..example * self.w + self.valid_width[example]].to_vec())
    }
}

impl<B: Backend> EvidenceBeliefModel<B> {
    /// Run the model on root positions at `opts.budget` exact queries.
    pub fn run(
        &self,
        roots: &[GameState],
        opts: &RunOptions,
        mut selection: Selection<'_>,
        random_seed: u64,
        device: &B::Device,
    ) -> anyhow::Result<V4Output<B>> {
        anyhow::ensure!(
            opts.content() == ContentMode::Normal || matches!(selection, Selection::Script(_)),
            "zero/shuffled content is an evaluation-only diagnostic and needs an external replay script"
        );
        // `Random(seed)` carries its own seed; every other selection uses `random_seed` (unused).
        let seed = match &selection {
            Selection::Random(s) => *s,
            _ => random_seed,
        };
        let mut opts = opts.clone();
        if matches!(selection, Selection::Utility) {
            opts.state_path = true;
        }
        let mut session = Session::new(self, roots, opts, seed, device)?;
        session.run_all(&mut selection)?;
        session.finish()
    }
}

/// Root CE of `logits` against a uniform target over `correct` (candidate indices).
pub fn root_ce(logits: &[f32], correct: &[usize]) -> f32 {
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lse = mx + logits.iter().map(|v| (v - mx).exp()).sum::<f32>().ln();
    let n = correct.len() as f32;
    correct.iter().map(|&c| (lse - logits[c]) / n).sum()
}

/// Mask value used for padding in batched logits (re-export for callers building targets).
pub const PAD_LOGIT: f32 = MASKED_LOGIT;

impl<'m, B: Backend> Session<'m, B> {
    /// The belief the session would hold with `ledger` in place of its own (same hypothesis
    /// tokens and base logits). Used to test permutation invariance and ledger deletion.
    pub fn belief_for(&self, ledger: &EvidenceLedger<B>) -> Beliefs<B> {
        let update = self
            .model
            .update
            .forward(self.tokens.clone(), self.mask.clone(), ledger);
        let z = self.z0.clone() + self.delta_of(&update);
        Beliefs {
            z,
            z0: self.z0.clone(),
            update,
        }
    }

    /// The hypothesis tokens `[b, w, cd]` the evidence path reads.
    pub fn tokens(&self) -> &Tensor<B, 3> {
        &self.tokens
    }
}

impl<'m, B: burn::tensor::backend::AutodiffBackend> Session<'m, B> {
    /// A session whose frozen base is computed on the graph-free inner backend and lifted in as
    /// constants. Under autodiff the base forward would otherwise keep a full activation graph that
    /// is never back-propagated (the base is detached), which exhausts VRAM; V3.5 solved the same
    /// problem by running its forward-only pass on the inference copy. Values are identical.
    pub fn new_frozen(
        model: &'m EvidenceBeliefModel<B>,
        roots: &[GameState],
        opts: RunOptions,
        random_seed: u64,
        device: &B::Device,
    ) -> anyhow::Result<Self> {
        use burn::module::AutodiffModule;
        anyhow::ensure!(opts.freeze.base, "new_frozen needs Freeze::base");
        let inner = model.valid();
        let dev = device.clone();
        let provide = move |obs: &[&ObservationV1],
                            legal: &[Vec<ActionId>],
                            facts: &[&[CandidateFactsV1]]|
              -> anyhow::Result<BaseStage<B>> {
            let inputs = CandidateInputs::<B::InnerBackend>::from_parts(obs, legal, facts, &dev)?;
            let s = inner.base_stage(inputs.board.clone(), &inputs.cands, inputs.facts.clone());
            Ok(BaseStage {
                tokens: Tensor::from_inner(s.tokens),
                root_node: Tensor::from_inner(s.root_node),
                z0: Tensor::from_inner(s.z0),
                wdl_logits: Tensor::from_inner(s.wdl_logits),
            })
        };
        Self::new_with(model, roots, opts, random_seed, device, Some(&provide))
    }
}
