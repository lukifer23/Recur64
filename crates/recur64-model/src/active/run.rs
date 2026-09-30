//! The active-search driver: one root encoding, then up to `budget` rounds of
//! select -> exact query -> encode -> plan, then the root readout.
//!
//! Examples run in lockstep. Every successful query is one exact edge
//! transition through [`recur64_statequery::QueryManager`]; the model never
//! touches a successor except through that tool.

use std::time::Instant;

use burn::prelude::*;
use burn::tensor::{Bool, Int, TensorData};
use recur64_core::{ActionId, GameState};
use recur64_statequery::{QueryManager, StatePacketV1};

use crate::config::{ACTIVE_ENGINEERING_MAX_BUDGET, ACTIVE_MAX_BUDGET};
use crate::model::{CandidateTensors, ModelOutput, Readout};

use super::accounting::{Accounting, QueryRecord, StepDiag};
use super::features::{EDGE_FEATS, EVENT_FEATS, STOP_FEATS, edge_features, event_features, scalar};
use super::model::ActiveSearchModel;
use super::modules::{MASKED_LOGIT, gather_one, gather_rows, rms_all};
use super::tree::{EdgeRef, Tree};

/// One scripted decision for one example at one step.
pub struct ScriptStep {
    /// Index into the frontier handed to the script: the edge to query.
    pub follow: usize,
    /// Indices into the frontier forming the selector target set (uniform
    /// target). Empty: no selector loss for this example at this step.
    pub targets: Vec<usize>,
}

/// Supplies the query schedule (and, for training, selector targets). Used for
/// teacher forcing and for the ORACLE control. The frontier is deterministic
/// ([`Tree::frontier`]).
pub trait QueryScript {
    fn next(
        &mut self,
        example: usize,
        step: usize,
        frontier: &[EdgeRef],
        tree: &Tree,
    ) -> anyhow::Result<ScriptStep>;
}

/// How the next edge is chosen.
pub enum Selection<'a> {
    /// The learned selector (argmax, lowest index on ties).
    Active,
    /// `fixed_bfs_actionid_v1`: minimum of `(parent depth, parent slot, ActionId)`.
    /// Reads no labels, scores or network output.
    Fixed,
    /// Uniform over the frontier, seeded per example.
    Random(u64),
    /// An external schedule.
    Script(&'a mut dyn QueryScript),
}

/// Run options.
#[derive(Debug, Clone)]
pub struct RunOptions {
    /// Number of exact queries to spend.
    pub budget: usize,
    /// Synchronise and check workspace / branch RMS and final outputs each step.
    pub health_checks: bool,
    /// Record per-step planner diagnostics (synchronises).
    pub diagnostics: bool,
    /// Synchronise the device before reading each section clock.
    pub timing_sync: bool,
    /// Must be true: the primary experiment forces the full budget and learned
    /// STOP is not implemented until `adaptive_stop_v1`.
    pub stop_masked: bool,
    /// Allow a budget above the V3.0 scientific maximum (up to the engineering
    /// ceiling). Such a run is recorded as engineering only and is never science.
    pub engineering_stress: bool,
}

impl RunOptions {
    pub fn forced(budget: usize) -> Self {
        Self {
            budget,
            health_checks: true,
            diagnostics: false,
            timing_sync: false,
            stop_masked: true,
            engineering_stress: false,
        }
    }
}

/// Selector logits of one round, kept for the process loss.
pub struct SelectorStep<B: Backend> {
    /// `[b, F + 1]`; the last column is STOP (masked).
    pub logits: Tensor<B, 2>,
    pub mask: Tensor<B, 2, Bool>,
    /// `[b, F + 1]` target distribution, zero rows without a target.
    pub target: Tensor<B, 2>,
    pub has_target: Vec<bool>,
}

/// Everything a run produces.
pub struct ActiveOutput<B: Backend> {
    pub readout: Readout<B>,
    pub selector_steps: Vec<SelectorStep<B>>,
    pub traces: Vec<Vec<QueryRecord>>,
    pub accounting: Accounting,
    pub diagnostics: Vec<StepDiag>,
}

impl<B: Backend> ActiveOutput<B> {
    /// The root readout in the shape the existing losses consume.
    pub fn into_model_output(self, executed_blocks: usize) -> ModelOutput<B> {
        ModelOutput {
            readouts: vec![self.readout],
            executed_blocks,
        }
    }
}

struct SplitMix(u64);
impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn host_vec<B: Backend, const D: usize>(t: Tensor<B, D>, what: &str) -> anyhow::Result<Vec<f32>> {
    t.into_data()
        .to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("reading {what} to host: {e:?}"))
}

fn scalar_of<B: Backend>(t: Tensor<B, 1>, what: &str) -> anyhow::Result<f32> {
    Ok(host_vec(t, what)?[0])
}

fn int2<B: Backend>(data: Vec<i32>, shape: [usize; 2], device: &B::Device) -> Tensor<B, 2, Int> {
    Tensor::from_data(TensorData::new(data, shape), device)
}

fn int1<B: Backend>(data: Vec<i32>, device: &B::Device) -> Tensor<B, 1, Int> {
    let n = data.len();
    Tensor::from_data(TensorData::new(data, [n]), device)
}

fn lap<B: Backend>(start: &mut Instant, sync: bool, device: &B::Device) -> f64 {
    if sync {
        let _ = B::sync(device);
    }
    let s = start.elapsed().as_secs_f64();
    *start = Instant::now();
    s
}

/// Softmax entropy and top-two logit margin over the valid frontier logits.
fn selector_stats(row: &[f32]) -> (f32, f32) {
    let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let z: f32 = row.iter().map(|v| (v - mx).exp()).sum();
    let entropy = row
        .iter()
        .map(|v| {
            let p = (v - mx).exp() / z;
            if p > 0.0 { -p * p.ln() } else { 0.0 }
        })
        .sum();
    let mut best = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;
    for &v in row {
        if v > best {
            second = best;
            best = v;
        } else if v > second {
            second = v;
        }
    }
    let margin = if second.is_finite() {
        best - second
    } else {
        0.0
    };
    (entropy, margin)
}

impl<B: Backend> ActiveSearchModel<B> {
    /// Run the model on root positions at `opts.budget` exact queries.
    ///
    /// Budget 0 performs exactly one root encoding and nothing else. The same
    /// function serves training (under an autodiff backend) and evaluation.
    pub fn run(
        &self,
        roots: &[GameState],
        opts: &RunOptions,
        mut selection: Selection<'_>,
        device: &B::Device,
    ) -> anyhow::Result<ActiveOutput<B>> {
        anyhow::ensure!(!roots.is_empty(), "empty batch");
        let ceiling = if opts.engineering_stress {
            ACTIVE_ENGINEERING_MAX_BUDGET
        } else {
            ACTIVE_MAX_BUDGET
        };
        anyhow::ensure!(
            opts.budget <= ceiling,
            "budget {} exceeds the maximum {ceiling} (V3.0 science is B0..B{ACTIVE_MAX_BUDGET}; \
             a larger budget is a different experiment, and engineering stress runs need \
             RunOptions::engineering_stress)",
            opts.budget
        );
        // Terminal roots are refused before any CandidateFacts or neural work.
        // `GameState::legal_actions` still lists moves at some rule-terminal
        // states (threefold, fifty-move), so legality of moves is not a test.
        for (i, r) in roots.iter().enumerate() {
            anyhow::ensure!(
                !r.is_terminal(),
                "root {i} is terminal ({}); terminal roots bypass neural evaluation",
                r.termination().map_or("?", |t| t.label())
            );
        }
        anyhow::ensure!(
            opts.stop_masked,
            "learned STOP is not implemented: the primary experiment forces the full budget \
             (adaptive_stop_v1 is a later identity)"
        );
        let a = self.active().clone();
        let qd = a.query_dim;
        let b = roots.len();
        let t_total = Instant::now();
        let mut acct = Accounting {
            requested_budget: opts.budget,
            engineering_only: opts.engineering_stress,
            batch: b,
            successful_queries: vec![0; b],
            query_depths: vec![Vec::new(); b],
            timing_synchronised: opts.timing_sync,
            ..Accounting::default()
        };

        // ---- root inputs (CandidateFacts are baseline work, timed separately) ----
        let mut clock = Instant::now();
        let facts: Vec<Vec<recur64_core::CandidateFactsV1>> =
            roots.iter().map(recur64_core::candidate_facts).collect();
        super::counters::note_root_facts(b);
        acct.root_facts_positions = b;
        acct.root_facts_s = lap::<B>(&mut clock, false, device);
        let obs: Vec<recur64_core::ObservationV1> = roots
            .iter()
            .map(recur64_core::encode_observation_v1)
            .collect();
        let legal: Vec<Vec<ActionId>> = roots.iter().map(|s| s.legal_actions()).collect();
        for (i, l) in legal.iter().enumerate() {
            anyhow::ensure!(
                !l.is_empty(),
                "root {i} has no legal moves; terminal roots bypass neural evaluation"
            );
        }
        let obs_refs: Vec<&recur64_core::ObservationV1> = obs.iter().collect();
        let fact_refs: Vec<&[recur64_core::CandidateFactsV1]> =
            facts.iter().map(Vec::as_slice).collect();
        let inputs = crate::candidate::CandidateInputs::<B>::from_parts(
            &obs_refs, &legal, &fact_refs, device,
        )?;
        let cands: &CandidateTensors<B> = &inputs.cands;

        // ---- exact tool state ----
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

        // ---- the single root execution ----
        clock = Instant::now();
        let stage = self.root_stage(inputs.board.clone(), cands, inputs.facts.clone());
        acct.root_encoder_runs = 1;
        acct.root_encoder_examples = b;
        acct.root_encoder_s = lap::<B>(&mut clock, opts.timing_sync, device);
        let tokens = stage.tokens.clone();
        let mut ws = self.planner.initial_workspace(b);
        let mut branch = self.planner.initial_branch(tokens.clone());
        let w = cands.width;

        let mut node_pool: Vec<Tensor<B, 3>> = vec![stage.root_node.clone().unsqueeze_dim::<3>(1)];
        let mut edge_embs: Vec<Tensor<B, 3>> = vec![tokens.clone()];
        let mut widths: Vec<usize> = vec![w];

        let mut traces: Vec<Vec<QueryRecord>> = vec![Vec::new(); b];
        let mut selector_steps = Vec::new();
        let mut diagnostics = Vec::new();
        let mut rngs: Vec<SplitMix> = (0..b)
            .map(|i| match &selection {
                Selection::Random(seed) => SplitMix(
                    seed ^ (i as u64)
                        .wrapping_mul(0xD6E8_FEB8_6659_FD93)
                        .wrapping_add(1),
                ),
                _ => SplitMix(0),
            })
            .collect();

        for step in 0..opts.budget {
            let remaining = opts.budget - step;
            clock = Instant::now();
            let fronts: Vec<Vec<EdgeRef>> = trees.iter().map(Tree::frontier).collect();
            let exhausted: Vec<bool> = fronts.iter().map(Vec::is_empty).collect();
            if exhausted.iter().all(|&e| e) {
                break;
            }
            let fmax = fronts.iter().map(Vec::len).max().unwrap_or(1);
            let mut offsets = Vec::with_capacity(widths.len());
            let mut acc = 0usize;
            for &wd in &widths {
                offsets.push(acc);
                acc += wd;
            }

            // ---- frontier tensors ----
            let mut node_idx = vec![0i32; b * fmax];
            let mut flat_idx = vec![0i32; b * fmax];
            let mut branch_idx = vec![0i32; b * fmax];
            let mut feats = vec![0.0f32; b * fmax * EDGE_FEATS];
            let mut mask = vec![false; b * (fmax + 1)];
            for (e, front) in fronts.iter().enumerate() {
                for (i, edge) in front.iter().enumerate() {
                    let at = e * fmax + i;
                    node_idx[at] = edge.node_slot as i32;
                    flat_idx[at] = (offsets[edge.node_slot] + edge.pos) as i32;
                    branch_idx[at] = edge.branch as i32;
                    let f = edge_features(edge.parent_depth + 1, remaining, edge.parent_depth % 2)?;
                    feats[at * EDGE_FEATS..(at + 1) * EDGE_FEATS].copy_from_slice(&f);
                    mask[e * (fmax + 1) + i] = true;
                }
            }
            let node_t = int2::<B>(node_idx, [b, fmax], device);
            let flat_t = int2::<B>(flat_idx, [b, fmax], device);
            let branch_t = int2::<B>(branch_idx, [b, fmax], device);
            let feats_t =
                Tensor::<B, 3>::from_data(TensorData::new(feats, [b, fmax, EDGE_FEATS]), device);
            let mask_t =
                Tensor::<B, 2, Bool>::from_data(TensorData::new(mask, [b, fmax + 1]), device);

            let e_cat = Tensor::cat(edge_embs.clone(), 1);
            let p_cat = Tensor::cat(node_pool.clone(), 1);
            let zp = ws.clone().mean_dim(1).squeeze_dim::<2>(1);
            let sel_in = Tensor::cat(
                vec![
                    gather_rows(e_cat.clone(), flat_t.clone()),
                    gather_rows(p_cat.clone(), node_t),
                    gather_rows(tokens.clone(), branch_t.clone()),
                    gather_rows(branch.clone(), branch_t),
                    zp.clone().unsqueeze_dim::<3>(1).expand([b, fmax, qd]),
                    feats_t,
                ],
                2,
            );
            let edge_logits = self.selector.edge_logits(sel_in);
            let mut stop_feats = vec![0.0f32; b * STOP_FEATS];
            for e in 0..b {
                stop_feats[e * STOP_FEATS..(e + 1) * STOP_FEATS]
                    .copy_from_slice(&scalar(remaining as f32));
            }
            let stop_t =
                Tensor::<B, 2>::from_data(TensorData::new(stop_feats, [b, STOP_FEATS]), device);
            let stop = self.selector.stop_logit(Tensor::cat(vec![zp, stop_t], 1));
            let logits = Tensor::cat(vec![edge_logits, stop], 1)
                .mask_fill(mask_t.clone().bool_not(), MASKED_LOGIT);
            acct.selector_rows_executed += b;
            acct.selector_edge_slots_executed += b * fmax;
            acct.selector_valid_edges += fronts.iter().map(Vec::len).sum::<usize>();

            // ---- choose ----
            let need_host = matches!(selection, Selection::Active) || opts.diagnostics;
            let host: Option<Vec<f32>> = if need_host {
                Some(host_vec(logits.clone(), "selector logits")?)
            } else {
                None
            };
            let mut chosen = vec![usize::MAX; b];
            let mut targets: Vec<Vec<usize>> = vec![Vec::new(); b];
            let mut stats: Vec<Option<(f32, f32)>> = vec![None; b];
            for e in 0..b {
                if exhausted[e] {
                    continue;
                }
                let front = &fronts[e];
                if let Some(h) = &host {
                    let row = &h[e * (fmax + 1)..e * (fmax + 1) + front.len()];
                    anyhow::ensure!(
                        row.iter().all(|v| v.is_finite()),
                        "non-finite selector logit at step {step}, example {e}"
                    );
                    stats[e] = Some(selector_stats(row));
                }
                chosen[e] = match &mut selection {
                    Selection::Active => {
                        let row = &host.as_ref().expect("host logits")
                            [e * (fmax + 1)..e * (fmax + 1) + front.len()];
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
                    Selection::Random(_) => (rngs[e].next() % front.len() as u64) as usize,
                    Selection::Script(script) => {
                        let s = script.next(e, step, front, &trees[e])?;
                        anyhow::ensure!(
                            s.follow < front.len(),
                            "script chose frontier index {} of {}",
                            s.follow,
                            front.len()
                        );
                        anyhow::ensure!(
                            s.targets.iter().all(|&t| t < front.len()),
                            "script target outside the frontier"
                        );
                        targets[e] = s.targets;
                        s.follow
                    }
                };
            }
            if matches!(selection, Selection::Script(_)) && targets.iter().any(|t| !t.is_empty()) {
                let mut tgt = vec![0.0f32; b * (fmax + 1)];
                let mut has = vec![false; b];
                for e in 0..b {
                    if !targets[e].is_empty() {
                        has[e] = true;
                        let p = 1.0 / targets[e].len() as f32;
                        for &t in &targets[e] {
                            tgt[e * (fmax + 1) + t] += p;
                        }
                    }
                }
                selector_steps.push(SelectorStep {
                    logits: logits.clone(),
                    mask: mask_t,
                    target: Tensor::from_data(TensorData::new(tgt, [b, fmax + 1]), device),
                    has_target: has,
                });
            }
            acct.planner_selector_s += lap::<B>(&mut clock, opts.timing_sync, device);

            // ---- exact queries (CPU) ----
            let mut children: Vec<Option<(usize, StatePacketV1)>> = (0..b).map(|_| None).collect();
            for e in 0..b {
                if exhausted[e] {
                    continue;
                }
                let edge = fronts[e][chosen[e]].clone();
                let parent_id = trees[e].node(edge.node_slot).id;
                let pkt = managers[e].query(parent_id, edge.action)?;
                let slot = trees[e].add_child(&edge, &pkt)?;
                anyhow::ensure!(
                    slot == step + 1,
                    "example {e}: child slot {slot} at step {step} (lockstep violated)"
                );
                acct.successful_queries[e] += 1;
                acct.query_depths[e].push(pkt.ply_from_root);
                traces[e].push(QueryRecord {
                    step,
                    parent_slot: edge.node_slot,
                    action: edge.action,
                    branch: edge.branch,
                    depth: pkt.ply_from_root,
                    terminal: pkt.terminal,
                    frontier_size: fronts[e].len(),
                    selector_entropy: stats[e].map(|s| s.0),
                    selector_margin: stats[e].map(|s| s.1),
                });
                children[e] = Some((slot, pkt));
            }
            let active_idx: Vec<usize> = (0..b).filter(|&e| children[e].is_some()).collect();
            let n_done = active_idx.len();
            acct.cpu_query_s += lap::<B>(&mut clock, false, device);

            // ---- encode the new states (only examples that received one) ----
            let a_w = active_idx
                .iter()
                .map(|&e| {
                    children[e]
                        .as_ref()
                        .map_or(0, |(_, p)| p.legal_actions.len())
                })
                .max()
                .unwrap_or(0)
                .max(1);
            let mut obs_buf = vec![0.0f32; n_done * 64 * 119];
            let (mut from_v, mut to_v, mut promo_v) = (
                vec![0i32; n_done * a_w],
                vec![0i32; n_done * a_w],
                vec![0i32; n_done * a_w],
            );
            let mut legal_valid = 0usize;
            for (row, &e) in active_idx.iter().enumerate() {
                let (_, p) = children[e].as_ref().expect("active example has a child");
                obs_buf[row * 64 * 119..(row + 1) * 64 * 119]
                    .copy_from_slice(p.observation.as_slice());
                legal_valid += p.legal_actions.len();
                for (i, &act) in p.legal_actions.iter().enumerate() {
                    let (f, t, pr) = ActionId::from_index(u32::from(act))?.decode();
                    from_v[row * a_w + i] = f as i32;
                    to_v[row * a_w + i] = t as i32;
                    promo_v[row * a_w + i] = i32::from(pr.code());
                }
            }
            let obs_t =
                Tensor::<B, 3>::from_data(TensorData::new(obs_buf, [n_done, 64, 119]), device);
            let (sq, child_pool_c) = self.query.forward(obs_t);
            let child_emb_c = self.query.action_embeddings(
                &sq,
                int2::<B>(from_v, [n_done, a_w], device),
                int2::<B>(to_v, [n_done, a_w], device),
                int2::<B>(promo_v, [n_done, a_w], device),
            );
            acct.query_encoder_calls += 1;
            acct.query_encoder_examples += n_done;
            acct.query_encoder_rows_executed += n_done;
            acct.query_action_slots_executed += n_done * a_w;
            acct.query_legal_actions_valid += legal_valid;
            acct.query_action_widths.push(a_w);
            acct.query_encoder_s += lap::<B>(&mut clock, opts.timing_sync, device);

            // ---- planner update (only examples that received a new state) ----
            let act_t = int1::<B>(active_idx.iter().map(|&e| e as i32).collect(), device);
            let mut ch_flat = vec![0i32; n_done];
            let mut ch_node = vec![0i32; n_done];
            let mut ch_branch = vec![0i32; n_done];
            let mut ev = vec![0.0f32; n_done * EVENT_FEATS];
            let mut onehot = vec![0.0f32; n_done * w];
            // Map a full-batch row to its compact row (or to the appended row).
            let mut zero_map = vec![n_done as i32; b];
            let mut merge_map: Vec<i32> = (0..b).map(|e| (n_done + e) as i32).collect();
            for (row, &e) in active_idx.iter().enumerate() {
                let (_, pkt) = children[e].as_ref().expect("active example has a child");
                let edge = &fronts[e][chosen[e]];
                ch_flat[row] = (offsets[edge.node_slot] + edge.pos) as i32;
                ch_node[row] = edge.node_slot as i32;
                ch_branch[row] = edge.branch as i32;
                let f = event_features(
                    pkt.ply_from_root,
                    remaining - 1,
                    edge.parent_depth % 2,
                    pkt.terminal,
                    pkt.in_check,
                )?;
                ev[row * EVENT_FEATS..(row + 1) * EVENT_FEATS].copy_from_slice(&f);
                onehot[row * w + edge.branch] = 1.0;
                zero_map[e] = row as i32;
                merge_map[e] = row as i32;
            }
            let ws_a = ws.clone().select(0, act_t.clone());
            let branch_a = branch.clone().select(0, act_t.clone());
            let ch_branch_t = int1::<B>(ch_branch, device);
            let event_in = Tensor::cat(
                vec![
                    gather_one(e_cat.select(0, act_t.clone()), int1::<B>(ch_flat, device)),
                    gather_one(p_cat.select(0, act_t.clone()), int1::<B>(ch_node, device)),
                    child_pool_c.clone(),
                    gather_one(tokens.clone().select(0, act_t), ch_branch_t.clone()),
                    gather_one(branch_a.clone(), ch_branch_t),
                    Tensor::<B, 2>::from_data(TensorData::new(ev, [n_done, EVENT_FEATS]), device),
                ],
                1,
            );
            let onehot_t =
                Tensor::<B, 3>::from_data(TensorData::new(onehot, [n_done, w, 1]), device);
            let out = self.planner.update(ws_a, branch_a, event_in, onehot_t);
            // Write the updated rows back; untouched examples keep their state.
            let merge_t = int1::<B>(merge_map, device);
            let prev_ws = ws.clone();
            let prev_br = branch.clone();
            ws = Tensor::cat(vec![out.workspace.clone(), ws], 0).select(0, merge_t.clone());
            branch = Tensor::cat(vec![out.branch.clone(), branch], 0).select(0, merge_t);
            acct.planner_update_calls += 1;
            acct.planner_update_examples += n_done;
            acct.planner_rows_executed += n_done;

            // Node stores keep full-batch rows; inactive rows are zero and never indexed.
            let zero_t = int1::<B>(zero_map, device);
            let pool_full = Tensor::cat(
                vec![child_pool_c, Tensor::<B, 2>::zeros([1, qd], device)],
                0,
            )
            .select(0, zero_t.clone());
            let emb_full = Tensor::cat(
                vec![child_emb_c, Tensor::<B, 3>::zeros([1, a_w, qd], device)],
                0,
            )
            .select(0, zero_t);
            node_pool.push(pool_full.unsqueeze_dim::<3>(1));
            edge_embs.push(emb_full);
            widths.push(a_w);

            if opts.health_checks || opts.diagnostics {
                let ws_rms = scalar_of(rms_all(ws.clone()), "workspace rms")?;
                let br_rms = scalar_of(rms_all(branch.clone()), "branch rms")?;
                anyhow::ensure!(
                    ws_rms.is_finite() && br_rms.is_finite(),
                    "non-finite workspace/branch state at step {step} (ws rms {ws_rms}, branch rms {br_rms})"
                );
                anyhow::ensure!(
                    f64::from(ws_rms) <= a.rms_ceiling && f64::from(br_rms) <= a.rms_ceiling,
                    "runaway state at step {step}: workspace rms {ws_rms}, branch rms {br_rms} exceed {}",
                    a.rms_ceiling
                );
                if opts.diagnostics {
                    let g = host_vec(out.gates.workspace.clone(), "workspace gates")?;
                    let gm = g.iter().sum::<f32>() / g.len() as f32;
                    let gs = (g.iter().map(|v| (v - gm) * (v - gm)).sum::<f32>() / g.len() as f32)
                        .sqrt();
                    let bg = host_vec(out.gates.branch.clone(), "branch gates")?;
                    diagnostics.push(StepDiag {
                        step,
                        workspace_rms: ws_rms,
                        workspace_delta: scalar_of(rms_all(ws.clone() - prev_ws), "ws delta")?,
                        branch_rms: br_rms,
                        branch_delta: scalar_of(rms_all(branch.clone() - prev_br), "branch delta")?,
                        gate_mean: gm,
                        gate_std: gs,
                        branch_gate_mean: bg.iter().sum::<f32>() / bg.len() as f32,
                        remaining,
                    });
                }
            }
            acct.steps_executed += 1;
            acct.planner_selector_s += lap::<B>(&mut clock, opts.timing_sync, device);
        }

        // ---- root readout ----
        clock = Instant::now();
        let policy = self.policy(tokens, branch, ws, cands);
        if opts.health_checks {
            let lp = host_vec(policy.log_probs.clone(), "policy log-probs")?;
            anyhow::ensure!(
                lp.iter().all(|v| v.is_finite()),
                "non-finite policy output; refusing to continue with a substituted policy"
            );
            let wdl = host_vec(stage.wdl_logits.clone(), "wdl logits")?;
            anyhow::ensure!(wdl.iter().all(|v| v.is_finite()), "non-finite wdl output");
        }
        acct.planner_selector_s += lap::<B>(&mut clock, opts.timing_sync, device);

        acct.total_successful_queries = acct.successful_queries.iter().sum();
        acct.state_transitions = managers
            .iter()
            .map(|m| m.successful_queries() as usize)
            .sum();
        acct.legal_moves_generated = managers.iter().map(|m| m.legal_moves_generated()).sum();
        acct.unique_nodes = trees.iter().map(Tree::len).sum();
        acct.terminal_nodes = trees.iter().map(Tree::terminal_count).sum();
        acct.transpositions_detected = trees.iter().map(Tree::transpositions).sum();
        acct.exhausted_examples = (0..b)
            .filter(|&e| acct.successful_queries[e] < opts.budget)
            .count();
        acct.final_frontier_empty = trees.iter().map(|t| t.frontier().is_empty()).collect();
        acct.total_s = t_total.elapsed().as_secs_f64();
        // The structural invariants are checked here, before any successful return,
        // so no caller can forget them.
        acct.check_invariants()
            .map_err(|e| anyhow::anyhow!("accounting invariant violated: {e}"))?;

        Ok(ActiveOutput {
            readout: Readout {
                policy,
                wdl_logits: stage.wdl_logits,
            },
            selector_steps,
            traces,
            accounting: acct,
            diagnostics,
        })
    }
}
