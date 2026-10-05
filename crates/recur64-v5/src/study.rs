//! Fixed-graph DEV reader evaluation and preregistered interventions.

use std::collections::{BTreeMap, HashSet};
use std::time::Instant;

use burn::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::ARCHITECTURE;
use crate::data::{DEV_DIGEST, TRAIN_DIGEST, V5Data};
use crate::evaluation::{
    COMPOSITION_SEED, CompositionPartition, EVAL_SCHEMA, EVAL_SEED, EvalRecord, EvaluationBundle,
    LoopHealth, PolicyMetrics, SHUFFLE_SEED, ShuffleMapping, TreatmentLabel, policy_metrics,
};
use crate::graph::{AcquiredGraph, EpisodeKey, Schedule, acquire};
use crate::model::{
    CounterfactualRelationalLoop, RootInputs, StreamLoopTensorTrace, Treatment, V5Inputs,
};

#[derive(Debug, Clone)]
pub struct EvaluationIdentity {
    pub source_sha: String,
    pub config_digest: String,
    pub model_hash: String,
    pub final_update: u64,
    pub scope: String,
    pub split: String,
    pub microbatch: usize,
    pub device: String,
}

pub const BASELINE_EVAL_SCHEMA: &str = "v5_final_baseline_evaluation_v3";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineRecord {
    pub position_id: String,
    pub family: String,
    pub mate_depth: u8,
    pub legal_actions: Vec<u16>,
    pub correct_indices: Vec<u32>,
    pub logits: Vec<f32>,
    pub metrics: PolicyMetrics,
    pub candidate_facts_evaluated: usize,
    pub candidate_facts_probe_evaluated: usize,
    pub candidate_facts_probe_seconds: f64,
    pub root_path_wall_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineEvaluation {
    pub schema: String,
    pub architecture: String,
    /// Scientific source of the evaluator publishing this report.
    pub source_sha: String,
    /// Scientific source that produced the immutable Stage A checkpoint.
    pub stage_a_source_sha: String,
    pub dev_record_id_digest: String,
    pub config_digest: String,
    pub train_digest: String,
    pub dev_digest: String,
    pub model_hash: String,
    pub baseline_fingerprint: String,
    pub final_update: u64,
    pub seed: u64,
    pub microbatch: usize,
    pub precision: String,
    pub device: String,
    pub scope: String,
    pub root_encoder_examples: usize,
    pub returned_encoder_examples: usize,
    pub exact_queries: usize,
    pub shared_core_applications: usize,
    pub records: Vec<BaselineRecord>,
}

/// SHA256 of lexicographically sorted record IDs, each followed by newline.
/// Multiplicity is retained; report validation independently refuses duplicates.
pub fn sorted_record_id_digest<'a>(ids: impl IntoIterator<Item = &'a str>) -> String {
    let mut sorted: Vec<_> = ids.into_iter().collect();
    sorted.sort_unstable();
    let mut hash = Sha256::new();
    for id in sorted {
        hash.update(id.as_bytes());
        hash.update(b"\n");
    }
    format!("{:x}", hash.finalize())
}

fn validate_baseline_record_ids(records: &[BaselineRecord], expected: &str) -> anyhow::Result<()> {
    let mut unique = HashSet::new();
    anyhow::ensure!(
        records
            .iter()
            .all(|row| unique.insert(row.position_id.as_str())),
        "duplicate final baseline record ID"
    );
    anyhow::ensure!(
        sorted_record_id_digest(records.iter().map(|row| row.position_id.as_str())) == expected,
        "baseline sorted DEV identity digest mismatch"
    );
    Ok(())
}

impl BaselineEvaluation {
    pub fn validate_against_data(&self, data: &V5Data) -> anyhow::Result<()> {
        self.validate()?;
        data.require_role(crate::native_data_v2::Role::Dev)?;
        data.verify_custody()?;
        let positions: BTreeMap<_, _> = data
            .dev
            .iter()
            .map(|&index| (data.position(index).id.as_str(), data.position(index)))
            .collect();
        for row in &self.records {
            let position = positions
                .get(row.position_id.as_str())
                .ok_or_else(|| anyhow::anyhow!("baseline record is not V5_HP_DEV_V2"))?;
            anyhow::ensure!(
                row.legal_actions == position.legal
                    && row.correct_indices == position.correct
                    && row.family == position.family
                    && row.mate_depth == position.mate_depth,
                "baseline labels/actions/cell disagree with authoritative V5_HP_DEV_V2 data"
            );
        }
        Ok(())
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.schema == BASELINE_EVAL_SCHEMA
                && self.architecture == ARCHITECTURE
                && self.config_digest == crate::config::V5Config::default().scientific_digest()?
                && self.train_digest == TRAIN_DIGEST
                && self.dev_digest == DEV_DIGEST
                && self.dev_record_id_digest
                    == crate::data::binding(crate::native_data_v2::Role::Dev)?.record_id_digest
                && [self.source_sha.as_str(), self.stage_a_source_sha.as_str()]
                    .iter()
                    .all(|sha| sha.len() == 40 && sha.bytes().all(|b| b.is_ascii_hexdigit()))
                && self.final_update == crate::stage::STAGE_A_UPDATES
                && self.seed == crate::stage::PILOT_SEED
                && matches!(self.microbatch, 1 | 2)
                && self.precision == "fp32"
                && self.scope == "all_dev_4500"
                && self.records.len() == 4_500
                && self.root_encoder_examples == 4_500
                && self.returned_encoder_examples == 0
                && self.exact_queries == 0
                && self.shared_core_applications == 0,
            "final baseline identity/accounting mismatch"
        );
        let mut ids = HashSet::new();
        let mut primary = 0;
        for row in &self.records {
            anyhow::ensure!(
                ids.insert(row.position_id.as_str())
                    && row.legal_actions.len() == row.logits.len()
                    && row.candidate_facts_evaluated == 2 * row.legal_actions.len()
                    && row.candidate_facts_probe_evaluated == row.legal_actions.len(),
                "duplicate or misaligned final baseline record"
            );
            let correct: Vec<usize> = row.correct_indices.iter().map(|&i| i as usize).collect();
            anyhow::ensure!(
                base_metrics(&row.logits, &correct)? == row.metrics,
                "baseline metrics disagree with per-position logits"
            );
            if row.family == "KQRvK" && row.mate_depth == 3 {
                primary += 1;
            }
        }
        anyhow::ensure!(
            primary == 750,
            "final baseline primary cell must contain 750 positions"
        );
        validate_baseline_record_ids(
            &self.records,
            &crate::data::binding(crate::native_data_v2::Role::Dev)?.record_id_digest,
        )?;
        Ok(())
    }
}

/// Final Stage A B0 evaluation: root path only, once per position. This is
/// separate from fixed-graph reader evaluation and performs no acquisition.
pub fn evaluate_final_baseline<B: Backend>(
    model: &CounterfactualRelationalLoop<B>,
    data: &V5Data,
    identity: EvaluationIdentity,
    stage_a_source_sha: String,
    fingerprint: String,
    device: &B::Device,
) -> anyhow::Result<BaselineEvaluation> {
    data.require_role(crate::native_data_v2::Role::Dev)?;
    data.verify_custody()?;
    anyhow::ensure!(
        matches!(identity.microbatch, 1 | 2),
        "invalid baseline microbatch"
    );
    let mut records = Vec::with_capacity(data.dev.len());
    let started = Instant::now();
    for indices in data.dev.chunks(identity.microbatch) {
        anyhow::ensure!(
            started.elapsed().as_secs() < 45 * 60,
            "baseline evaluation exceeded the bounded 45-minute process window"
        );
        let roots = data.roots(indices)?;
        let mut facts_times = Vec::new();
        for (&index, root) in indices.iter().zip(&roots) {
            data.validate_root_alignment(index, root)?;
            let probe = Instant::now();
            let facts = recur64_core::candidate_facts(root);
            anyhow::ensure!(facts.len() == data.position(index).legal.len());
            facts_times.push(probe.elapsed().as_secs_f64());
        }
        let wall = Instant::now();
        let refs: Vec<_> = roots.iter().collect();
        let input = RootInputs::<B>::from_roots(&refs, device)?;
        let width = input.cands.width;
        let z0 = values(model.base_root(&input).z0)?;
        let elapsed = wall.elapsed().as_secs_f64() / indices.len() as f64;
        for (row, &index) in indices.iter().enumerate() {
            let position = data.position(index);
            let logits = z0[row * width..row * width + position.legal.len()].to_vec();
            let correct: Vec<_> = position.correct.iter().map(|&i| i as usize).collect();
            records.push(BaselineRecord {
                position_id: position.id.clone(),
                family: position.family.clone(),
                mate_depth: position.mate_depth,
                legal_actions: position.legal.clone(),
                correct_indices: position.correct.clone(),
                metrics: base_metrics(&logits, &correct)?,
                logits,
                candidate_facts_evaluated: 2 * position.legal.len(),
                candidate_facts_probe_evaluated: position.legal.len(),
                candidate_facts_probe_seconds: facts_times[row],
                root_path_wall_seconds: elapsed,
            });
        }
    }
    let report = BaselineEvaluation {
        schema: BASELINE_EVAL_SCHEMA.into(),
        architecture: ARCHITECTURE.into(),
        source_sha: identity.source_sha,
        stage_a_source_sha,
        dev_record_id_digest: crate::data::binding(crate::native_data_v2::Role::Dev)?
            .record_id_digest,
        config_digest: identity.config_digest,
        train_digest: TRAIN_DIGEST.into(),
        dev_digest: DEV_DIGEST.into(),
        model_hash: identity.model_hash,
        baseline_fingerprint: fingerprint,
        final_update: identity.final_update,
        seed: crate::stage::PILOT_SEED,
        microbatch: identity.microbatch,
        precision: "fp32".into(),
        device: identity.device,
        scope: identity.scope,
        root_encoder_examples: records.len(),
        returned_encoder_examples: 0,
        exact_queries: 0,
        shared_core_applications: 0,
        records,
    };
    report.validate()?;
    Ok(report)
}

struct PreparedCell {
    indices: Vec<usize>,
    roots: Vec<recur64_core::GameState>,
    b0: Vec<Vec<f32>>,
    candidate_facts_seconds: Vec<f64>,
    graphs: BTreeMap<String, Vec<AcquiredGraph>>,
    query_seconds: BTreeMap<String, Vec<f64>>,
}

fn values<const D: usize, B: Backend>(tensor: Tensor<B, D>) -> anyhow::Result<Vec<f32>> {
    Ok(tensor.into_data().to_vec::<f32>()?)
}

fn stable_hash(parts: &[&[u8]]) -> u64 {
    let mut hash = Sha256::new();
    hash.update(b"recur64.v5.eval_hash.v1\0");
    for part in parts {
        hash.update(part);
        hash.update([0]);
    }
    u64::from_le_bytes(hash.finalize()[..8].try_into().expect("eight bytes"))
}

fn graph_manifest_hash(graphs: impl Iterator<Item = String>) -> String {
    let mut hash = Sha256::new();
    hash.update(b"recur64.v5.evaluation_graphs.v1\0");
    for digest in graphs {
        hash.update(digest.as_bytes());
        hash.update(b"\n");
    }
    format!("{:x}", hash.finalize())
}

fn prepare_cell<B: Backend>(
    model: &CounterfactualRelationalLoop<B>,
    data: &V5Data,
    indices: &[usize],
    microbatch: usize,
    device: &B::Device,
) -> anyhow::Result<PreparedCell> {
    let roots = data.roots(indices)?;
    let candidate_facts_seconds = roots
        .iter()
        .map(|root| {
            let started = Instant::now();
            let facts = recur64_core::candidate_facts(root);
            anyhow::ensure!(facts.len() == root.legal_actions().len());
            Ok(started.elapsed().as_secs_f64())
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut b0 = Vec::with_capacity(indices.len());
    for (indices_chunk, roots_chunk) in indices.chunks(microbatch).zip(roots.chunks(microbatch)) {
        for (&index, root) in indices_chunk.iter().zip(roots_chunk) {
            data.validate_root_alignment(index, root)?;
        }
        let refs: Vec<&recur64_core::GameState> = roots_chunk.iter().collect();
        let input = RootInputs::<B>::from_roots(&refs, device)?;
        let width = input.cands.width;
        let z0 = values(model.base_root(&input).z0)?;
        for (row, &index) in indices_chunk.iter().enumerate() {
            b0.push(z0[row * width..row * width + data.position(index).legal.len()].to_vec());
        }
    }
    let mut graphs = BTreeMap::new();
    let mut query_seconds = BTreeMap::new();
    for schedule in [Schedule::UniformFrontierV1, Schedule::BaseRankedDepthV1] {
        let mut schedule_graphs = Vec::with_capacity(indices.len());
        let mut schedule_seconds = Vec::with_capacity(indices.len());
        for (row, &index) in indices.iter().enumerate() {
            let started = Instant::now();
            let graph = acquire(
                &roots[row],
                EpisodeKey {
                    position_id: data.position(index).id.clone(),
                    schedule,
                    run_seed: EVAL_SEED,
                    occurrence_ordinal: 0,
                },
                8,
                (schedule == Schedule::BaseRankedDepthV1).then_some(b0[row].as_slice()),
            )?;
            schedule_seconds.push(started.elapsed().as_secs_f64());
            schedule_graphs.push(graph);
        }
        graphs.insert(schedule.id().into(), schedule_graphs);
        query_seconds.insert(schedule.id().into(), schedule_seconds);
    }
    Ok(PreparedCell {
        indices: indices.to_vec(),
        roots,
        b0,
        candidate_facts_seconds,
        graphs,
        query_seconds,
    })
}

fn shuffled_graphs(
    data: &V5Data,
    cell: &PreparedCell,
    schedule: Schedule,
) -> anyhow::Result<(Vec<AcquiredGraph>, Vec<ShuffleMapping>)> {
    let source = &cell.graphs[schedule.id()];
    let mut out = source.clone();
    let mut mappings = Vec::new();
    for (recipient, graph) in source.iter().enumerate() {
        let recipient_id = &data.position(cell.indices[recipient]).id;
        for (node_index, node) in graph.nodes.iter().enumerate() {
            let candidates: Vec<(usize, usize)> = source
                .iter()
                .enumerate()
                .filter(|(donor, _)| *donor != recipient)
                .flat_map(|(donor, graph)| {
                    graph
                        .nodes
                        .iter()
                        .enumerate()
                        .filter(move |(_, candidate)| candidate.depth == node.depth)
                        .map(move |(donor_node, _)| (donor, donor_node))
                })
                .collect();
            anyhow::ensure!(
                !candidates.is_empty(),
                "cannot derange {} {:?} depth {} within family/depth cell",
                recipient_id,
                schedule,
                node.depth
            );
            let path_bytes: Vec<u8> = node.path.iter().flat_map(|x| x.to_le_bytes()).collect();
            let seed = SHUFFLE_SEED.to_le_bytes();
            let pick = stable_hash(&[
                recipient_id.as_bytes(),
                schedule.id().as_bytes(),
                &path_bytes,
                &seed,
            ]) as usize
                % candidates.len();
            let (donor, donor_node) = candidates[pick];
            let donor_position_id = data.position(cell.indices[donor]).id.clone();
            anyhow::ensure!(donor_position_id != *recipient_id, "shuffle self-mapping");
            let donor_record = &source[donor].nodes[donor_node];
            out[recipient].nodes[node_index].payload = donor_record.payload.clone();
            mappings.push(ShuffleMapping {
                schedule,
                recipient_position_id: recipient_id.clone(),
                recipient_path: node.path.clone(),
                recipient_depth: node.depth,
                donor_position_id,
                donor_path: donor_record.path.clone(),
                donor_depth: donor_record.depth,
            });
        }
        out[recipient].digest.clear();
        out[recipient].digest = out[recipient].compute_digest()?;
        out[recipient].verify()?;
    }
    Ok((out, mappings))
}

fn composition_partition(
    position_id: &str,
    schedule: Schedule,
    graph: &AcquiredGraph,
) -> anyhow::Result<(Vec<bool>, CompositionPartition)> {
    anyhow::ensure!(
        graph.nodes.len() >= 2,
        "composition needs at least two acquired nodes"
    );
    let mut scored: Vec<(u64, usize)> = graph
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| {
            let path: Vec<u8> = node.path.iter().flat_map(|x| x.to_le_bytes()).collect();
            (
                stable_hash(&[
                    position_id.as_bytes(),
                    schedule.id().as_bytes(),
                    &path,
                    &COMPOSITION_SEED.to_le_bytes(),
                ]),
                index,
            )
        })
        .collect();
    let mut a: Vec<bool> = scored.iter().map(|(hash, _)| hash & 1 == 0).collect();
    if a.iter().all(|x| *x) || a.iter().all(|x| !*x) {
        scored.sort_unstable();
        a.fill(false);
        for (rank, &(_, index)) in scored.iter().enumerate() {
            a[index] = rank.is_multiple_of(2);
        }
    }
    let group_a_paths = graph
        .nodes
        .iter()
        .zip(&a)
        .filter(|(_, is_a)| **is_a)
        .map(|(node, _)| node.path.clone())
        .collect();
    let group_b_paths = graph
        .nodes
        .iter()
        .zip(&a)
        .filter(|(_, is_a)| !**is_a)
        .map(|(node, _)| node.path.clone())
        .collect();
    Ok((
        a,
        CompositionPartition {
            schedule,
            position_id: position_id.into(),
            group_a_paths,
            group_b_paths,
        },
    ))
}

fn base_metrics(logits: &[f32], correct: &[usize]) -> anyhow::Result<PolicyMetrics> {
    policy_metrics(
        logits,
        logits,
        &vec![0.0; logits.len()],
        &vec![0.0; logits.len()],
        correct,
    )
}

fn state_rms(
    before: &[f32],
    after: &[f32],
    row: usize,
    token_width: usize,
    valid_tokens: usize,
    hidden: usize,
) -> (f64, f64) {
    let start = row * token_width * hidden;
    let end = start + valid_tokens * hidden;
    let count = (valid_tokens * hidden).max(1) as f64;
    let state = after[start..end]
        .iter()
        .map(|&x| f64::from(x).powi(2))
        .sum::<f64>()
        / count;
    let update = before[start..end]
        .iter()
        .zip(&after[start..end])
        .map(|(&a, &b)| f64::from(b - a).powi(2))
        .sum::<f64>()
        / count;
    (state.sqrt(), update.sqrt())
}

#[allow(clippy::too_many_arguments)]
fn attention_health(
    attention: &[f32],
    row: usize,
    heads: usize,
    query_width: usize,
    memory_width: usize,
    valid_queries: usize,
    valid_memory: &[usize],
) -> (f64, f64) {
    let mut entropy = 0.0;
    let mut max_weight = 0.0;
    let distributions = (heads * valid_queries).max(1) as f64;
    for head in 0..heads {
        for query in 0..valid_queries {
            let base = ((row * heads + head) * query_width + query) * memory_width;
            let mut local_max = 0.0_f64;
            for &memory in valid_memory {
                let probability = f64::from(attention[base + memory]);
                if probability > 0.0 {
                    entropy -= probability * probability.ln();
                }
                local_max = local_max.max(probability);
            }
            max_weight += local_max;
        }
    }
    (entropy / distributions, max_weight / distributions)
}

fn stream_health<B: Backend>(
    traces: &[StreamLoopTensorTrace<B>],
    stream: &str,
    graphs: &[AcquiredGraph],
    candidate_width: usize,
    legal_widths: &[usize],
) -> anyhow::Result<Vec<Vec<LoopHealth>>> {
    let mut by_row = vec![Vec::with_capacity(traces.len()); graphs.len()];
    for (iteration, trace) in traces.iter().enumerate() {
        let [batch, evidence_width, hidden] = trace.evidence_before.dims();
        let hypothesis_width = trace.hypothesis_before.dims()[1];
        anyhow::ensure!(batch == graphs.len() && hypothesis_width == candidate_width);
        let ev_before = values(trace.evidence_before.clone())?;
        let ev_after = values(trace.evidence_after.clone())?;
        let hyp_before = values(trace.hypothesis_before.clone())?;
        let hyp_after = values(trace.hypothesis_after.clone())?;
        let ev_attention = values(trace.evidence_attention.clone())?;
        let hyp_attention = values(trace.hypothesis_attention.clone())?;
        let [_, heads, ev_queries, memory_width] = trace.evidence_attention.dims();
        let hyp_queries = trace.hypothesis_attention.dims()[2];
        for (row, graph) in graphs.iter().enumerate() {
            let evidence_valid = graph.actual_q * 4;
            let legal = legal_widths[row];
            let mut evidence_memory: Vec<usize> = (0..evidence_valid).collect();
            evidence_memory.extend(evidence_width..evidence_width + legal);
            evidence_memory.extend(
                evidence_width + candidate_width..evidence_width + candidate_width + crate::SQUARES,
            );
            let mut hypothesis_memory: Vec<usize> = (0..legal).collect();
            hypothesis_memory.extend(candidate_width..candidate_width + evidence_valid);
            hypothesis_memory.extend(
                candidate_width + evidence_width..candidate_width + evidence_width + crate::SQUARES,
            );
            let (ev_state, ev_update) = state_rms(
                &ev_before,
                &ev_after,
                row,
                evidence_width,
                evidence_valid,
                hidden,
            );
            let (hyp_state, hyp_update) = state_rms(
                &hyp_before,
                &hyp_after,
                row,
                hypothesis_width,
                legal,
                hidden,
            );
            let (ev_entropy, ev_max) = attention_health(
                &ev_attention,
                row,
                heads,
                ev_queries,
                memory_width,
                evidence_valid,
                &evidence_memory,
            );
            let (hyp_entropy, hyp_max) = attention_health(
                &hyp_attention,
                row,
                heads,
                hyp_queries,
                memory_width,
                legal,
                &hypothesis_memory,
            );
            by_row[row].push(LoopHealth {
                iteration: iteration + 1,
                stream: stream.into(),
                evidence_state_rms: ev_state,
                evidence_update_rms: ev_update,
                hypothesis_state_rms: hyp_state,
                hypothesis_update_rms: hyp_update,
                evidence_attention_entropy: ev_entropy,
                hypothesis_attention_entropy: hyp_entropy,
                evidence_attention_max: ev_max,
                hypothesis_attention_max: hyp_max,
            });
        }
    }
    Ok(by_row)
}

fn trace_health<B: Backend>(
    factual: &[StreamLoopTensorTrace<B>],
    null: &[StreamLoopTensorTrace<B>],
    graphs: &[AcquiredGraph],
    candidate_width: usize,
    legal_widths: &[usize],
) -> anyhow::Result<Vec<Vec<LoopHealth>>> {
    let mut factual = stream_health(factual, "factual", graphs, candidate_width, legal_widths)?;
    let null = stream_health(null, "null", graphs, candidate_width, legal_widths)?;
    for (row, extra) in factual.iter_mut().zip(null) {
        row.extend(extra);
    }
    Ok(factual)
}

#[allow(clippy::too_many_arguments)]
fn records_for<B: Backend>(
    model: &CounterfactualRelationalLoop<B>,
    data: &V5Data,
    cell: &PreparedCell,
    schedule: Schedule,
    graphs: &[AcquiredGraph],
    q: usize,
    r: usize,
    treatment_label: TreatmentLabel,
    treatment: Treatment,
    node_masks: Option<&[Vec<bool>]>,
    microbatch: usize,
    device: &B::Device,
) -> anyhow::Result<Vec<(EvalRecord, Vec<f32>)>> {
    let mut records = Vec::with_capacity(cell.indices.len());
    for start in (0..cell.indices.len()).step_by(microbatch) {
        let end = (start + microbatch).min(cell.indices.len());
        let mut prefixes = Vec::with_capacity(end - start);
        for graph in &graphs[start..end] {
            prefixes.push(graph.prefix(q)?);
        }
        let examples: Vec<(&recur64_core::GameState, &AcquiredGraph)> =
            cell.roots[start..end].iter().zip(&prefixes).collect();
        let input = V5Inputs::<B>::from_examples(&examples, device)?;
        let mask = node_masks
            .map(|all| input.payload_token_mask(&all[start..end], device))
            .transpose()?;
        let started = Instant::now();
        let base = model.base(&input);
        let traced = model.paired_traced_with_base_payload_mask(&input, base, r, treatment, mask);
        let elapsed = started.elapsed().as_secs_f64();
        let width = input.cands.width;
        let legal_widths: Vec<usize> = cell.indices[start..end]
            .iter()
            .map(|&index| data.position(index).legal.len())
            .collect();
        let health = trace_health(
            &traced.factual,
            &traced.null,
            &prefixes,
            width,
            &legal_widths,
        )?;
        let output = traced.output;
        let logits = values(output.logits)?;
        let z0 = values(output.z0)?;
        let raw = values(output.raw_delta)?;
        let centered = values(output.centered_delta)?;
        for (row, graph) in prefixes.iter().enumerate() {
            let at = start + row;
            let index = cell.indices[at];
            let legal = data.position(index).legal.len();
            let range = row * width..row * width + legal;
            let correct: Vec<usize> = data
                .position(index)
                .correct
                .iter()
                .map(|&x| x as usize)
                .collect();
            anyhow::ensure!(
                z0[range.clone()] == cell.b0[at],
                "reader replay changed B0 for {}",
                data.position(index).id
            );
            let branches: HashSet<usize> =
                graph.nodes.iter().map(|node| node.root_candidate).collect();
            let record = EvalRecord {
                position_id: data.position(index).id.clone(),
                family: data.position(index).family.clone(),
                mate_depth: data.position(index).mate_depth,
                schedule,
                q,
                r,
                treatment: treatment_label,
                graph_digest: Some(graph.digest.clone()),
                graph_structure_digest: Some(graph.compute_structure_digest()?),
                actual_q: graph.actual_q,
                exhausted_frontier: graph.exhausted_frontier,
                maximum_depth: graph.nodes.iter().map(|node| node.depth).max().unwrap_or(0),
                branches_covered: branches.len(),
                legal_generations: graph.legal_generations,
                legal_moves_generated: graph.legal_moves_generated,
                root_candidates: legal,
                root_candidate_facts_evaluated: legal,
                root_candidate_facts_probe_wall_seconds: cell.candidate_facts_seconds[at],
                q8_acquisition_wall_seconds: cell.query_seconds[schedule.id()][at],
                reader_wall_seconds: elapsed / (end - start) as f64,
                returned_encoder_examples: graph.actual_q,
                physical_rows: end - start,
                padded_rows: 0,
                shared_core_applications_per_example: 4 * r,
                metrics: policy_metrics(
                    &logits[range.clone()],
                    &z0[range.clone()],
                    &raw[range.clone()],
                    &centered[range.clone()],
                    &correct,
                )?,
                loop_health: health[row].clone(),
            };
            records.push((record, logits[range].to_vec()));
        }
    }
    Ok(records)
}

pub fn evaluate_reader<B: Backend>(
    model: &CounterfactualRelationalLoop<B>,
    data: &V5Data,
    indices: &[usize],
    identity: EvaluationIdentity,
    device: &B::Device,
) -> anyhow::Result<EvaluationBundle> {
    anyhow::ensure!(
        matches!(identity.microbatch, 1 | 2),
        "evaluation microbatch must be the frozen 2 or authorized fallback 1"
    );
    anyhow::ensure!(!indices.is_empty(), "empty evaluation scope");
    data.require_role(crate::native_data_v2::Role::Dev)?;
    data.verify_custody()?;
    let mut by_cell: BTreeMap<(String, u8), Vec<usize>> = BTreeMap::new();
    for &index in indices {
        let position = data.position(index);
        by_cell
            .entry((position.family.clone(), position.mate_depth))
            .or_default()
            .push(index);
    }
    let mut records = Vec::new();
    let mut shuffle_mappings = Vec::new();
    let mut composition_partitions = Vec::new();
    let mut graph_digests = Vec::new();
    let mut replay_exact = true;
    let replay_tolerance = if identity.device == "cpu" {
        0.0
    } else {
        1.0e-6
    };
    for ((family, depth), cell_indices) in by_cell {
        let cell = prepare_cell(model, data, &cell_indices, identity.microbatch, device)?;
        for schedule in [Schedule::UniformFrontierV1, Schedule::BaseRankedDepthV1] {
            let graphs = &cell.graphs[schedule.id()];
            graph_digests.extend(graphs.iter().map(|graph| graph.digest.clone()));
            for (row, &index) in cell.indices.iter().enumerate() {
                let correct: Vec<usize> = data
                    .position(index)
                    .correct
                    .iter()
                    .map(|&x| x as usize)
                    .collect();
                records.push(EvalRecord {
                    position_id: data.position(index).id.clone(),
                    family: family.clone(),
                    mate_depth: depth,
                    schedule,
                    q: 0,
                    r: 0,
                    treatment: TreatmentLabel::Baseline,
                    graph_digest: None,
                    graph_structure_digest: None,
                    actual_q: 0,
                    exhausted_frontier: false,
                    maximum_depth: 0,
                    branches_covered: 0,
                    legal_generations: 0,
                    legal_moves_generated: 0,
                    root_candidates: cell.b0[row].len(),
                    root_candidate_facts_evaluated: cell.b0[row].len(),
                    root_candidate_facts_probe_wall_seconds: cell.candidate_facts_seconds[row],
                    q8_acquisition_wall_seconds: 0.0,
                    reader_wall_seconds: 0.0,
                    returned_encoder_examples: 0,
                    physical_rows: 1,
                    padded_rows: 0,
                    shared_core_applications_per_example: 0,
                    metrics: base_metrics(&cell.b0[row], &correct)?,
                    loop_health: Vec::new(),
                });
            }
            let mut normal_logits: BTreeMap<usize, Vec<Vec<f32>>> = BTreeMap::new();
            for q in [2, 4, 8] {
                for r in [1, 2, 4] {
                    let evaluated = records_for(
                        model,
                        data,
                        &cell,
                        schedule,
                        graphs,
                        q,
                        r,
                        TreatmentLabel::Normal,
                        Treatment::Normal,
                        None,
                        identity.microbatch,
                        device,
                    )?;
                    if q == 8 && matches!(r, 1 | 4) {
                        normal_logits.insert(r, evaluated.iter().map(|x| x.1.clone()).collect());
                    }
                    records.extend(evaluated.into_iter().map(|x| x.0));
                }
            }
            let (shuffled, mappings) = shuffled_graphs(data, &cell, schedule)?;
            shuffle_mappings.extend(mappings);
            for r in [1, 4] {
                for (label, treatment, selected_graphs) in [
                    (
                        TreatmentLabel::PayloadShuffle,
                        Treatment::Normal,
                        shuffled.as_slice(),
                    ),
                    (
                        TreatmentLabel::NoRelationBias,
                        Treatment::NoRelationBias,
                        graphs.as_slice(),
                    ),
                    (
                        TreatmentLabel::NoHypothesisFeedback,
                        Treatment::NoHypothesisFeedback,
                        graphs.as_slice(),
                    ),
                    (
                        TreatmentLabel::AllPayloadNull,
                        Treatment::AllPayloadNull,
                        graphs.as_slice(),
                    ),
                ] {
                    records.extend(
                        records_for(
                            model,
                            data,
                            &cell,
                            schedule,
                            selected_graphs,
                            8,
                            r,
                            label,
                            treatment,
                            None,
                            identity.microbatch,
                            device,
                        )?
                        .into_iter()
                        .map(|x| x.0),
                    );
                }
            }
            if family == "KQRvK" && depth == 3 {
                let mut a_masks = Vec::with_capacity(graphs.len());
                for (row, graph) in graphs.iter().enumerate() {
                    let (mask, partition) = composition_partition(
                        &data.position(cell.indices[row]).id,
                        schedule,
                        graph,
                    )?;
                    a_masks.push(mask);
                    composition_partitions.push(partition);
                }
                let none_masks: Vec<Vec<bool>> = graphs
                    .iter()
                    .map(|graph| vec![false; graph.actual_q])
                    .collect();
                let b_masks: Vec<Vec<bool>> = a_masks
                    .iter()
                    .map(|mask| mask.iter().map(|x| !x).collect())
                    .collect();
                let both_masks: Vec<Vec<bool>> = graphs
                    .iter()
                    .map(|graph| vec![true; graph.actual_q])
                    .collect();
                for r in [1, 4] {
                    for (label, masks) in [
                        (TreatmentLabel::CompositionNeither, &none_masks),
                        (TreatmentLabel::CompositionA, &a_masks),
                        (TreatmentLabel::CompositionB, &b_masks),
                        (TreatmentLabel::CompositionBoth, &both_masks),
                    ] {
                        let evaluated = records_for(
                            model,
                            data,
                            &cell,
                            schedule,
                            graphs,
                            8,
                            r,
                            label,
                            Treatment::Normal,
                            Some(masks),
                            identity.microbatch,
                            device,
                        )?;
                        if label == TreatmentLabel::CompositionBoth {
                            for (row, (_, logits)) in evaluated.iter().enumerate() {
                                let max_abs = logits
                                    .iter()
                                    .zip(&normal_logits[&r][row])
                                    .map(|(a, b)| (a - b).abs())
                                    .fold(0.0_f32, f32::max);
                                replay_exact &= f64::from(max_abs) <= replay_tolerance;
                            }
                        }
                        records.extend(evaluated.into_iter().map(|x| x.0));
                    }
                }
            }
        }
    }
    anyhow::ensure!(
        replay_exact,
        "normal replay did not reproduce the source graph evaluation"
    );
    Ok(EvaluationBundle {
        schema: EVAL_SCHEMA.into(),
        architecture: ARCHITECTURE.into(),
        source_sha: identity.source_sha,
        config_digest: identity.config_digest,
        train_digest: TRAIN_DIGEST.into(),
        dev_digest: DEV_DIGEST.into(),
        model_hash: identity.model_hash,
        graph_manifest_hash: graph_manifest_hash(graph_digests.into_iter()),
        scope: identity.scope,
        split: identity.split,
        seed: 5_301,
        final_update: identity.final_update,
        microbatch: identity.microbatch,
        precision: "fp32".into(),
        acquisition_seed: EVAL_SEED,
        intervention_seed: SHUFFLE_SEED,
        composition_seed: COMPOSITION_SEED,
        normal_replay_exact: replay_exact,
        shuffle_mappings,
        composition_partitions,
        records,
    })
}

/// Conditional forward-only R8 diagnostic on the preregistered primary cell.
pub fn evaluate_r8<B: Backend>(
    model: &CounterfactualRelationalLoop<B>,
    data: &V5Data,
    indices: &[usize],
    mut identity: EvaluationIdentity,
    device: &B::Device,
) -> anyhow::Result<EvaluationBundle> {
    data.require_role(crate::native_data_v2::Role::Dev)?;
    data.verify_custody()?;
    anyhow::ensure!(
        indices.len() == 750,
        "R8 diagnostic requires all 750 KQRvK M3 positions"
    );
    anyhow::ensure!(
        indices.iter().all(|&index| {
            data.position(index).family == "KQRvK" && data.position(index).mate_depth == 3
        }),
        "R8 diagnostic scope differs from KQRvK M3"
    );
    identity.scope = "KQRvK_M3_R8_conditional".into();
    let cell = prepare_cell(model, data, indices, identity.microbatch, device)?;
    let mut records = Vec::new();
    let mut graph_digests = Vec::new();
    for schedule in [Schedule::UniformFrontierV1, Schedule::BaseRankedDepthV1] {
        let graphs = &cell.graphs[schedule.id()];
        graph_digests.extend(graphs.iter().map(|graph| graph.digest.clone()));
        records.extend(
            records_for(
                model,
                data,
                &cell,
                schedule,
                graphs,
                8,
                8,
                TreatmentLabel::Normal,
                Treatment::Normal,
                None,
                identity.microbatch,
                device,
            )?
            .into_iter()
            .map(|x| x.0),
        );
    }
    Ok(EvaluationBundle {
        schema: EVAL_SCHEMA.into(),
        architecture: ARCHITECTURE.into(),
        source_sha: identity.source_sha,
        config_digest: identity.config_digest,
        train_digest: TRAIN_DIGEST.into(),
        dev_digest: DEV_DIGEST.into(),
        model_hash: identity.model_hash,
        graph_manifest_hash: graph_manifest_hash(graph_digests.into_iter()),
        scope: identity.scope,
        split: identity.split,
        seed: 5_301,
        final_update: identity.final_update,
        microbatch: identity.microbatch,
        precision: "fp32".into(),
        acquisition_seed: EVAL_SEED,
        intervention_seed: SHUFFLE_SEED,
        composition_seed: COMPOSITION_SEED,
        normal_replay_exact: true,
        shuffle_mappings: Vec::new(),
        composition_partitions: Vec::new(),
        records,
    })
}

#[cfg(test)]
mod baseline_identity_tests {
    use super::*;

    fn record(id: &str) -> BaselineRecord {
        BaselineRecord {
            position_id: id.into(),
            family: "KQRvK".into(),
            mate_depth: 3,
            legal_actions: vec![1],
            correct_indices: vec![0],
            logits: vec![0.0],
            metrics: base_metrics(&[0.0], &[0]).unwrap(),
            candidate_facts_evaluated: 2,
            candidate_facts_probe_evaluated: 1,
            candidate_facts_probe_seconds: 0.0,
            root_path_wall_seconds: 0.0,
        }
    }

    fn report() -> BaselineEvaluation {
        let mut records: Vec<_> = (0..4500)
            .map(|i| record(&format!("identity-test-{i}")))
            .collect();
        for row in records.iter_mut().skip(750) {
            row.mate_depth = 1;
        }
        BaselineEvaluation {
            schema: BASELINE_EVAL_SCHEMA.into(),
            architecture: ARCHITECTURE.into(),
            source_sha: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
            stage_a_source_sha: "d11659eca0774e0064bed0ef64ead2b725886d93".into(),
            dev_record_id_digest: crate::data::binding(crate::native_data_v2::Role::Dev)
                .unwrap()
                .record_id_digest,
            config_digest: crate::config::V5Config::default()
                .scientific_digest()
                .unwrap(),
            train_digest: TRAIN_DIGEST.into(),
            dev_digest: DEV_DIGEST.into(),
            model_hash: "identity-test-only".into(),
            baseline_fingerprint: "identity-test-only".into(),
            final_update: 1200,
            seed: 5301,
            microbatch: 2,
            precision: "fp32".into(),
            device: "cuda".into(),
            scope: "all_dev_4500".into(),
            root_encoder_examples: 4500,
            returned_encoder_examples: 0,
            exact_queries: 0,
            shared_core_applications: 0,
            records,
        }
    }

    #[test]
    fn baseline_sorted_ids_have_deterministic_distinct_membership_identity() {
        let expected = sorted_record_id_digest(["a", "b", "c"]);
        assert_eq!(expected, sorted_record_id_digest(["c", "a", "b"]));
        for ids in [
            vec!["a", "b"],
            vec!["a", "b", "c", "d"],
            vec!["a", "b", "changed"],
            vec!["a", "b", "c", "c"],
        ] {
            assert_ne!(expected, sorted_record_id_digest(ids));
        }
        let rows = vec![record("a"), record("b"), record("c")];
        assert!(validate_baseline_record_ids(&rows, &expected).is_ok());
        assert!(validate_baseline_record_ids(&rows, "wrong").is_err());
        for changed in [
            vec![record("a"), record("b")],
            vec![record("a"), record("b"), record("c"), record("d")],
            vec![record("a"), record("b"), record("changed")],
            vec![record("a"), record("b"), record("c"), record("c")],
        ] {
            assert!(validate_baseline_record_ids(&changed, &expected).is_err());
        }
    }

    #[test]
    fn baseline_v3_validates_artifact_and_record_identity_separately() {
        let binding = crate::data::binding(crate::native_data_v2::Role::Dev).unwrap();
        assert_ne!(DEV_DIGEST, binding.record_id_digest);
        let original = report();
        // Fixture IDs are deliberately not scientific DEV IDs: all preceding
        // header/metrics/alignment gates pass, but exact scientific membership refuses.
        assert_eq!(
            original.validate().unwrap_err().to_string(),
            "baseline sorted DEV identity digest mismatch"
        );
        let mut wrong = original.clone();
        wrong.dev_digest = binding.record_id_digest.clone();
        assert_eq!(
            wrong.validate().unwrap_err().to_string(),
            "final baseline identity/accounting mismatch"
        );
        let mut wrong = original.clone();
        wrong.dev_record_id_digest = DEV_DIGEST.into();
        assert_eq!(
            wrong.validate().unwrap_err().to_string(),
            "final baseline identity/accounting mismatch"
        );
        let mut wrong = original.clone();
        wrong.records[1] = wrong.records[0].clone();
        assert_eq!(
            wrong.validate().unwrap_err().to_string(),
            "duplicate or misaligned final baseline record"
        );
        let mut wrong = original.clone();
        wrong.schema = "v5_final_baseline_evaluation_v2".into();
        assert!(wrong.validate().is_err());
        let mut wrong = original.clone();
        wrong.stage_a_source_sha.clear();
        assert!(wrong.validate().is_err());
        let encoded = serde_json::to_value(&original).unwrap();
        assert_eq!(encoded["schema"], "v5_final_baseline_evaluation_v3");
        assert_ne!(encoded["source_sha"], encoded["stage_a_source_sha"]);
        assert_eq!(encoded["dev_record_id_digest"], binding.record_id_digest);
        let decoded: BaselineEvaluation = serde_json::from_value(encoded.clone()).unwrap();
        assert_eq!(serde_json::to_value(decoded).unwrap(), encoded);
        let mut missing = encoded;
        missing
            .as_object_mut()
            .unwrap()
            .remove("stage_a_source_sha");
        assert!(serde_json::from_value::<BaselineEvaluation>(missing).is_err());
    }
}
