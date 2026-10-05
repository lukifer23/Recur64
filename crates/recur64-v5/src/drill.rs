//! Disposable 24-position V5 engineering optimization drill.

use std::collections::BTreeMap;
use std::time::Instant;

use burn::module::AutodiffModule;
use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Bool, TensorData};
use recur64_model::train::adamw;
use recur64_runtime::learner::lr_at;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::{ARCHITECTURE, V5Config};
use crate::data::{FIT_DIGEST, TRAIN_DIGEST, V5Data};
use crate::graph::{AcquiredGraph, EpisodeKey, Schedule, acquire};
use crate::loss::correct_set_loss;
use crate::model::{BaseOutput, CounterfactualRelationalLoop, RootInputs, Treatment, V5Inputs};
use crate::stage::{PILOT_SEED, baseline_fingerprint};

pub const DRILL_SCHEMA: &str = "v5_engineering_drill_v2";
pub const DRILL_UPDATES: u64 = 200;
pub const DRILL_LR: f64 = 1.0e-3;
pub const DRILL_WARMUP: u64 = 20;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrillReport {
    pub schema: String,
    pub source_sha: String,
    pub architecture: String,
    pub config_digest: String,
    pub train_digest: String,
    pub fit_digest: String,
    pub seed: u64,
    pub q: usize,
    pub r: usize,
    pub updates: u64,
    pub peak_lr: f64,
    pub warmup: u64,
    pub physical_microbatch: usize,
    pub selected_position_ids: Vec<String>,
    pub selected_by_cell: BTreeMap<String, Vec<String>>,
    pub initial_mean_set_loss: f64,
    pub final_mean_set_loss: f64,
    pub relative_loss_reduction: f64,
    pub relative_logit_movement_l2: f64,
    pub action_changes: usize,
    pub examples: usize,
    pub finite_training: bool,
    pub baseline_exact: bool,
    pub initial_below_point_zero_five: bool,
    pub pass: bool,
    pub classification: String,
    pub wall_seconds: f64,
    pub graph_manifest_digests: Vec<String>,
    pub disposable_parameters_reused_by_pilot: bool,
}

/// Check the measured engineering prerequisite without treating a Q16
/// diagnostic as a Q8 success. Failed/non-finite/uninformative Q8 cannot be
/// repaired by silently selecting another report.
pub fn validated_pilot_prerequisite(
    q8: &DrillReport,
    q16: Option<&DrillReport>,
    source: &str,
    config: &str,
    microbatch: usize,
) -> anyhow::Result<()> {
    let validate = |report: &DrillReport, q| -> anyhow::Result<()> {
        let unique: std::collections::HashSet<_> = report.selected_position_ids.iter().collect();
        anyhow::ensure!(
            report.schema == DRILL_SCHEMA
                && report.architecture == ARCHITECTURE
                && report.source_sha == source
                && report.config_digest == config
                && report.train_digest == TRAIN_DIGEST
                && report.fit_digest == FIT_DIGEST
                && report.seed == PILOT_SEED
                && report.q == q
                && report.r == 4
                && report.updates == DRILL_UPDATES
                && report.peak_lr == DRILL_LR
                && report.warmup == DRILL_WARMUP
                && report.physical_microbatch == microbatch
                && report.examples == 48
                && unique.len() == 24
                && report.selected_by_cell.len() == 6
                && report.selected_by_cell.values().all(|ids| ids.len() == 4)
                && !report.disposable_parameters_reused_by_pilot,
            "drill identity/recipe mismatch"
        );
        anyhow::ensure!(
            report.finite_training
                && report.baseline_exact
                && report.initial_mean_set_loss.is_finite()
                && report.final_mean_set_loss.is_finite()
                && report.initial_mean_set_loss >= 0.05
                && !report.initial_below_point_zero_five,
            "drill is invalid, non-finite or uninformative; stop for review"
        );
        let reduction = (report.initial_mean_set_loss - report.final_mean_set_loss)
            / report.initial_mean_set_loss;
        anyhow::ensure!(
            (reduction - report.relative_loss_reduction).abs() <= 1e-12
                && report.pass == (reduction >= 0.2),
            "drill pass flag disagrees with measured losses"
        );
        Ok(())
    };
    validate(q8, 8)?;
    if q8.pass {
        anyhow::ensure!(
            q16.is_none(),
            "Q16 is not permitted after a passing Q8 drill"
        );
        return Ok(());
    }
    let diagnostic = q16.ok_or_else(|| anyhow::anyhow!("Q8 drill failed; a passing preregistered Q16 diagnostic is required before pilot training"))?;
    validate(diagnostic, 16)?;
    anyhow::ensure!(
        diagnostic.selected_position_ids == q8.selected_position_ids
            && diagnostic.selected_by_cell == q8.selected_by_cell
            && diagnostic.pass,
        "Q16 diagnostic failed or changed the fixed drill positions"
    );
    Ok(())
}

fn selection_hash(id: &str) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"recur64.v5.drill_selection.v1|");
    hash.update(id.as_bytes());
    hash.finalize().into()
}

pub fn select_positions(data: &V5Data) -> anyhow::Result<Vec<usize>> {
    data.require_role(crate::native_data_v2::Role::Train)?;
    let mut cells: BTreeMap<(String, u8), Vec<usize>> = BTreeMap::new();
    for &index in &data.fit {
        let position = data.position(index);
        if matches!(position.family.as_str(), "KQRvK" | "KRRvK")
            && matches!(position.mate_depth, 1..=3)
        {
            cells
                .entry((position.family.clone(), position.mate_depth))
                .or_default()
                .push(index);
        }
    }
    anyhow::ensure!(
        cells.len() == 6,
        "drill requires the six family/depth cells"
    );
    let mut selected = Vec::with_capacity(24);
    for ((family, depth), mut indices) in cells {
        anyhow::ensure!(
            indices.len() >= 4,
            "{family} M{depth} has fewer than four FIT positions"
        );
        indices.sort_by_key(|&index| selection_hash(&data.position(index).id));
        selected.extend_from_slice(&indices[..4]);
    }
    Ok(selected)
}

fn correct_mask<B: Backend>(
    data: &V5Data,
    indices: &[usize],
    width: usize,
    device: &B::Device,
) -> anyhow::Result<Tensor<B, 2, Bool>> {
    let mut mask = vec![false; indices.len() * width];
    for (row, &index) in indices.iter().enumerate() {
        for &correct in &data.position(index).correct {
            anyhow::ensure!((correct as usize) < data.position(index).legal.len());
            mask[row * width + correct as usize] = true;
        }
    }
    Ok(Tensor::from_data(
        TensorData::new(mask, [indices.len(), width]),
        device,
    ))
}

fn scalar<B: Backend>(value: Tensor<B, 1>) -> f64 {
    f64::from(value.into_data().to_vec::<f32>().expect("f32 scalar")[0])
}

fn eval_logits<B: AutodiffBackend>(
    model: &CounterfactualRelationalLoop<B>,
    data: &V5Data,
    examples: &[(usize, recur64_core::GameState, AcquiredGraph)],
    microbatch: usize,
    device: &B::Device,
) -> anyhow::Result<(f64, Vec<Vec<f32>>)> {
    let inner = model.valid();
    let mut loss = 0.0;
    let mut logits = Vec::with_capacity(examples.len());
    for chunk in examples.chunks(microbatch) {
        let roots: Vec<&recur64_core::GameState> = chunk.iter().map(|x| &x.1).collect();
        let root_input = RootInputs::<B::InnerBackend>::from_roots(&roots, device)?;
        let base_inner = inner.base_root(&root_input);
        let paired: Vec<(&recur64_core::GameState, &AcquiredGraph)> =
            chunk.iter().map(|x| (&x.1, &x.2)).collect();
        let input = V5Inputs::<B>::from_examples(&paired, device)?;
        let base = BaseOutput {
            context: Tensor::from_inner(base_inner.context),
            pooled: Tensor::from_inner(base_inner.pooled),
            hypotheses: Tensor::from_inner(base_inner.hypotheses),
            z0: Tensor::from_inner(base_inner.z0),
        };
        let output = model.paired_with_base(&input, base, 4, Treatment::Normal);
        let width = input.cands.width;
        let correct = correct_mask::<B>(
            data,
            &chunk.iter().map(|x| x.0).collect::<Vec<_>>(),
            width,
            device,
        )?;
        loss += scalar(correct_set_loss(
            output.logits.clone(),
            input.cands.mask.clone(),
            correct,
        )) * chunk.len() as f64;
        let host = output.logits.into_data().to_vec::<f32>()?;
        for (row, (index, _, _)) in chunk.iter().enumerate() {
            let legal = data.position(*index).legal.len();
            logits.push(host[row * width..row * width + legal].to_vec());
        }
    }
    Ok((loss / examples.len() as f64, logits))
}

pub fn run<B: AutodiffBackend>(
    source_sha: String,
    data: &V5Data,
    q: usize,
    microbatch: usize,
    device: &B::Device,
) -> anyhow::Result<DrillReport> {
    anyhow::ensure!(
        matches!(q, 8 | 16),
        "drill Q must be 8 or the conditional 16"
    );
    anyhow::ensure!(
        matches!(microbatch, 1 | 2),
        "drill microbatch must be 1 or 2"
    );
    data.require_role(crate::native_data_v2::Role::Train)?;
    data.verify_custody()?;
    let selected = select_positions(data)?;
    <B as Backend>::seed(device, PILOT_SEED);
    let mut model = CounterfactualRelationalLoop::<B>::new(V5Config::default(), device);
    let baseline_before = baseline_fingerprint(&model, device)?;
    let inner = model.valid();
    let roots = data.roots(&selected)?;
    let mut b0 = Vec::with_capacity(selected.len());
    for root_chunk in roots.chunks(microbatch) {
        let refs: Vec<&recur64_core::GameState> = root_chunk.iter().collect();
        let input = RootInputs::<B::InnerBackend>::from_roots(&refs, device)?;
        let width = input.cands.width;
        let values = inner.base_root(&input).z0.into_data().to_vec::<f32>()?;
        for (row, root) in root_chunk.iter().enumerate() {
            b0.push(values[row * width..row * width + root.legal_actions().len()].to_vec());
        }
    }
    let mut examples = Vec::with_capacity(48);
    let mut graph_digests = Vec::with_capacity(48);
    for (row, &index) in selected.iter().enumerate() {
        data.validate_root_alignment(index, &roots[row])?;
        for schedule in [Schedule::UniformFrontierV1, Schedule::BaseRankedDepthV1] {
            let graph = acquire(
                &roots[row],
                EpisodeKey {
                    position_id: data.position(index).id.clone(),
                    schedule,
                    run_seed: PILOT_SEED,
                    occurrence_ordinal: 0,
                },
                q,
                (schedule == Schedule::BaseRankedDepthV1).then_some(b0[row].as_slice()),
            )?;
            graph_digests.push(graph.digest.clone());
            examples.push((index, roots[row].clone(), graph));
        }
    }
    let started = Instant::now();
    let (initial_loss, initial_logits) = eval_logits(&model, data, &examples, microbatch, device)?;
    anyhow::ensure!(initial_loss.is_finite(), "non-finite initial drill loss");
    let mut optim = adamw::<B, CounterfactualRelationalLoop<B>>();
    let mut finite = true;
    for update in 0..DRILL_UPDATES {
        let lr = lr_at(update, DRILL_LR, DRILL_WARMUP, DRILL_UPDATES);
        let mut accumulator = GradientsAccumulator::<CounterfactualRelationalLoop<B>>::new();
        let inner = model.valid();
        let mut update_loss = 0.0;
        for chunk in examples.chunks(microbatch) {
            let roots: Vec<&recur64_core::GameState> = chunk.iter().map(|x| &x.1).collect();
            let root_input = RootInputs::<B::InnerBackend>::from_roots(&roots, device)?;
            let base_inner = inner.base_root(&root_input);
            let paired: Vec<(&recur64_core::GameState, &AcquiredGraph)> =
                chunk.iter().map(|x| (&x.1, &x.2)).collect();
            let input = V5Inputs::<B>::from_examples(&paired, device)?;
            let base = BaseOutput {
                context: Tensor::from_inner(base_inner.context),
                pooled: Tensor::from_inner(base_inner.pooled),
                hypotheses: Tensor::from_inner(base_inner.hypotheses),
                z0: Tensor::from_inner(base_inner.z0),
            };
            let output = model.paired_with_base(&input, base, 4, Treatment::Normal);
            let indices: Vec<usize> = chunk.iter().map(|x| x.0).collect();
            let correct = correct_mask::<B>(data, &indices, input.cands.width, device)?;
            let loss = correct_set_loss(output.logits, input.cands.mask, correct)
                .mul_scalar(chunk.len() as f32 / examples.len() as f32);
            update_loss += scalar(loss.clone());
            accumulator.accumulate(&model, GradientsParams::from_grads(loss.backward(), &model));
        }
        finite &= update_loss.is_finite();
        anyhow::ensure!(finite, "non-finite drill loss at update {update}");
        model = optim.step(lr, model, accumulator.grads());
    }
    let (final_loss, final_logits) = eval_logits(&model, data, &examples, microbatch, device)?;
    finite &= final_loss.is_finite();
    let baseline_after = baseline_fingerprint(&model, device)?;
    let mut movement_sq = 0.0;
    let mut initial_sq = 0.0;
    let mut action_changes = 0;
    for (before, after) in initial_logits.iter().zip(&final_logits) {
        for (&a, &b) in before.iter().zip(after) {
            movement_sq += f64::from(b - a).powi(2);
            initial_sq += f64::from(a).powi(2);
        }
        let argmax = |x: &[f32]| {
            x.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                .map(|x| x.0)
                .expect("nonempty drill logits")
        };
        action_changes += usize::from(argmax(before) != argmax(after));
    }
    let relative_reduction = (initial_loss - final_loss) / initial_loss;
    let initial_low = initial_loss < 0.05;
    let pass =
        finite && !initial_low && relative_reduction >= 0.20 && baseline_before == baseline_after;
    let classification = if initial_low {
        "UNINFORMATIVE_REVIEW"
    } else if pass {
        "PASS"
    } else {
        "FAIL"
    };
    let mut selected_by_cell: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for &index in &selected {
        let position = data.position(index);
        selected_by_cell
            .entry(format!("{}_M{}", position.family, position.mate_depth))
            .or_default()
            .push(position.id.clone());
    }
    Ok(DrillReport {
        schema: DRILL_SCHEMA.into(),
        source_sha,
        architecture: ARCHITECTURE.into(),
        config_digest: V5Config::default().scientific_digest()?,
        train_digest: TRAIN_DIGEST.into(),
        fit_digest: FIT_DIGEST.into(),
        seed: PILOT_SEED,
        q,
        r: 4,
        updates: DRILL_UPDATES,
        peak_lr: DRILL_LR,
        warmup: DRILL_WARMUP,
        physical_microbatch: microbatch,
        selected_position_ids: selected
            .iter()
            .map(|&i| data.position(i).id.clone())
            .collect(),
        selected_by_cell,
        initial_mean_set_loss: initial_loss,
        final_mean_set_loss: final_loss,
        relative_loss_reduction: relative_reduction,
        relative_logit_movement_l2: movement_sq.sqrt() / initial_sq.sqrt().max(1.0e-12),
        action_changes,
        examples: examples.len(),
        finite_training: finite,
        baseline_exact: baseline_before == baseline_after,
        initial_below_point_zero_five: initial_low,
        pass,
        classification: classification.into(),
        wall_seconds: started.elapsed().as_secs_f64(),
        graph_manifest_digests: graph_digests,
        disposable_parameters_reused_by_pilot: false,
    })
}

#[cfg(test)]
mod prerequisite_tests {
    use super::*;

    // Isolated report-interface fixtures, never chess measurements.
    fn report(q: usize, final_loss: f64) -> DrillReport {
        let mut by_cell = BTreeMap::new();
        for family in ["KQRvK", "KRRvK"] {
            for depth in 1..=3 {
                by_cell.insert(
                    format!("{family}-M{depth}"),
                    (0..4)
                        .map(|i| format!("interface-{family}-{depth}-{i}"))
                        .collect::<Vec<_>>(),
                );
            }
        }
        let ids = by_cell.values().flatten().cloned().collect();
        DrillReport {
            schema: DRILL_SCHEMA.into(),
            source_sha: "interface-source".into(),
            architecture: ARCHITECTURE.into(),
            config_digest: "interface-config".into(),
            train_digest: TRAIN_DIGEST.into(),
            fit_digest: FIT_DIGEST.into(),
            seed: PILOT_SEED,
            q,
            r: 4,
            updates: DRILL_UPDATES,
            peak_lr: DRILL_LR,
            warmup: DRILL_WARMUP,
            physical_microbatch: 2,
            selected_position_ids: ids,
            selected_by_cell: by_cell,
            initial_mean_set_loss: 1.0,
            final_mean_set_loss: final_loss,
            relative_loss_reduction: 1.0 - final_loss,
            relative_logit_movement_l2: 0.0,
            action_changes: 0,
            examples: 48,
            finite_training: true,
            baseline_exact: true,
            initial_below_point_zero_five: false,
            pass: final_loss <= 0.8,
            classification: "interface-fixture".into(),
            wall_seconds: 0.0,
            graph_manifest_digests: Vec::new(),
            disposable_parameters_reused_by_pilot: false,
        }
    }

    fn validate(q8: &DrillReport, q16: Option<&DrillReport>) -> anyhow::Result<()> {
        validated_pilot_prerequisite(q8, q16, "interface-source", "interface-config", 2)
    }

    #[test]
    fn q8_success_or_the_single_matched_q16_diagnostic_can_qualify() {
        let passing = report(8, 0.7);
        let failed = report(8, 0.9);
        let diagnostic = report(16, 0.7);
        assert!(validate(&passing, None).is_ok());
        assert!(validate(&failed, None).is_err());
        assert!(validate(&failed, Some(&diagnostic)).is_ok());
        assert!(validate(&passing, Some(&diagnostic)).is_err());
        assert!(validate(&failed, Some(&report(16, 0.9))).is_err());
        let mut changed = diagnostic;
        changed.selected_position_ids[0] = "different-position".into();
        assert!(validate(&failed, Some(&changed)).is_err());
    }

    #[test]
    fn invalid_uninformative_and_forged_reports_cannot_unlock_training() {
        for mutate in 0..5 {
            let mut input = report(8, 0.7);
            match mutate {
                0 => input.finite_training = false,
                1 => input.baseline_exact = false,
                2 => input.initial_mean_set_loss = 0.04,
                3 => input.relative_loss_reduction = 0.9,
                _ => input.pass = false,
            }
            assert!(validate(&input, None).is_err());
        }
    }
}
