//! CPU/CUDA qualification for the actual paired V5 graph.

pub mod diagnostic;

use std::path::PathBuf;
use std::time::Instant;

use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Record, Recorder};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Bool, TensorData};
use recur64_model::active::coverage::{CoverageRow, gradient_coverage};
use recur64_model::train::adamw;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::config::V5Config;
use crate::graph::{AcquiredGraph, EpisodeKey, Schedule, acquire};
use crate::loss::correct_set_loss;
use crate::model::{CounterfactualRelationalLoop, Treatment, V5Inputs};
use crate::profile::{PhaseTiming, SynchronizedProfile};

#[derive(Debug, Clone, Serialize)]
pub struct ExecutionAccounting {
    pub requested_q_per_position: usize,
    pub actual_q: Vec<usize>,
    pub exhausted_frontier: Vec<bool>,
    pub maximum_depth: Vec<u8>,
    pub root_branches: Vec<usize>,
    pub graph_digests: Vec<String>,
    pub root_encoder_examples: usize,
    pub returned_encoder_examples: usize,
    pub returned_encoder_physical_rows: usize,
    pub returned_encoder_padding_rows: usize,
    pub legal_candidates: Vec<usize>,
    pub candidate_physical_rows: usize,
    pub root_candidate_facts_calls: usize,
    pub root_candidate_facts_successor_boards: usize,
    pub raw_packet_preparations: usize,
    pub core_applications_per_example: usize,
    pub core_applications_whole_batch: usize,
}

fn accounting(
    roots: &[recur64_core::GameState],
    graphs: &[AcquiredGraph],
    r: usize,
) -> ExecutionAccounting {
    let batch = roots.len();
    let legal: Vec<_> = roots
        .iter()
        .map(|root| root.legal_actions().len())
        .collect();
    let actual: usize = graphs.iter().map(|graph| graph.actual_q).sum();
    let physical = batch * graphs.iter().map(|graph| graph.actual_q).max().unwrap_or(0);
    ExecutionAccounting {
        requested_q_per_position: graphs[0].requested_q,
        actual_q: graphs.iter().map(|graph| graph.actual_q).collect(),
        exhausted_frontier: graphs
            .iter()
            .map(|graph| graph.exhausted_frontier)
            .collect(),
        maximum_depth: graphs
            .iter()
            .map(|graph| graph.nodes.iter().map(|node| node.depth).max().unwrap_or(0))
            .collect(),
        root_branches: graphs
            .iter()
            .map(|graph| {
                graph
                    .nodes
                    .iter()
                    .map(|node| node.root_candidate)
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
            })
            .collect(),
        graph_digests: graphs.iter().map(|graph| graph.digest.clone()).collect(),
        root_encoder_examples: batch,
        returned_encoder_examples: actual,
        returned_encoder_physical_rows: physical,
        returned_encoder_padding_rows: physical - actual,
        candidate_physical_rows: batch * legal.iter().copied().max().unwrap_or(0),
        root_candidate_facts_calls: 2 * batch,
        root_candidate_facts_successor_boards: 2 * legal.iter().sum::<usize>(),
        legal_candidates: legal,
        // Current graph-free baseline constructs its own raw inputs. This is
        // duplicated HOST/UPLOAD work, NOT a second returned-state encoding.
        raw_packet_preparations: 2 * actual,
        core_applications_per_example: 4 * r,
        core_applications_whole_batch: 4 * r * batch,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ShapeTiming {
    pub q: usize,
    pub r: usize,
    pub cold_seconds: f64,
    pub warm_seconds: f64,
    pub loss: f64,
    pub core_applications_per_example: usize,
    pub accounting: ExecutionAccounting,
}

#[derive(Debug, Clone, Serialize)]
pub struct QualificationReport {
    pub schema: String,
    pub source_sha: String,
    pub pass: bool,
    pub device: String,
    pub precision: String,
    pub microbatch: usize,
    pub architecture: String,
    pub config_digest: String,
    pub parameters: usize,
    pub parameter_bytes_fp32: usize,
    pub query_seconds: f64,
    pub qualification_wall_seconds: f64,
    pub timing_contract: String,
    pub timing_contract_digest: String,
    pub synchronized_profile_q8_r4: Vec<PhaseTiming>,
    pub synchronized_profile_wall_seconds: f64,
    pub synchronized_profile_accounting: ExecutionAccounting,
    pub checkpoint_profile: Vec<PhaseTiming>,
    pub profile_outputs_and_all_gradients_exact: bool,
    pub profile_adamw_parameters_and_moments_exact: bool,
    pub profile_phase_accounting_consistent: bool,
    pub matrix: Vec<ShapeTiming>,
    pub worst_update_warm_seconds: f64,
    pub repeated_updates: usize,
    pub repeated_update_first_seconds: f64,
    pub repeated_update_last_seconds: f64,
    pub null_centered_max_abs: f32,
    pub null_tolerance: f32,
    pub payload_input_gradient_l2: f32,
    pub gradient_groups: Vec<String>,
    pub gradient_coverage: Vec<CoverageRow>,
    pub repeated_update_seconds: Vec<f64>,
    pub repeated_device_used_mib: Vec<Option<u64>>,
    pub baseline_exact_after_reader_updates: bool,
    pub baseline_parameters_exact_after_reader_updates: bool,
    pub checkpoint_restore_exact: bool,
    pub optimizer_moments_restore_exact: bool,
    pub resumed_update_parameters_exact: bool,
    pub resumed_update_moments_exact: bool,
    pub graph_free_baseline_matches_reference: bool,
    pub r8_forward_seconds: f64,
    pub detected_vram_mib: Option<u64>,
    pub device_used_baseline_mib: Option<u64>,
    pub device_used_peak_mib: Option<u64>,
    pub device_used_peak_delta_mib: Option<u64>,
    pub requested_q: Vec<usize>,
    pub requested_r: Vec<usize>,
    pub notes: Vec<String>,
}

type Opt<B> = OptimizerAdaptor<AdamW, CounterfactualRelationalLoop<B>, B>;

fn gpu_memory() -> Option<(u64, u64)> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let line = String::from_utf8(output.stdout).ok()?;
    let mut values = line.lines().next()?.split(',').map(str::trim);
    Some((values.next()?.parse().ok()?, values.next()?.parse().ok()?))
}

fn roots(count: usize) -> Vec<recur64_core::GameState> {
    let fixtures = [
        "6k1/8/8/8/8/8/4Q3/3RK3 w - - 0 1",
        "3rk3/4q3/8/8/8/8/8/6K1 b - - 0 1",
    ];
    (0..count)
        .map(|index| {
            recur64_core::GameState::from_fen(fixtures[index % fixtures.len()])
                .expect("qualified real chess fixture")
        })
        .collect()
}

fn graphs(roots: &[recur64_core::GameState], q: usize) -> anyhow::Result<Vec<AcquiredGraph>> {
    roots
        .iter()
        .enumerate()
        .map(|(index, root)| {
            acquire(
                root,
                EpisodeKey {
                    position_id: format!("qualification-{index}"),
                    schedule: Schedule::UniformFrontierV1,
                    run_seed: 0x5301_0A11,
                    occurrence_ordinal: index as u64,
                },
                q,
                None,
            )
        })
        .collect()
}

fn first_legal_correct<B: Backend>(input: &V5Inputs<B>, device: &B::Device) -> Tensor<B, 2, Bool> {
    let mut data = vec![false; input.batch * input.cands.width];
    for row in 0..input.batch {
        data[row * input.cands.width] = true;
    }
    Tensor::from_data(
        TensorData::new(data, [input.batch, input.cands.width]),
        device,
    )
}

fn max_abs(values: Tensor<impl Backend, 2>) -> f32 {
    values
        .into_data()
        .to_vec::<f32>()
        .expect("f32")
        .into_iter()
        .map(f32::abs)
        .fold(0.0, f32::max)
}

fn logits<B: AutodiffBackend>(
    model: &CounterfactualRelationalLoop<B>,
    roots: &[recur64_core::GameState],
    graphs: &[AcquiredGraph],
    r: usize,
    device: &B::Device,
) -> anyhow::Result<Vec<f32>> {
    let examples: Vec<_> = roots.iter().zip(graphs).collect();
    let input = V5Inputs::<B>::from_examples(&examples, device)?;
    let base = model.base_frozen(&examples, device)?;
    Ok(model
        .paired_with_base(&input, base, r, Treatment::Normal)
        .logits
        .into_data()
        .to_vec()?)
}

fn update<B: AutodiffBackend>(
    model: CounterfactualRelationalLoop<B>,
    optim: &mut Opt<B>,
    roots: &[recur64_core::GameState],
    graphs: &[AcquiredGraph],
    r: usize,
    device: &B::Device,
    track_input: bool,
) -> anyhow::Result<(CounterfactualRelationalLoop<B>, f64, Vec<CoverageRow>, f32)> {
    update_observed(
        model,
        optim,
        roots,
        graphs,
        r,
        device,
        track_input,
        &mut |_| {},
    )
}

#[allow(clippy::too_many_arguments)]
fn update_observed<B: AutodiffBackend>(
    mut model: CounterfactualRelationalLoop<B>,
    optim: &mut Opt<B>,
    roots: &[recur64_core::GameState],
    graphs: &[AcquiredGraph],
    r: usize,
    device: &B::Device,
    track_input: bool,
    phase: &mut dyn FnMut(&str),
) -> anyhow::Result<(CounterfactualRelationalLoop<B>, f64, Vec<CoverageRow>, f32)> {
    let examples: Vec<_> = roots.iter().zip(graphs).collect();
    let mut input = V5Inputs::<B>::from_examples_profiled(&examples, device, phase)?;
    let tracked = track_input.then(|| input.states.clone().require_grad());
    if let Some(value) = &tracked {
        input.states = value.clone();
    }
    phase("input_gradient_tracking");
    let base = model.base_frozen_profiled(&examples, device, phase)?;
    let output = model.paired_profiled(&input, base, r, Treatment::Normal, None, phase);
    let correct = first_legal_correct(&input, device);
    let loss = correct_set_loss(output.logits, input.cands.mask.clone(), correct);
    let loss_value = f64::from(loss.clone().into_data().to_vec::<f32>()?[0]);
    anyhow::ensure!(loss_value.is_finite(), "qualification loss is non-finite");
    phase("target_loss_and_scalar_readback");
    let raw = loss.backward();
    phase("backward_both_streams");
    let input_norm = tracked
        .and_then(|value| value.grad(&raw))
        .map(|gradient| {
            gradient
                .into_data()
                .to_vec::<f32>()
                .expect("f32")
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt()
        })
        .unwrap_or(0.0);
    let grads = GradientsParams::from_grads(raw, &model);
    let coverage = gradient_coverage::<B, _>(&model, &grads);
    anyhow::ensure!(
        coverage.iter().all(|row| row.finite),
        "qualification gradient is non-finite"
    );
    phase("gradient_health_and_input_readback");
    model = optim.step(3.0e-4, model, grads);
    B::sync(device)
        .map_err(|error| anyhow::anyhow!("post-AdamW completion fence failed: {error:?}"))?;
    phase("adamw_and_completion_fence");
    Ok((model, loss_value, coverage, input_norm))
}

fn checkpoint_roundtrip<B: AutodiffBackend>(
    model: &CounterfactualRelationalLoop<B>,
    optim: &Opt<B>,
    device: &B::Device,
    roots: &[recur64_core::GameState],
    graphs: &[AcquiredGraph],
    phase: &mut dyn FnMut(&str),
) -> anyhow::Result<(bool, bool, bool, bool)> {
    let dir: PathBuf = std::env::temp_dir().join(format!(
        "recur64-v5-qualification-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    // Exclusive creation: never erase an interrupted/invalid attempt.
    std::fs::create_dir(&dir)?;
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model.clone().save_file(dir.join("model"), &recorder)?;
    recorder.record(optim.to_record(), dir.join("optimizer"))?;
    phase("checkpoint_save_model_and_optimizer");
    let template = CounterfactualRelationalLoop::<B>::new(V5Config::default(), device);
    let loaded = template.load_file(dir.join("model"), &recorder, device)?;
    let record = recorder.load(dir.join("optimizer"), device)?;
    let mut loaded_optim = adamw::<B, CounterfactualRelationalLoop<B>>().load_record(record);
    phase("checkpoint_load_model_and_optimizer");
    let equal = model.parameter_digest()? == loaded.parameter_digest()?
        && logits(model, roots, graphs, 4, device)? == logits(&loaded, roots, graphs, 4, device)?;
    let moments_equal = optimizer_digest(optim)? == optimizer_digest(&loaded_optim)?;
    phase("checkpoint_full_contents_and_forward_verification");
    let mut continued_optim = optim.clone();
    let (continued, _, _, _) = update(
        model.clone(),
        &mut continued_optim,
        roots,
        graphs,
        4,
        device,
        false,
    )?;
    let (resumed, _, _, _) = update(loaded, &mut loaded_optim, roots, graphs, 4, device, false)?;
    let continued_equal = continued.parameter_digest()? == resumed.parameter_digest()?;
    let continued_moments_equal =
        optimizer_digest(&continued_optim)? == optimizer_digest(&loaded_optim)?;
    phase("checkpoint_two_continuations_and_full_contents_verification");
    // Only this successfully verified, uniquely created temporary artifact is
    // disposable. Resolve and constrain its target before recursive removal.
    if equal && moments_equal && continued_equal && continued_moments_equal {
        let resolved = dir.canonicalize()?;
        let temp_parent = std::env::temp_dir().canonicalize()?;
        anyhow::ensure!(
            resolved.parent() == Some(temp_parent.as_path()),
            "checkpoint cleanup escaped the temporary directory"
        );
        std::fs::remove_dir_all(&resolved)?;
    } else {
        eprintln!(
            "V5 failed checkpoint qualification retained at {}",
            dir.display()
        );
    }
    Ok((
        equal,
        moments_equal,
        continued_equal,
        continued_moments_equal,
    ))
}

/// Hash every optimizer moment and counter in stable ParamId order. Streaming
/// serialization avoids materializing a second JSON copy of the FP32 moments.
fn optimizer_digest<B: AutodiffBackend>(optim: &Opt<B>) -> anyhow::Result<String> {
    struct HashWriter(Sha256);
    impl std::io::Write for HashWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.update(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let item = optim.to_record().into_item::<FullPrecisionSettings>();
    let sorted: std::collections::BTreeMap<_, _> = item.into_iter().collect();
    let mut writer = HashWriter(Sha256::new());
    serde_json::to_writer(&mut writer, &sorted)?;
    Ok(format!("{:x}", writer.0.finalize()))
}

/// Hash every gradient component (and absent-gradient marker), not merely one
/// changed parameter or a norm. Same cloned model means identical ParamIds.
fn gradient_digest<B: AutodiffBackend>(
    model: &CounterfactualRelationalLoop<B>,
    grads: &GradientsParams,
) -> anyhow::Result<String> {
    use burn::module::{ModuleVisitor, Param};
    struct Visitor<'a, B: AutodiffBackend> {
        grads: &'a GradientsParams,
        hash: Sha256,
        error: Option<String>,
        marker: std::marker::PhantomData<B>,
    }
    impl<B: AutodiffBackend> ModuleVisitor<B> for Visitor<'_, B> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            self.hash.update(param.id.to_string().as_bytes());
            match self.grads.get::<B::InnerBackend, D>(param.id) {
                None => self.hash.update([0]),
                Some(gradient) => {
                    self.hash.update([1]);
                    let data = gradient.into_data();
                    for dim in data.shape.dims::<D>() {
                        self.hash.update((dim as u64).to_le_bytes());
                    }
                    match data.to_vec::<f32>() {
                        Ok(values) => {
                            for value in values {
                                self.hash.update(value.to_bits().to_le_bytes());
                            }
                        }
                        Err(error) => {
                            self.error = Some(format!("gradient readback failed: {error:?}"))
                        }
                    }
                }
            }
        }
    }
    let mut visitor = Visitor::<B> {
        grads,
        hash: Sha256::new(),
        error: None,
        marker: std::marker::PhantomData,
    };
    model.visit(&mut visitor);
    anyhow::ensure!(
        visitor.error.is_none(),
        "{}",
        visitor.error.unwrap_or_default()
    );
    Ok(format!("{:x}", visitor.hash.finalize()))
}

/// Same weights, graph and existing AdamW state on this device. The instrumented
/// arm changes completion fences ONLY. This is not CPU/CUDA parity.
fn profile_parity<B: AutodiffBackend>(
    model: &CounterfactualRelationalLoop<B>,
    optim: &Opt<B>,
    roots: &[recur64_core::GameState],
    graphs: &[AcquiredGraph],
    r: usize,
    device: &B::Device,
) -> anyhow::Result<(bool, bool)> {
    let run = |observed: bool| -> anyhow::Result<_> {
        let mut profile = SynchronizedProfile::<B>::new(device)?;
        let mut phase = |name: &str| {
            if observed {
                profile.mark(name);
            }
        };
        let examples: Vec<_> = roots.iter().zip(graphs).collect();
        let mut input = V5Inputs::<B>::from_examples_profiled(&examples, device, &mut phase)?;
        let tracked = input.states.clone().require_grad();
        input.states = tracked.clone();
        let base = model.base_frozen_profiled(&examples, device, &mut phase)?;
        let out = model.paired_profiled(&input, base, r, Treatment::Normal, None, &mut phase);
        let logits = out.logits.clone().into_data();
        let centered = out.centered_delta.into_data();
        let loss = correct_set_loss(
            out.logits,
            input.cands.mask.clone(),
            first_legal_correct(&input, device),
        );
        let value = loss.clone().into_data();
        let raw = loss.backward();
        phase("parity_backward");
        let payload_gradient = tracked
            .grad(&raw)
            .ok_or_else(|| anyhow::anyhow!("profile parity missing payload gradient"))?
            .into_data();
        let gradients = GradientsParams::from_grads(raw, model);
        let gradient_hash = gradient_digest::<B>(model, &gradients)?;
        let mut optimizer = optim.clone();
        let next = optimizer.step(3.0e-4, model.clone(), gradients);
        B::sync(device)
            .map_err(|error| anyhow::anyhow!("profile parity AdamW fence failed: {error:?}"))?;
        phase("parity_adamw");
        profile.finish()?;
        Ok((
            (logits, centered, value, payload_gradient, gradient_hash),
            (next.parameter_digest()?, optimizer_digest(&optimizer)?),
        ))
    };
    let normal = run(false)?;
    let profiled = run(true)?;
    Ok((normal.0 == profiled.0, normal.1 == profiled.1))
}

pub fn run<B>(
    source_sha: &str,
    device_label: &str,
    microbatch: usize,
    device: &B::Device,
) -> anyhow::Result<QualificationReport>
where
    B: AutodiffBackend,
{
    let qualification_start = Instant::now();
    anyhow::ensure!(!source_sha.is_empty(), "qualification source SHA is empty");
    crate::graph::validate_uniform_frontier_contract()?;
    anyhow::ensure!(
        matches!(microbatch, 1 | 2),
        "qualification microbatch must be 1 or 2"
    );
    <B as Backend>::seed(device, 5301);
    let cfg = V5Config::default();
    let mut model = CounterfactualRelationalLoop::<B>::new(cfg.clone(), device);
    anyhow::ensure!(
        model.num_params() <= 10_000_000,
        "V5 exceeds 10M parameters"
    );
    let mut optim = adamw::<B, CounterfactualRelationalLoop<B>>();
    let roots = roots(microbatch);
    let query_started = Instant::now();
    let q8 = graphs(&roots, 8)?;
    let query_seconds = query_started.elapsed().as_secs_f64();
    let base_before = crate::stage::baseline_fingerprint(&model, device)?;
    let base_parameters_before = model.baseline_parameter_digest()?;
    let examples: Vec<_> = roots.iter().zip(&q8).collect();
    let graph_free_baseline_matches_reference = {
        let reference_input = V5Inputs::<B>::from_examples(&examples, device)?;
        let reference = model.base(&reference_input);
        let frozen = model.base_frozen(&examples, device)?;
        reference.context.into_data() == frozen.context.into_data()
            && reference.hypotheses.into_data() == frozen.hypotheses.into_data()
            && reference.z0.into_data() == frozen.z0.into_data()
    };
    let mut matrix = Vec::new();
    let mut gradient_groups = Vec::new();
    let mut tracked_coverage = Vec::new();
    let mut payload_input_gradient_l2 = 0.0;
    let memory_baseline = (device_label == "cuda").then(gpu_memory).flatten();
    let mut memory_peak = memory_baseline.map(|value| value.1);

    for q in [2, 4, 8] {
        let prefixed: Vec<_> = q8
            .iter()
            .map(|graph| graph.prefix(q))
            .collect::<anyhow::Result<_>>()?;
        for r in [1, 2, 4] {
            let cold = Instant::now();
            let (next, _, _, _) = update(model, &mut optim, &roots, &prefixed, r, device, false)?;
            model = next;
            if device_label == "cuda"
                && let Some((_, used)) = gpu_memory()
            {
                memory_peak = Some(memory_peak.unwrap_or(used).max(used));
            }
            let cold_seconds = cold.elapsed().as_secs_f64();
            let warm = Instant::now();
            let track = q == 8 && r == 4;
            let (next, loss, coverage, input_norm) =
                update(model, &mut optim, &roots, &prefixed, r, device, track)?;
            model = next;
            let warm_seconds = warm.elapsed().as_secs_f64();
            if track {
                tracked_coverage = coverage.clone();
                payload_input_gradient_l2 = input_norm;
                for prefix in ["state.", "evidence.", "hypothesis.", "correction_"] {
                    if coverage
                        .iter()
                        .any(|row| row.name.starts_with(prefix) && row.finite && row.nonzero)
                    {
                        gradient_groups.push(prefix.trim_end_matches('.').into());
                    }
                }
                anyhow::ensure!(
                    coverage
                        .iter()
                        .filter(|row| row.name.starts_with("root."))
                        .all(|row| !row.has_grad),
                    "graph-free baseline received a Stage B gradient"
                );
            }
            matrix.push(ShapeTiming {
                q,
                r,
                cold_seconds,
                warm_seconds,
                loss,
                core_applications_per_example: 4 * r,
                accounting: accounting(&roots, &prefixed, r),
            });
        }
    }

    let repeated_first;
    let mut repeated_update_seconds = Vec::with_capacity(50);
    let mut repeated_device_used_mib = Vec::with_capacity(50);
    let mut repeated_last = 0.0;
    let mut worst_warm: f64 = 0.0;
    {
        let start = Instant::now();
        let (next, _, _, _) = update(model, &mut optim, &roots, &q8, 4, device, false)?;
        model = next;
        let sampled = if device_label == "cuda" {
            gpu_memory().map(|(_, used)| used)
        } else {
            None
        };
        if let Some(used) = sampled {
            memory_peak = Some(memory_peak.unwrap_or(used).max(used));
        }
        repeated_first = start.elapsed().as_secs_f64();
        repeated_update_seconds.push(repeated_first);
        repeated_device_used_mib.push(sampled);
        worst_warm = worst_warm.max(repeated_first);
    }
    for _ in 1..50 {
        let start = Instant::now();
        let (next, _, _, _) = update(model, &mut optim, &roots, &q8, 4, device, false)?;
        model = next;
        let sampled = if device_label == "cuda" {
            gpu_memory().map(|(_, used)| used)
        } else {
            None
        };
        if let Some(used) = sampled {
            memory_peak = Some(memory_peak.unwrap_or(used).max(used));
        }
        repeated_last = start.elapsed().as_secs_f64();
        repeated_update_seconds.push(repeated_last);
        repeated_device_used_mib.push(sampled);
        worst_warm = worst_warm.max(repeated_last);
    }
    for timing in &matrix {
        worst_warm = worst_warm.max(timing.warm_seconds);
    }
    // Dedicated synchronized profiling update uses disposable clones. It does
    // not alter the resident update sequence or pilot initialization.
    let (profile_outputs_and_all_gradients_exact, profile_adamw_parameters_and_moments_exact) =
        profile_parity(&model, &optim, &roots, &q8, 4, device)?;
    let mut profile = SynchronizedProfile::<B>::new(device)?;
    let profile_start = Instant::now();
    let mut profile_optim = optim.clone();
    let _ = update_observed(
        model.clone(),
        &mut profile_optim,
        &roots,
        &q8,
        4,
        device,
        true,
        &mut |name| profile.mark(name),
    )?;
    let synchronized_profile_wall_seconds = profile_start.elapsed().as_secs_f64();
    let synchronized_profile_q8_r4 = profile.finish()?;
    let base_after = crate::stage::baseline_fingerprint(&model, device)?;
    let base_parameters_exact = base_parameters_before == model.baseline_parameter_digest()?;
    let examples: Vec<_> = roots.iter().zip(&q8).collect();
    let input = V5Inputs::<B>::from_examples(&examples, device)?;
    let null = model.paired(&input, 4, Treatment::AllPayloadNull);
    let null_error = max_abs(null.centered_delta);
    let null_tolerance = if device_label == "cpu" { 0.0 } else { 1.0e-6 };
    anyhow::ensure!(
        null_error <= null_tolerance,
        "paired-null centered error {null_error:e} exceeds {null_tolerance:e}"
    );
    let r8_started = Instant::now();
    let _ = model
        .paired(&input, 8, Treatment::Normal)
        .logits
        .into_data();
    let r8_forward_seconds = r8_started.elapsed().as_secs_f64();
    let (
        (
            checkpoint_restore_exact,
            optimizer_moments_restore_exact,
            resumed_update_parameters_exact,
            resumed_update_moments_exact,
        ),
        checkpoint_profile,
    ) = {
        let mut profile = SynchronizedProfile::<B>::new(device)?;
        let result = checkpoint_roundtrip(&model, &optim, device, &roots, &q8, &mut |name| {
            profile.mark(name)
        })?;
        (result, profile.finish()?)
    };
    if device_label == "cuda"
        && let Some((_, used)) = gpu_memory()
    {
        memory_peak = Some(memory_peak.unwrap_or(used).max(used));
    }
    let memory_delta = memory_baseline
        .zip(memory_peak)
        .map(|(baseline, peak)| peak.saturating_sub(baseline.1));
    let memory_pass = device_label != "cuda" || memory_delta.is_some_and(|delta| delta <= 3_072);
    let phase_count = |name: &str| {
        synchronized_profile_q8_r4
            .iter()
            .filter(|timing| timing.phase == name)
            .count()
    };
    let profile_phase_accounting_consistent = phase_count("root_candidate_facts_exact_cpu") == 2
        && phase_count("frozen_root_encoder_and_candidate_path") == 1
        && phase_count("returned_state_encoder") == 1
        && phase_count("adamw_and_completion_fence") == 1
        && ["factual", "null"].iter().all(|stream| {
            phase_count(&format!("{stream}.initialize")) == 1
                && (1..=4).all(|r| {
                    ["evidence", "hypothesis"]
                        .iter()
                        .all(|block| phase_count(&format!("{stream}.r{r}.{block}")) == 1)
                })
        })
        && synchronized_profile_q8_r4
            .iter()
            .all(|phase| phase.seconds.is_finite() && phase.seconds >= 0.0);
    let pass = gradient_groups.len() == 4
        && payload_input_gradient_l2.is_finite()
        && payload_input_gradient_l2 > 0.0
        && base_before == base_after
        && base_parameters_exact
        && checkpoint_restore_exact
        && optimizer_moments_restore_exact
        && resumed_update_parameters_exact
        && resumed_update_moments_exact
        && graph_free_baseline_matches_reference
        && profile_outputs_and_all_gradients_exact
        && profile_adamw_parameters_and_moments_exact
        && profile_phase_accounting_consistent
        && memory_pass
        && null_error <= null_tolerance;
    Ok(QualificationReport {
        schema: "v5_qualification_report_v1".into(),
        source_sha: source_sha.into(),
        pass,
        device: device_label.into(),
        precision: "fp32".into(),
        microbatch,
        architecture: crate::config::ARCHITECTURE.into(),
        config_digest: cfg.scientific_digest()?,
        parameters: model.num_params(),
        parameter_bytes_fp32: model.num_params() * 4,
        query_seconds,
        qualification_wall_seconds: qualification_start.elapsed().as_secs_f64(),
        timing_contract: crate::profile::CONTRACT.into(),
        timing_contract_digest: crate::profile::contract_digest(),
        synchronized_profile_q8_r4,
        synchronized_profile_wall_seconds,
        synchronized_profile_accounting: accounting(&roots, &q8, 4),
        checkpoint_profile,
        profile_outputs_and_all_gradients_exact,
        profile_adamw_parameters_and_moments_exact,
        profile_phase_accounting_consistent,
        matrix,
        worst_update_warm_seconds: worst_warm,
        repeated_updates: 50,
        repeated_update_first_seconds: repeated_first,
        repeated_update_last_seconds: repeated_last,
        null_centered_max_abs: null_error,
        null_tolerance,
        payload_input_gradient_l2,
        gradient_groups,
        gradient_coverage: tracked_coverage,
        repeated_update_seconds,
        repeated_device_used_mib,
        baseline_exact_after_reader_updates: base_before == base_after,
        baseline_parameters_exact_after_reader_updates: base_parameters_exact,
        checkpoint_restore_exact,
        optimizer_moments_restore_exact,
        resumed_update_parameters_exact,
        resumed_update_moments_exact,
        graph_free_baseline_matches_reference,
        r8_forward_seconds,
        detected_vram_mib: memory_baseline.map(|value| value.0),
        device_used_baseline_mib: memory_baseline.map(|value| value.1),
        device_used_peak_mib: memory_peak,
        device_used_peak_delta_mib: memory_delta,
        requested_q: vec![2, 4, 8],
        requested_r: vec![1, 2, 4],
        notes: vec![
            "actual paired factual/null graph; both streams differentiated".into(),
            "R8 is forward-only engineering qualification".into(),
            "cold shape timings are separated from repeated warm updates".into(),
            "all update timings now end after Backend::sync completes AdamW; NVML subprocess sampling overhead is included in resident-loop wall intervals".into(),
            "synchronized component profiling includes fence overhead and host preparation/upload; it is not uninstrumented throughput or an online active decision".into(),
            "current raw packets/root CandidateFacts are prepared twice (autodiff reader inputs plus graph-free baseline inputs); root/state encoder executions remain once each".into(),
            "CandidateFacts performs one successor-board play and terminal/reply inspection per legal root candidate per call; this common baseline work is NOT Q and exact reply enumeration is not separately counted".into(),
            "successful unique qualification checkpoint temporaries are removed after verification; failed/interrupted temporaries are preserved".into(),
            "checkpoint verification compares every FP32 parameter, optimizer moment/counter, and one continued AdamW update exactly".into(),
            "CUDA memory is the nvidia-smi device-used increase over the pre-run baseline; unrelated GPU workloads were not terminated".into(),
        ],
    })
}

#[cfg(test)]
mod baseline_diagnostic {
    use super::*;

    #[test]
    fn synchronized_observers_preserve_all_gradients_and_adamw_with_padding() {
        type B = burn::backend::Autodiff<burn::backend::Flex>;
        let _guard = crate::CPU_TEST_RNG
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Default::default();
        <B as Backend>::seed(&device, 5301);
        let model = CounterfactualRelationalLoop::<B>::new(V5Config::default(), &device);
        let roots = roots(2);
        let mut graphs = graphs(&roots, 8).unwrap();
        graphs[1] = graphs[1].prefix(2).unwrap();
        let optim = adamw::<B, CounterfactualRelationalLoop<B>>();
        for r in [1, 2, 4] {
            assert_eq!(
                profile_parity(&model, &optim, &roots, &graphs, r, &device).unwrap(),
                (true, true),
                "R{r}"
            );
            let counts = accounting(&roots, &graphs, r);
            assert_eq!(counts.actual_q, [8, 2]);
            assert_eq!(counts.root_encoder_examples, 2);
            assert_eq!(counts.returned_encoder_examples, 10);
            assert_eq!(counts.returned_encoder_physical_rows, 16);
            assert_eq!(counts.returned_encoder_padding_rows, 6);
            assert_eq!(counts.raw_packet_preparations, 20);
            assert_eq!(counts.core_applications_whole_batch, 8 * r);
        }
        let mut profile = SynchronizedProfile::<B>::new(&device).unwrap();
        let mut profile_optim = optim;
        update_observed(
            model,
            &mut profile_optim,
            &roots,
            &graphs,
            4,
            &device,
            true,
            &mut |name| profile.mark(name),
        )
        .unwrap();
        let phases = profile.finish().unwrap();
        assert_eq!(
            phases
                .iter()
                .filter(|p| p.phase.ends_with(".evidence") || p.phase.ends_with(".hypothesis"))
                .count(),
            16
        );
        assert_eq!(
            phases
                .iter()
                .filter(|p| p.phase == "root_candidate_facts_exact_cpu")
                .count(),
            2
        );
        assert_eq!(
            phases
                .iter()
                .filter(|p| p.phase == "returned_state_encoder")
                .count(),
            1
        );
        assert_eq!(phases.last().unwrap().phase, "adamw_and_completion_fence");
    }

    #[test]
    fn explicit_attention_preserves_old_autodiff_outputs_and_payload_gradients_exactly() {
        type B = burn::backend::Autodiff<burn::backend::Flex>;
        let _guard = crate::CPU_TEST_RNG
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Default::default();
        <B as Backend>::seed(&device, 5301);
        let model = CounterfactualRelationalLoop::<B>::new(V5Config::default(), &device);
        let roots = roots(2);
        let before = model.baseline_parameter_digest().unwrap();
        for q in [2, 4, 8] {
            let graphs = graphs(&roots, q).unwrap();
            let examples: Vec<_> = roots.iter().zip(&graphs).collect();
            for r in [1, 2, 4] {
                let run = || {
                    let mut tracked_input =
                        V5Inputs::<B>::from_examples(&examples, &device).unwrap();
                    let tracked = tracked_input.states.clone().require_grad();
                    tracked_input.states = tracked.clone();
                    let out = model.paired(&tracked_input, r, Treatment::Normal);
                    let logits = out.logits.clone().into_data();
                    let delta = out.centered_delta.clone().into_data();
                    let target = out.centered_delta.clone().slice([0..1, 0..1]).sum()
                        - out.centered_delta.slice([0..1, 1..2]).sum();
                    let grads = target.backward();
                    let input_grad = tracked.grad(&grads).unwrap().into_data();
                    (logits, delta, input_grad)
                };
                assert_eq!(run(), crate::model::with_legacy_softmax(run), "Q{q}/R{r}");
            }
        }
        assert_eq!(before, model.baseline_parameter_digest().unwrap());
    }

    #[test]
    fn isolate_pinned_softmax_backend_dispatch_difference() {
        type Free = burn::backend::Flex;
        type Ad = burn::backend::Autodiff<Free>;
        let device = Default::default();
        // Fixed, nonsaturated data: no RNG or model-performance selection.
        let values: Vec<f32> = (0..2 * 8 * 64 * 64)
            .map(|i| ((i * 73 % 997) as f32 - 498.0) / 179.0)
            .collect();
        let data = TensorData::new(values, [2, 8, 64, 64]);
        let free = Tensor::<Free, 4>::from_data(data.clone(), &device);
        let ad = Tensor::<Ad, 4>::from_data(data, &device);
        let direct = burn::tensor::activation::softmax(free.clone(), 3).into_data();
        let reference = burn::tensor::activation::softmax(ad, 3).into_data();
        let max = free.clone().detach().max_dim(3);
        let exp = (free - max).exp();
        let explicit = (exp.clone() / exp.sum_dim(3)).into_data();
        let x = direct.to_vec::<f32>().unwrap();
        let y = reference.to_vec::<f32>().unwrap();
        let error = x
            .iter()
            .zip(&y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        println!(
            "V5_SOFTMAX_DISPATCH_DIAGNOSTIC {}",
            serde_json::json!({
                "graph_free_builtin_vs_autodiff_max_abs": error,
                "explicit_graph_free_vs_autodiff_tensor_data_exact": explicit == reference,
                "production_execution_unchanged": true
            })
        );
        assert!(error.is_finite() && error > 0.0);
        assert_eq!(explicit, reference);
    }

    #[test]
    fn measure_graph_free_against_autodiff_reference_without_relaxing_the_gate() {
        type B = burn::backend::Autodiff<burn::backend::Flex>;
        let _guard = crate::CPU_TEST_RNG
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Default::default();
        <B as Backend>::seed(&device, 5301);
        let model = CounterfactualRelationalLoop::<B>::new(V5Config::default(), &device);
        let roots = roots(2);
        let graphs = graphs(&roots, 8).unwrap();
        let examples: Vec<_> = roots.iter().zip(&graphs).collect();
        let input = V5Inputs::<B>::from_examples(&examples, &device).unwrap();
        let reference = model.base(&input);
        let frozen = model.base_frozen(&examples, &device).unwrap();
        for (name, ad, constant) in [
            (
                "context",
                reference.context.into_data(),
                frozen.context.into_data(),
            ),
            (
                "hypotheses",
                reference.hypotheses.into_data(),
                frozen.hypotheses.into_data(),
            ),
            ("z0", reference.z0.into_data(), frozen.z0.into_data()),
        ] {
            let x = ad.to_vec::<f32>().unwrap();
            let y = constant.to_vec::<f32>().unwrap();
            assert_eq!(x.len(), y.len());
            let max_abs = x
                .iter()
                .zip(&y)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            let first = x
                .iter()
                .zip(&y)
                .enumerate()
                .find(|(_, (a, b))| a.to_bits() != b.to_bits());
            println!(
                "V5_BASELINE_REFERENCE_DIAGNOSTIC {}",
                serde_json::json!({"field": name,
                "tensor_data_equal": ad == constant, "shape_ad": ad.shape, "shape_graph_free": constant.shape,
                "dtype_ad": format!("{:?}", ad.dtype), "dtype_graph_free": format!("{:?}", constant.dtype),
                "max_abs": max_abs, "first_value_bit_mismatch": first.map(|(i,(a,b))| (i,*a,*b)),
                "values_bit_exact": first.is_none(), "qualifying_gate_unchanged": true})
            );
            assert!(max_abs.is_finite());
            // The corrective execution experiment must satisfy the ORIGINAL
            // bit-exact comparison, not a relaxed tolerance.
            assert_eq!(ad, constant, "graph-free baseline mismatch: {name}");
        }
    }
}
