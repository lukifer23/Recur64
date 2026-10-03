//! CPU/CUDA qualification for the actual paired V5 graph.

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

#[derive(Debug, Clone, Serialize)]
pub struct ShapeTiming {
    pub q: usize,
    pub r: usize,
    pub cold_seconds: f64,
    pub warm_seconds: f64,
    pub loss: f64,
    pub core_applications_per_example: usize,
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
    mut model: CounterfactualRelationalLoop<B>,
    optim: &mut Opt<B>,
    roots: &[recur64_core::GameState],
    graphs: &[AcquiredGraph],
    r: usize,
    device: &B::Device,
    track_input: bool,
) -> anyhow::Result<(CounterfactualRelationalLoop<B>, f64, Vec<CoverageRow>, f32)> {
    let examples: Vec<_> = roots.iter().zip(graphs).collect();
    let mut input = V5Inputs::<B>::from_examples(&examples, device)?;
    let tracked = track_input.then(|| input.states.clone().require_grad());
    if let Some(value) = &tracked {
        input.states = value.clone();
    }
    let base = model.base_frozen(&examples, device)?;
    let output = model.paired_with_base(&input, base, r, Treatment::Normal);
    let correct = first_legal_correct(&input, device);
    let loss = correct_set_loss(output.logits, input.cands.mask.clone(), correct);
    let loss_value = f64::from(loss.clone().into_data().to_vec::<f32>()?[0]);
    anyhow::ensure!(loss_value.is_finite(), "qualification loss is non-finite");
    let raw = loss.backward();
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
    model = optim.step(3.0e-4, model, grads);
    Ok((model, loss_value, coverage, input_norm))
}

fn checkpoint_roundtrip<B: AutodiffBackend>(
    model: &CounterfactualRelationalLoop<B>,
    optim: &Opt<B>,
    device: &B::Device,
    roots: &[recur64_core::GameState],
    graphs: &[AcquiredGraph],
) -> anyhow::Result<(bool, bool, bool, bool)> {
    let dir: PathBuf = std::env::temp_dir().join(format!(
        "recur64-v5-qualification-{}-{}",
        std::process::id(),
        model.num_params()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model.clone().save_file(dir.join("model"), &recorder)?;
    recorder.record(optim.to_record(), dir.join("optimizer"))?;
    let template = CounterfactualRelationalLoop::<B>::new(V5Config::default(), device);
    let loaded = template.load_file(dir.join("model"), &recorder, device)?;
    let record = recorder.load(dir.join("optimizer"), device)?;
    let mut loaded_optim = adamw::<B, CounterfactualRelationalLoop<B>>().load_record(record);
    let equal = model.parameter_digest()? == loaded.parameter_digest()?
        && logits(model, roots, graphs, 4, device)? == logits(&loaded, roots, graphs, 4, device)?;
    let moments_equal = optimizer_digest(optim)? == optimizer_digest(&loaded_optim)?;
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
    let _ = std::fs::remove_dir_all(dir);
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

pub fn run<B>(
    source_sha: &str,
    device_label: &str,
    microbatch: usize,
    device: &B::Device,
) -> anyhow::Result<QualificationReport>
where
    B: AutodiffBackend,
{
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
        checkpoint_restore_exact,
        optimizer_moments_restore_exact,
        resumed_update_parameters_exact,
        resumed_update_moments_exact,
    ) = checkpoint_roundtrip(&model, &optim, device, &roots, &q8)?;
    if device_label == "cuda"
        && let Some((_, used)) = gpu_memory()
    {
        memory_peak = Some(memory_peak.unwrap_or(used).max(used));
    }
    let memory_delta = memory_baseline
        .zip(memory_peak)
        .map(|(baseline, peak)| peak.saturating_sub(baseline.1));
    let memory_pass = device_label != "cuda" || memory_delta.is_some_and(|delta| delta <= 3_072);
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
            "checkpoint verification compares every FP32 parameter, optimizer moment/counter, and one continued AdamW update exactly".into(),
            "CUDA memory is the nvidia-smi device-used increase over the pre-run baseline; unrelated GPU workloads were not terminated".into(),
        ],
    })
}
