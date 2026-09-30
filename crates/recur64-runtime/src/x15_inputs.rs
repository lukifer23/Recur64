//! Wire the deterministic coprocessor and the visual renderer into a Chimera
//! forward-pass batch.
//!
//! This is the one place that turns `(Observation V1, canonical legal list)`
//! into the tensors [`recur64_model::chimera::ChimeraModel`] consumes, so the
//! inference path, the probe harness and the training step all feed the model
//! identically. It also exposes the per-stage phase timings the performance
//! requirement asks for (compute vs render vs upload), because a coprocessor or
//! a renderer that becomes the bottleneck must be visible rather than hidden
//! inside "forward".

use std::time::Instant;

use burn::prelude::*;
use burn::tensor::TensorData;

use recur64_compute::ComputeProvider;
use recur64_coproc::{OUTPUT_LEN, visual::image_bytes};
use recur64_core::{ActionId, GameState, ObservationV1, encode_observation_v1};
use recur64_model::action::CandidateBatch;
use recur64_model::experimental::ExperimentalConfig;
use recur64_model::model::CandidateTensors;

/// Wall time (microseconds) of each input-preparation stage for one batch.
#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize)]
pub struct X15PhaseTimes {
    pub observation_us: u64,
    pub compute_us: u64,
    pub visual_render_us: u64,
    pub upload_us: u64,
}

/// A prepared Chimera batch: the tensors plus the phase breakdown.
pub struct X15Batch<B: Backend> {
    pub input: recur64_model::chimera::ChimeraInput<B>,
    pub cands: CandidateTensors<B>,
    pub phases: X15PhaseTimes,
    /// Positions in the batch.
    pub batch: usize,
}

/// Build the deterministic inputs for a batch of positions.
///
/// `provider` computes `ComputeBankV1` when `exp.compute` is enabled and its
/// provider is active; `exp.visual` drives the renderer. Both are skipped
/// entirely when their pathway is off, so the "pathway off" configuration does
/// no unnecessary host work.
pub fn build_x15_batch<B: Backend>(
    states: &[GameState],
    exp: &ExperimentalConfig,
    provider: &dyn ComputeProvider,
    device: &B::Device,
) -> anyhow::Result<X15Batch<B>> {
    build_x15_batch_padded(states, exp, provider, device, 0)
}

/// [`build_x15_batch`] with the candidate width padded to at least `min_width`,
/// so a run can use one fixed candidate shape (fewer kernel compilations).
pub fn build_x15_batch_padded<B: Backend>(
    states: &[GameState],
    exp: &ExperimentalConfig,
    provider: &dyn ComputeProvider,
    device: &B::Device,
    min_width: usize,
) -> anyhow::Result<X15Batch<B>> {
    anyhow::ensure!(
        !states.is_empty(),
        "an X15 batch needs at least one position"
    );
    let b = states.len();
    let mut phases = X15PhaseTimes::default();

    let t0 = Instant::now();
    let observations: Vec<ObservationV1> = states.iter().map(encode_observation_v1).collect();
    let legal: Vec<Vec<ActionId>> = states.iter().map(|s| s.legal_actions()).collect();
    anyhow::ensure!(
        legal.iter().all(|l| !l.is_empty()),
        "an X15 batch cannot contain a terminal position (the policy path refuses it)"
    );
    phases.observation_us = t0.elapsed().as_micros() as u64;

    // --- deterministic compute bank ---
    let use_compute = exp.is_chimera() && exp.compute.enabled && exp.compute.provider.is_active();
    let t1 = Instant::now();
    let compute_tensor = if use_compute {
        let inputs: Vec<Vec<u8>> = observations
            .iter()
            .zip(&legal)
            .map(|(o, l)| recur64_compute::encode_input(o, l, exp.compute.mate_depth))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let mut out: Vec<Vec<u8>> = inputs.iter().map(|_| vec![0u8; OUTPUT_LEN]).collect();
        provider
            .compute_batch(&inputs, &mut out)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        phases.compute_us = t1.elapsed().as_micros() as u64;

        // Bytes -> floats in 0..255, laid out [b, tokens, fields].
        let mut data = Vec::with_capacity(b * exp.compute.tokens * exp.compute.fields);
        for bank in &out {
            debug_assert_eq!(bank.len(), OUTPUT_LEN);
            data.extend(bank.iter().map(|v| *v as f32));
        }
        Some(Tensor::<B, 3>::from_data(
            TensorData::new(data, [b, exp.compute.tokens, exp.compute.fields]),
            device,
        ))
    } else {
        None
    };

    // --- visual render ---
    let use_visual = exp.is_chimera()
        && exp.visual.enabled
        && exp.visual.provider == recur64_model::experimental::VisualProviderKind::RenderV1;
    let t2 = Instant::now();
    let visual_tensor = if use_visual {
        let side = exp.visual.resolution;
        let per = image_bytes(side);
        let mut image = vec![0u8; b * per];
        for (i, (o, l)) in observations.iter().zip(&legal).enumerate() {
            let input = recur64_compute::encode_input(o, l, 0)?;
            let slot = &mut image[i * per..(i + 1) * per];
            recur64_coproc::visual::render_board(&input, slot, side)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        phases.visual_render_us = t2.elapsed().as_micros() as u64;

        // NHWC bytes -> NCHW floats in 0..1.
        let plane = side * side;
        let channels = 3usize;
        let mut data = vec![0f32; b * channels * plane];
        for i in 0..b {
            for px in 0..plane {
                for c in 0..channels {
                    let src = i * per + px * channels + c;
                    let dst = i * channels * plane + c * plane + px;
                    data[dst] = image[src] as f32 / 255.0;
                }
            }
        }
        Some(Tensor::<B, 4>::from_data(
            TensorData::new(data, [b, channels, side, side]),
            device,
        ))
    } else {
        None
    };

    // --- symbolic + candidate upload ---
    let t3 = Instant::now();
    let mut flat = Vec::with_capacity(b * recur64_core::OBS_LEN);
    for o in &observations {
        flat.extend_from_slice(o.as_slice());
    }
    let board = Tensor::<B, 3>::from_data(
        TensorData::new(
            flat,
            [
                b,
                recur64_coproc::SQUARES,
                recur64_core::FEATURES_PER_SQUARE,
            ],
        ),
        device,
    );
    let lists: Vec<Vec<(u32, u32, u8)>> = legal
        .iter()
        .map(|l| {
            l.iter()
                .map(|id| {
                    let (from, to, promo) = id.decode();
                    (from as usize as u32, to as usize as u32, promo.code())
                })
                .collect()
        })
        .collect();
    let cb = CandidateBatch::from_lists_min_width(&lists, min_width);
    let cands = CandidateTensors::from_batch(&cb, device);
    let use_facts =
        exp.is_chimera() && exp.candidate_facts.enabled && exp.candidate_facts.provider.is_active();
    let cand_facts = if use_facts {
        let width = cands.width;
        let data = crate::candidate_facts::candidate_facts(states, width)?;
        Some(Tensor::<B, 3>::from_data(
            TensorData::new(data, [b, width, crate::candidate_facts::FIELDS]),
            device,
        ))
    } else {
        None
    };
    phases.upload_us = t3.elapsed().as_micros() as u64;

    Ok(X15Batch {
        input: recur64_model::chimera::ChimeraInput {
            board,
            compute: compute_tensor,
            visual: visual_tensor,
            cand_facts,
        },
        cands,
        phases,
        batch: b,
    })
}

/// The reproducible non-terminal positions the probes use: the start position
/// plus seeded random legal playouts, all from a single fixed seed so every
/// probe run sees the same inputs.
pub fn probe_positions(n: usize, max_plies: usize) -> Vec<GameState> {
    use recur64_core::StandardMove;
    use recur64_search::Rng;

    let mut rng = Rng::new(20260929);
    let mut out = Vec::with_capacity(n);
    let mut i = 0usize;
    while out.len() < n {
        let mut s = GameState::startpos();
        let plies = if max_plies == 0 {
            0
        } else {
            (rng.next_u64() % max_plies as u64 + 1) as usize
        };
        for _ in 0..plies {
            if s.termination().is_some() {
                break;
            }
            let legal = s.legal_actions();
            let a = legal[(rng.next_u64() % legal.len() as u64) as usize];
            let (from, to, promo) = a.to_physical(s.perspective());
            s.apply(StandardMove::new(
                from,
                to,
                (!promo.is_none()).then_some(promo),
            ))
            .expect("generated move is legal");
        }
        if s.termination().is_none() {
            out.push(s);
        }
        i += 1;
        if i > 100_000 {
            panic!("could not generate {n} non-terminal positions");
        }
    }
    out
}

/// The provider label a config selects, or `none`.
pub fn provider_for_config(exp: &ExperimentalConfig) -> anyhow::Result<Box<dyn ComputeProvider>> {
    let kind = if exp.is_chimera() && exp.compute.enabled {
        exp.compute.provider
    } else {
        recur64_coproc::ComputeProviderKind::None
    };
    recur64_compute::provider_for(kind)
}

/// The candidate width a set of positions should be padded to: the next
/// standard bucket at or above the widest legal-move list.
pub fn bucket_width(states: &[GameState]) -> usize {
    let max = states
        .iter()
        .map(|s| s.legal_actions().len())
        .max()
        .unwrap_or(0);
    recur64_model::action::CandidateBatch::WIDTH_BUCKETS
        .iter()
        .copied()
        .find(|&w| w >= max)
        .unwrap_or(max)
}
