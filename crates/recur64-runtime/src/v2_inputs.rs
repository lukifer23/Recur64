//! Chimera V2 input construction: positions -> observation, candidate lists, root
//! candidate facts, and the unpacked world-model tensors.
//!
//! Every batch uses ONE fixed candidate width (`v2.w_cap`) and one reply width
//! (`v2.r_cap`) so the GPU sees a single shape per run. A position that does not
//! fit is a visible error, never a truncation.

use std::time::Instant;

use burn::prelude::*;
use burn::tensor::TensorData;

use recur64_compute::{WorldModelProvider, encode_input};
use recur64_coproc::world::{
    REPLY_BYTES, ROOT_FIELDS, SUCC_BYTES, WORLD_HEADER, reply_offset, root_offset, succ_offset,
    world_output_len,
};
use recur64_core::{GameState, ObservationV1, encode_observation_v1};
use recur64_model::action::CandidateBatch;
use recur64_model::chimera2::{
    BOARD_CODES, ChimeraV2Input, REPLY_FEAT_DIM, SUCC_FLAG_DIM, WorldTensors,
};
use recur64_model::experimental::ExperimentalConfig;
use recur64_model::model::CandidateTensors;

/// Wall-clock split of building one batch (compute accounting).
#[derive(Debug, Clone, Copy, Default)]
pub struct V2PhaseTimes {
    pub observation_us: u64,
    /// World-model wall (tool CPU): 0 when the schedule needs no tools.
    pub world_us: u64,
    pub unpack_us: u64,
    pub upload_us: u64,
    /// Positions whose world model was expanded (root + all successors + all replies).
    pub world_states: usize,
    pub successors: usize,
    pub replies: usize,
}

pub struct V2Batch<B: Backend> {
    pub input: ChimeraV2Input<B>,
    pub cands: CandidateTensors<B>,
    pub batch: usize,
    pub phases: V2PhaseTimes,
}

/// Normalisation contract for the unpacked reply / successor features (all `0..=1`).
fn clip(x: u8, scale: f32) -> f32 {
    (x as f32 / scale).min(1.0)
}

/// Root candidate facts as floats (`0..=1`), `[b, w, 8]` row-major, from the world
/// model's own root section (the WASM twin returns the same bytes).
pub fn root_facts_from_bytes(bytes: &[u8], w_cap: usize) -> Vec<f32> {
    let mut out = vec![0f32; w_cap * ROOT_FIELDS];
    for (i, o) in out.iter_mut().enumerate() {
        let v = bytes[root_offset() + i] as f32;
        // Fields: mate, check, capture, captured/9, attacked, promotion, gain/8, stalemate.
        let field = i % ROOT_FIELDS;
        *o = match field {
            3 => v / 9.0,
            6 => v / 8.0,
            _ => v,
        };
    }
    out
}

/// Unpack one position's world bytes into the tensor layouts of
/// [`WorldTensors`]. Appends to the four flat buffers.
#[allow(clippy::too_many_arguments)]
fn unpack_one(
    bytes: &[u8],
    w_cap: usize,
    r_cap: usize,
    succ_board: &mut Vec<f32>,
    succ_flags: &mut Vec<f32>,
    reply_feats: &mut Vec<f32>,
    reply_mask: &mut Vec<f32>,
) -> (usize, usize) {
    let n = u16::from_le_bytes([bytes[0], bytes[1]]) as usize;
    let (mut succs, mut replies) = (0usize, 0usize);
    let so = succ_offset(w_cap);
    let ro = reply_offset(w_cap);
    for ci in 0..w_cap {
        let s = so + ci * SUCC_BYTES;
        let live = ci < n;
        // One-hot placement.
        for sq in 0..64 {
            let code = bytes[s + 4 + sq] as usize;
            for k in 0..BOARD_CODES {
                succ_board.push(if live && k == code { 1.0 } else { 0.0 });
            }
        }
        // Flags: terminal one-hot (5), in check, replies/32, castle nibble (4), ep, clock/100.
        let terminal = bytes[s] as usize;
        for k in 0..5 {
            succ_flags.push(if live && k == terminal { 1.0 } else { 0.0 });
        }
        succ_flags.push(if live { f32::from(bytes[s + 1]) } else { 0.0 });
        succ_flags.push(if live { clip(bytes[s + 2], 32.0) } else { 0.0 });
        for bit in 0..4 {
            succ_flags.push(if live {
                f32::from((bytes[s + 68] >> bit) & 1)
            } else {
                0.0
            });
        }
        succ_flags.push(if live && bytes[s + 69] > 0 { 1.0 } else { 0.0 });
        succ_flags.push(if live {
            clip(bytes[s + 70], 100.0)
        } else {
            0.0
        });
        if live {
            succs += 1;
        }
        for ri in 0..r_cap {
            let base = ro + (ci * r_cap + ri) * REPLY_BYTES;
            let valid = live && bytes[base] == 1;
            reply_mask.push(if valid { 1.0 } else { 0.0 });
            if !valid {
                reply_feats.extend(std::iter::repeat_n(0.0f32, REPLY_FEAT_DIM));
                continue;
            }
            replies += 1;
            let f = &bytes[base + 1..base + 9];
            reply_feats.extend_from_slice(&[
                f32::from(f[0]),
                f32::from(f[1]),
                f32::from(f[2]),
                f32::from(f[3]) / 9.0,
                f32::from(f[4]),
                f32::from(f[5]),
                f32::from(f[6]) / 8.0,
                f32::from(f[7]),
            ]);
            let terminal = bytes[base + 9] as usize;
            for k in 0..5 {
                reply_feats.push(if k == terminal { 1.0 } else { 0.0 });
            }
            reply_feats.push(f32::from(bytes[base + 10]));
            // next-player summary: legal, mates, checks, captures, promotions, max cap, max gain
            reply_feats.extend_from_slice(&[
                clip(bytes[base + 12], 32.0),
                clip(bytes[base + 13], 8.0),
                clip(bytes[base + 14], 16.0),
                clip(bytes[base + 15], 16.0),
                clip(bytes[base + 16], 4.0),
                f32::from(bytes[base + 17]) / 9.0,
                f32::from(bytes[base + 18]) / 8.0,
            ]);
        }
    }
    (succs, replies)
}

/// Encode the coprocessor input for each position (canonical observation + legal moves).
pub fn world_inputs(states: &[GameState]) -> anyhow::Result<Vec<Vec<u8>>> {
    states
        .iter()
        .map(|s| {
            let obs: ObservationV1 = encode_observation_v1(s);
            encode_input(&obs, &s.legal_actions(), 0)
        })
        .collect()
}

/// Compute (or reuse) the packed world bytes for `states`.
pub fn compute_world_bytes(
    states: &[GameState],
    provider: &dyn WorldModelProvider,
    w_cap: usize,
    r_cap: usize,
) -> anyhow::Result<Vec<Vec<u8>>> {
    let inputs = world_inputs(states)?;
    let out = provider
        .world_batch(&inputs, w_cap, r_cap)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    debug_assert!(
        out.iter()
            .all(|o| o.len() == world_output_len(w_cap, r_cap))
    );
    Ok(out)
}

/// Build a batch. `world_bytes` may be supplied (a dataset cache); otherwise, when
/// the schedule needs the world model, `provider` computes it. Root facts always
/// come from the world model's root section when bytes exist, and otherwise from
/// the native `CandidateFactsV1` function (identical semantics, tested).
pub fn build_v2_batch<B: Backend>(
    states: &[GameState],
    exp: &ExperimentalConfig,
    provider: Option<&dyn WorldModelProvider>,
    world_bytes: Option<&[Vec<u8>]>,
    device: &B::Device,
) -> anyhow::Result<V2Batch<B>> {
    anyhow::ensure!(!states.is_empty(), "a V2 batch needs at least one position");
    let v = &exp.v2;
    let (w_cap, r_cap) = (v.w_cap, v.r_cap);
    let b = states.len();
    let mut phases = V2PhaseTimes::default();

    let t0 = Instant::now();
    let observations: Vec<ObservationV1> = states.iter().map(encode_observation_v1).collect();
    let legal: Vec<Vec<recur64_core::ActionId>> =
        states.iter().map(|s| s.legal_actions()).collect();
    anyhow::ensure!(
        legal.iter().all(|l| !l.is_empty()),
        "a V2 batch cannot contain a terminal position"
    );
    for (i, l) in legal.iter().enumerate() {
        anyhow::ensure!(
            l.len() <= w_cap,
            "position {i} has {} legal moves, more than v2.w_cap {w_cap}",
            l.len()
        );
    }
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
    let cb = CandidateBatch::from_lists_min_width(&lists, w_cap);
    let cands = CandidateTensors::from_batch(&cb, device);
    debug_assert_eq!(cands.width, w_cap);
    phases.observation_us = t0.elapsed().as_micros() as u64;

    // World model bytes (tool CPU).
    let owned: Vec<Vec<u8>>;
    let bytes: Option<&[Vec<u8>]> = if let Some(w) = world_bytes {
        anyhow::ensure!(
            w.len() == b,
            "cached world bytes for {} positions, batch has {b}",
            w.len()
        );
        Some(w)
    } else if v.info_schedule.needs_world_model() {
        let p = provider.ok_or_else(|| {
            anyhow::anyhow!("the info schedule needs a world-model provider and none was given")
        })?;
        let t1 = Instant::now();
        owned = compute_world_bytes(states, p, w_cap, r_cap)?;
        phases.world_us = t1.elapsed().as_micros() as u64;
        Some(&owned)
    } else {
        None
    };

    // Root facts.
    let facts_flat: Vec<f32> = match bytes {
        Some(bs) => bs
            .iter()
            .flat_map(|x| root_facts_from_bytes(x, w_cap))
            .collect(),
        None => crate::candidate_facts::candidate_facts(states, w_cap)?,
    };

    // World tensors.
    let t2 = Instant::now();
    let world = match (bytes, v.info_schedule.needs_world_model()) {
        (Some(bs), true) => {
            let (mut sb, mut sf, mut rf, mut rm) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for x in bs {
                anyhow::ensure!(
                    x.len() == world_output_len(w_cap, r_cap),
                    "world bytes are {} long, expected {}",
                    x.len(),
                    world_output_len(w_cap, r_cap)
                );
                let (s, r) = unpack_one(x, w_cap, r_cap, &mut sb, &mut sf, &mut rf, &mut rm);
                phases.successors += s;
                phases.replies += r;
                phases.world_states += 1 + s + r;
            }
            phases.unpack_us = t2.elapsed().as_micros() as u64;
            let t3 = Instant::now();
            let w = WorldTensors {
                succ_board: Tensor::<B, 4>::from_data(
                    TensorData::new(sb, [b, w_cap, 64, BOARD_CODES]),
                    device,
                ),
                succ_flags: Tensor::<B, 3>::from_data(
                    TensorData::new(sf, [b, w_cap, SUCC_FLAG_DIM]),
                    device,
                ),
                reply_feats: Tensor::<B, 4>::from_data(
                    TensorData::new(rf, [b, w_cap, r_cap, REPLY_FEAT_DIM]),
                    device,
                ),
                reply_mask: Tensor::<B, 3>::from_data(
                    TensorData::new(rm, [b, w_cap, r_cap]),
                    device,
                ),
            };
            phases.upload_us += t3.elapsed().as_micros() as u64;
            Some(w)
        }
        _ => None,
    };

    let t4 = Instant::now();
    let mut board = Vec::with_capacity(b * recur64_coproc::OBS_LEN);
    for o in &observations {
        board.extend_from_slice(o.as_slice());
    }
    let board = Tensor::<B, 3>::from_data(TensorData::new(board, [b, 64, 119]), device);
    let cand_facts = Tensor::<B, 3>::from_data(
        TensorData::new(
            facts_flat,
            [b, w_cap, recur64_model::experimental::CANDIDATE_FACT_FIELDS],
        ),
        device,
    );
    let visual = if exp.visual.enabled {
        Some(render_visual::<B>(
            &observations,
            &legal,
            exp.visual.resolution,
            device,
        )?)
    } else {
        None
    };
    phases.upload_us += t4.elapsed().as_micros() as u64;
    let _ = WORLD_HEADER;
    Ok(V2Batch {
        input: ChimeraV2Input {
            board,
            visual,
            cand_facts,
            world,
        },
        cands,
        batch: b,
        phases,
    })
}

fn render_visual<B: Backend>(
    observations: &[ObservationV1],
    legal: &[Vec<recur64_core::ActionId>],
    side: usize,
    device: &B::Device,
) -> anyhow::Result<Tensor<B, 4>> {
    let per = recur64_coproc::visual::image_bytes(side);
    let b = observations.len();
    let mut image = vec![0u8; b * per];
    for (i, (o, l)) in observations.iter().zip(legal).enumerate() {
        let input = encode_input(o, l, 0)?;
        recur64_coproc::visual::render_board(&input, &mut image[i * per..(i + 1) * per], side)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    // NHWC u8 -> NCHW f32 in 0..1.
    let mut data = vec![0f32; b * 3 * side * side];
    for i in 0..b {
        for y in 0..side {
            for x in 0..side {
                for c in 0..3 {
                    data[((i * 3 + c) * side + y) * side + x] =
                        image[i * per + (y * side + x) * 3 + c] as f32 / 255.0;
                }
            }
        }
    }
    Ok(Tensor::<B, 4>::from_data(
        TensorData::new(data, [b, 3, side, side]),
        device,
    ))
}
