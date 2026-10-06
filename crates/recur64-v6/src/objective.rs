//! `v6_content_differential_aux_v2_successor_only`: control vs successor-content
//! differential auxiliary objective on the unchanged one-pass reader.
use crate::{
    intervene::{Condition, Maps, intervened, verify_intervention},
    loss,
    model::{Arm, Inputs, Output, Reader, inputs},
    p0::{Plan, episode},
    packet::Packet,
    probe_eval,
    qualify::{self, Opt, scalar},
};
use anyhow::{Result, ensure};
use burn::module::AutodiffModule;
use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::{Bool, backend::AutodiffBackend};
use recur64_v5::{data::V5Data, model::RootInputs};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{path::Path, time::Instant};

pub const OBJECTIVE: &str = "v6_content_differential_aux_v2_successor_only";
pub const LAUNCH_PLAN: &str = "v6_objective_content_probe_v2";
pub const PLAN_SCHEMA: &str = "v6_objective_probe_plan_v2";
pub const P0_PRODUCER: &str = "42f47b852451d59645dcc50120e4a426b034f493";
pub const P0_PLAN_DIGEST: &str = "e7e09f8d853dc69f983c33e61e0186205fdfee21fc650edaafdcc1b7c1c5dee9";
pub const P0_PLAN_RAW_SHA: &str =
    "3be158ebab91f3e691d4fdf723d450c6d4448447d1570a9371fbc7c8682a7e49";
pub const P0_EPISODE_DIGEST: &str =
    "208658ccaef1d0160694b45dd72de1e53ff63d97f5199c029f80a65aebb57092";
pub const AUX_WEIGHT: f64 = 0.5;
pub const PARAMETERS_TOTAL: usize = 6_703_152;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Objective {
    Control,
    Treatment,
}
impl Objective {
    pub fn name(self) -> &'static str {
        match self {
            Objective::Control => "control",
            Objective::Treatment => "treatment",
        }
    }
}

// ---------------------------------------------------------------- frozen inputs

/// Load and independently re-verify the original frozen 96-root/192-packet plan.
/// Packets keep their producer source; nothing is regenerated or reselected.
pub fn load_p0_plan(path: &Path, binding: &Path) -> Result<Plan> {
    let b: Value = serde_json::from_slice(&std::fs::read(binding)?)?;
    ensure!(
        b["schema"] == "v6_p0_measured_plan_binding_v1"
            && b["source_sha"] == P0_PRODUCER
            && b["plan_digest"] == P0_PLAN_DIGEST
            && b["episode_digest"] == P0_EPISODE_DIGEST
            && b["raw_sha256"] == P0_PLAN_RAW_SHA
            && b["train_digest"] == recur64_v5::data::TRAIN_DIGEST
            && recur64_v5::stage::hash_file(path)? == P0_PLAN_RAW_SHA,
        "original frozen plan raw/binding identity mismatch"
    );
    let plan: Plan = serde_json::from_slice(&std::fs::read(path)?)?;
    let mut unsigned = plan.clone();
    unsigned.digest.clear();
    ensure!(
        plan.digest == P0_PLAN_DIGEST
            && crate::digest(&unsigned)? == plan.digest
            && plan.schema == "v6_competent_base_plan_v2"
            && plan.source_sha == P0_PRODUCER
            && plan.config_digest == crate::packet::config_digest()?
            && plan.train_digest == recur64_v5::data::TRAIN_DIGEST
            && plan.seed == 6300
            && plan.updates == 200
            && plan.entries.len() == 96
            && plan.episode_digest == P0_EPISODE_DIGEST,
        "original frozen plan content mismatch"
    );
    let episodes: Vec<_> = (0..200)
        .map(|u| (0..12).map(|k| episode(u, k)).collect::<Vec<_>>())
        .collect();
    ensure!(
        crate::digest(&episodes)? == plan.episode_digest,
        "episode schedule digest mismatch"
    );
    let cells = ["KQRvK M2", "KQRvK M3", "KRRvK M2", "KRRvK M3"];
    let mut ids = std::collections::BTreeSet::new();
    for (cell, chunk) in cells.iter().zip(plan.entries.chunks(24)) {
        ensure!(
            chunk.iter().all(|e| &e.cell == cell)
                && chunk[..12].iter().all(|e| !e.baseline_correct)
                && chunk[12..].iter().all(|e| e.baseline_correct),
            "panel cell/stratum order mismatch"
        );
    }
    for e in &plan.entries {
        ensure!(
            ids.insert(e.id.clone())
                && e.packets.len() == 2
                && e.packets.iter().enumerate().all(|(i, g)| {
                    g.generation_role == "p0_acquisition"
                        && g.root_id == e.id
                        && g.policy == crate::packet::Policy::all()[i]
                        && g.source_sha == P0_PRODUCER
                        && g.verify(&g.source_sha).is_ok()
                }),
            "frozen packet identity/digest mismatch"
        );
    }
    Ok(plan)
}

pub fn verify_packets_against_data(plan: &Plan, data: &V5Data) -> Result<()> {
    data.require_role(recur64_v5::native_data_v2::Role::Train)?;
    for e in &plan.entries {
        ensure!(
            data.position(e.index).id == e.id,
            "panel ID/data index mismatch"
        );
        let root = data.roots(&[e.index])?.remove(0);
        for p in &e.packets {
            ensure!(p.legal == data.position(e.index).legal, "action alignment");
            p.verify_against_root(&root)?;
        }
    }
    Ok(())
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PanelRow {
    pub index: usize,
    pub id: String,
    pub cell: String,
    pub baseline_correct: bool,
    pub packet_digests: Vec<String>,
}
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ProbePlan {
    pub schema: String,
    pub objective: String,
    pub launch_plan: String,
    pub source_sha: String,
    pub config_digest: String,
    pub train_digest: String,
    pub contract_sha256: String,
    pub p0_producer: String,
    pub p0_plan_digest: String,
    pub p0_plan_raw_sha256: String,
    pub episode_digest: String,
    pub panel: Vec<PanelRow>,
    pub maps_digest: String,
    pub map_summary: Value,
    pub seed: u64,
    pub updates: u64,
    pub aux_weight: f64,
    pub conditions: Vec<String>,
    pub initial_model_sha256: String,
    pub initial_parameter_digest: String,
    pub digest: String,
}
impl ProbePlan {
    pub fn seal(&mut self) -> Result<()> {
        self.digest.clear();
        self.digest = crate::digest(self)?;
        Ok(())
    }
    pub fn validate(&self) -> Result<()> {
        let mut c = self.clone();
        c.digest.clear();
        ensure!(
            self.schema == PLAN_SCHEMA
                && self.objective == OBJECTIVE
                && self.launch_plan == LAUNCH_PLAN
                && self.source_sha == crate::SOURCE
                && self.config_digest == crate::packet::config_digest()?
                && self.train_digest == recur64_v5::data::TRAIN_DIGEST
                && self.p0_producer == P0_PRODUCER
                && self.p0_plan_digest == P0_PLAN_DIGEST
                && self.p0_plan_raw_sha256 == P0_PLAN_RAW_SHA
                && self.episode_digest == P0_EPISODE_DIGEST
                && self.panel.len() == 96
                && self.seed == 6300
                && self.updates == 200
                && self.aux_weight == AUX_WEIGHT
                && self.conditions
                    == Condition::all()
                        .iter()
                        .map(|c| c.name().to_string())
                        .collect::<Vec<_>>()
                && self.digest == crate::digest(&c)?,
            "probe plan identity/tamper mismatch"
        );
        Ok(())
    }
}
pub fn maps_summary(maps: &Maps) -> Value {
    let one = |m: &crate::intervene::DonorMap| {
        let rows: Vec<_> = m.rows.iter().flatten().flatten().collect();
        let exact = rows.iter().filter(|r| r.depth_delta == 0).count();
        json!({"seed": m.seed, "rows": rows.len(), "exact_depth": exact,
               "nearest_depth_fallback": rows.len() - exact,
               "max_depth_delta": rows.iter().map(|r| r.depth_delta).max(),
               "min_pool": rows.iter().map(|r| r.pool).min(), "max_pool": rows.iter().map(|r| r.pool).max()})
    };
    json!({"e002": one(&maps.e002), "e402": one(&maps.e402), "e502": one(&maps.e502)})
}
pub fn build_plan(
    p0: &Plan,
    maps: &Maps,
    contract_sha256: String,
    initial_model_sha256: String,
    initial_parameter_digest: String,
) -> Result<ProbePlan> {
    let mut p = ProbePlan {
        schema: PLAN_SCHEMA.into(),
        objective: OBJECTIVE.into(),
        launch_plan: LAUNCH_PLAN.into(),
        source_sha: crate::SOURCE.into(),
        config_digest: crate::packet::config_digest()?,
        train_digest: recur64_v5::data::TRAIN_DIGEST.into(),
        contract_sha256,
        p0_producer: P0_PRODUCER.into(),
        p0_plan_digest: p0.digest.clone(),
        p0_plan_raw_sha256: P0_PLAN_RAW_SHA.into(),
        episode_digest: p0.episode_digest.clone(),
        panel: p0
            .entries
            .iter()
            .map(|e| PanelRow {
                index: e.index,
                id: e.id.clone(),
                cell: e.cell.clone(),
                baseline_correct: e.baseline_correct,
                packet_digests: e.packets.iter().map(|p| p.digest.clone()).collect(),
            })
            .collect(),
        maps_digest: crate::digest(maps)?,
        map_summary: maps_summary(maps),
        seed: 6300,
        updates: 200,
        aux_weight: AUX_WEIGHT,
        conditions: Condition::all()
            .iter()
            .map(|c| c.name().to_string())
            .collect(),
        initial_model_sha256,
        initial_parameter_digest,
        digest: String::new(),
    };
    p.seal()?;
    Ok(p)
}
/// Everything the fitting/endpoint code needs, re-derived and cross-checked.
pub struct Frozen {
    pub p0: Plan,
    pub maps: Maps,
    pub plan: ProbePlan,
    pub bank: Vec<Vec<Packet>>,
}
pub fn check_all_interventions(p0: &Plan, maps: &Maps) -> Result<usize> {
    let mut checked = 0;
    for ei in 0..p0.entries.len() {
        for policy in 0..2 {
            for c in Condition::all() {
                let p = intervened(p0, maps, ei, policy, c)?;
                checked += verify_intervention(p0, maps, ei, policy, c, &p)?;
            }
        }
    }
    Ok(checked)
}
pub fn load_frozen(
    p0_path: &Path,
    p0_binding: &Path,
    plan_path: &Path,
    plan_binding: &Path,
    contract: &Path,
    data: &V5Data,
) -> Result<Frozen> {
    let p0 = load_p0_plan(p0_path, p0_binding)?;
    verify_packets_against_data(&p0, data)?;
    let plan: ProbePlan = serde_json::from_slice(&std::fs::read(plan_path)?)?;
    plan.validate()?;
    let b: Value = serde_json::from_slice(&std::fs::read(plan_binding)?)?;
    ensure!(
        b["schema"] == "v6_objective_probe_plan_binding_v2"
            && b["source_sha"] == crate::SOURCE
            && b["plan_raw_sha256"] == recur64_v5::stage::hash_file(plan_path)?
            && b["plan_digest"] == plan.digest
            && plan.contract_sha256 == recur64_v5::stage::hash_file(contract)?,
        "committed probe plan binding / contract hash mismatch"
    );
    let maps = crate::intervene::all_maps(&p0)?;
    ensure!(
        crate::digest(&maps)? == plan.maps_digest,
        "frozen donor mappings changed"
    );
    ensure!(
        plan.panel.iter().zip(&p0.entries).all(|(r, e)| {
            r.id == e.id
                && r.index == e.index
                && r.cell == e.cell
                && r.baseline_correct == e.baseline_correct
                && r.packet_digests
                    == e.packets
                        .iter()
                        .map(|p| p.digest.clone())
                        .collect::<Vec<_>>()
        }),
        "panel/packet identity differs from frozen plan"
    );
    check_all_interventions(&p0, &maps)?;
    let mut bank = Vec::new();
    for ei in 0..p0.entries.len() {
        bank.push(
            (0..2)
                .map(|p| intervened(&p0, &maps, ei, p, Condition::SuccessorOnlyE002))
                .collect::<Result<Vec<_>>>()?,
        );
    }
    Ok(Frozen {
        p0,
        maps,
        plan,
        bank,
    })
}

// ------------------------------------------------------------------ objective

pub struct Views<B: Backend> {
    pub real: Inputs<B>,
    pub successor: Option<Inputs<B>>,
    pub correct: Tensor<B, 2, Bool>,
}
/// Real and (optionally) successor-only inputs share one frozen-root pass; the
/// frozen B0 outputs and structure are identical by construction.
pub fn make_views<B: AutodiffBackend>(
    base: &crate::baseline::FrozenBase<B>,
    data: &V5Data,
    plan: &Plan,
    bank: Option<&[Vec<Packet>]>,
    entries: [usize; 2],
    policy: usize,
    device: &B::Device,
) -> Result<Views<B>> {
    data.require_role(recur64_v5::native_data_v2::Role::Train)?;
    let indices: Vec<usize> = entries.iter().map(|&e| plan.entries[e].index).collect();
    for &e in &entries {
        ensure!(
            data.position(plan.entries[e].index).id == plan.entries[e].id
                && data.position(plan.entries[e].index).legal
                    == plan.entries[e].packets[policy].legal,
            "frozen packet/data action alignment"
        );
    }
    let roots = data.roots(&indices)?;
    let refs: Vec<_> = roots.iter().collect();
    let out = base
        .valid()
        .base_root(&RootInputs::<B::InnerBackend>::from_roots(&refs, device)?);
    let w = out.z0.dims()[1];
    let real: Vec<Packet> = entries
        .iter()
        .map(|&e| plan.entries[e].packets[policy].clone())
        .collect();
    for (p, r) in real.iter().zip(&roots) {
        p.verify_against_root(r)?;
    }
    let build = |packets: &[Packet]| {
        inputs(
            packets,
            Tensor::from_inner(out.hypotheses.clone()),
            Tensor::from_inner(out.z0.clone()),
            device,
        )
    };
    let successor = match bank {
        Some(bank) => {
            let packets: Vec<Packet> = entries.iter().map(|&e| bank[e][policy].clone()).collect();
            Some(build(&packets)?)
        }
        None => None,
    };
    Ok(Views {
        real: build(&real)?,
        successor,
        correct: crate::p0::targets::<B>(data, &indices, w, device)?,
    })
}

/// Treatment auxiliary: `A(d)`, `d = delta_G - delta_Ssucc(G)`, both with gradient.
pub fn differential_aux<B: Backend>(
    g: &Output<B>,
    s: &Output<B>,
    eligible: Tensor<B, 2, Bool>,
    correct: Tensor<B, 2, Bool>,
) -> Tensor<B, 1> {
    g.iteration_delta
        .iter()
        .zip(&s.iteration_delta)
        .map(|(a, b)| loss::eligible_bce(a.clone() - b.clone(), eligible.clone(), correct.clone()))
        .reduce(|a, b| a + b)
        .unwrap()
        .div_scalar(g.iteration_delta.len() as f32)
}
/// One microbatch: returns `(policy, auxiliary)`; composite = policy + 0.5 * auxiliary.
pub fn microbatch<B: AutodiffBackend>(
    m: &Reader<B>,
    objective: Objective,
    v: &Views<B>,
    phase: &mut dyn FnMut(&str),
) -> (Tensor<B, 1>, Tensor<B, 1>, Output<B>) {
    let g = {
        let mut f = |n: &str| phase(&format!("factual:{n}"));
        m.forward(&v.real, 1, false, false, &mut f)
    };
    match objective {
        Objective::Control => {
            let (p, a) = loss::components(&g, &v.real, v.correct.clone());
            (p, a, g)
        }
        Objective::Treatment => {
            let s = {
                let mut f = |n: &str| phase(&format!("successor:{n}"));
                m.forward(
                    v.successor
                        .as_ref()
                        .expect("treatment needs a successor view"),
                    1,
                    false,
                    false,
                    &mut f,
                )
            };
            let p = recur64_v5::loss::correct_set_loss(
                g.logits.clone(),
                v.real.legal.clone(),
                v.correct.clone(),
            );
            let a = differential_aux(&g, &s, v.real.eligible.clone(), v.correct.clone());
            (p, a, g)
        }
    }
}
/// Single (non-accumulated) update used by qualification; mirrors `qualify::step`.
pub fn step<B: AutodiffBackend>(
    m: Reader<B>,
    o: &mut Opt<B>,
    objective: Objective,
    v: &Views<B>,
    lr: f64,
    profile: bool,
) -> Result<(Reader<B>, Value)> {
    let d = v.real.states.device();
    let mut phases = Vec::new();
    let mut last = Instant::now();
    let mut failure = None;
    let mut phase = |name: &str| {
        if profile {
            if let Err(e) = B::sync(&d) {
                failure = Some(format!("{e:?}"));
            }
            phases.push((name.to_owned(), last.elapsed().as_secs_f64()));
            last = Instant::now();
        }
    };
    let (policy, aux, g) = microbatch(&m, objective, v, &mut phase);
    let to_host = |t: Tensor<B, 2>| {
        t.into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("{e:?}"))
    };
    let logits = to_host(g.logits.clone())?;
    let raw = to_host(g.raw_delta.clone())?;
    let (pv, av) = (scalar(policy.clone()), scalar(aux.clone()));
    let loss = policy + aux.mul_scalar(AUX_WEIGHT as f32);
    let lv = scalar(loss.clone());
    ensure!(lv.is_finite(), "nonfinite loss");
    phase("policy_and_auxiliary");
    let grads = GradientsParams::from_grads(loss.backward(), &m);
    phase("backward");
    let gs = qualify::gradients(&m, &grads)?;
    phase("gradient_readback");
    let m = o.step(lr, m, grads);
    phase("AdamW");
    ensure!(failure.is_none(), "CUDA fence failure: {failure:?}");
    Ok((
        m,
        json!({"logits": crate::digest(&logits)?, "raw_delta": crate::digest(&raw)?, "loss": lv,
               "policy_loss": pv, "auxiliary_loss": av, "gradient_digest": crate::digest(&gs)?,
               "gradients": gs, "phases": phases}),
    ))
}

// ------------------------------------------------------------ initial reader

fn recorder() -> NamedMpkFileRecorder<FullPrecisionSettings> {
    NamedMpkFileRecorder::<FullPrecisionSettings>::new()
}
/// ONE canonical fresh seed-6300 one-pass reader; never loaded from any trained weights.
pub fn create_initial<B: AutodiffBackend>(dir: &Path, device: &B::Device) -> Result<Value> {
    ensure!(!dir.exists(), "canonical initial reader cannot overwrite");
    std::fs::create_dir_all(dir)?;
    B::seed(device, 6300);
    let m = Reader::<B>::new(Arm::OnePass, device);
    let inventory = qualify::inventory(&m)?;
    recorder().record(m.clone().into_record(), dir.join("model"))?;
    Ok(json!({
        "model_sha256": recur64_v5::stage::hash_file(&dir.join("model.mpk"))?,
        "parameter_digest": crate::digest(&inventory)?,
        "reader_parameters": inventory.iter().map(|x| x.elements).sum::<usize>(),
        "seed": 6300,
    }))
}
pub fn load_initial<B: AutodiffBackend>(
    dir: &Path,
    plan: &ProbePlan,
    device: &B::Device,
) -> Result<Reader<B>> {
    ensure!(
        recur64_v5::stage::hash_file(&dir.join("model.mpk"))? == plan.initial_model_sha256,
        "canonical initial reader hash mismatch"
    );
    let m = Reader::<B>::new(Arm::OnePass, device)
        .load_record(recorder().load(dir.join("model"), device)?);
    ensure!(
        crate::digest(&qualify::inventory(&m)?)? == plan.initial_parameter_digest,
        "canonical initial reader parameter digest mismatch"
    );
    Ok(m)
}

// ---------------------------------------------------------------------- fitting

#[derive(Serialize, Deserialize)]
struct Checkpoint {
    source: String,
    objective: String,
    arm: String,
    config: String,
    plan: String,
    update: usize,
    model_sha: String,
    optimizer_sha: String,
    wall_seconds: f64,
    precision: String,
    backend: String,
    microbatch: usize,
    effective_batch: usize,
    seed: u64,
    episode_digest: String,
    history: Vec<Value>,
}
pub struct Run<'a> {
    pub objective: Objective,
    pub frozen: &'a Frozen,
    pub initial: &'a Path,
    pub dir: &'a Path,
    pub resume: bool,
    pub max_minutes: f64,
}
pub fn learn<B: AutodiffBackend>(
    base: &crate::baseline::FrozenBase<B>,
    data: &V5Data,
    run: Run,
    device: &B::Device,
) -> Result<Value> {
    let Run {
        objective,
        frozen,
        initial,
        dir,
        resume,
        max_minutes,
    } = run;
    data.require_role(recur64_v5::native_data_v2::Role::Train)?;
    frozen.plan.validate()?;
    ensure!(
        max_minutes > 0. && max_minutes <= 45.,
        "bounded invocation required"
    );
    let arm = objective.name();
    let bank = (objective == Objective::Treatment).then_some(frozen.bank.as_slice());
    let before = base.baseline_parameter_digest()?;
    let matrix_at = |update: usize| dir.join(format!("endpoint-{update:03}.json"));
    let (mut m, mut opt, mut update, mut history, wall) = if resume {
        let meta: Checkpoint = serde_json::from_slice(&std::fs::read(dir.join("latest.json"))?)?;
        let at = dir.join(format!("update-{:03}", meta.update));
        ensure!(
            meta.source == crate::SOURCE
                && meta.objective == OBJECTIVE
                && meta.arm == arm
                && meta.config == crate::packet::config_digest()?
                && meta.plan == frozen.plan.digest
                && meta.precision == "fp32"
                && meta.backend == "cuda"
                && meta.microbatch == 2
                && meta.effective_batch == 24
                && meta.seed == 6300
                && meta.episode_digest == frozen.plan.episode_digest
                && meta.history.len() == meta.update
                && meta.update < 200
                && meta.model_sha == recur64_v5::stage::hash_file(&at.join("model.mpk"))?
                && meta.optimizer_sha == recur64_v5::stage::hash_file(&at.join("optimizer.mpk"))?,
            "resume identity/refusal"
        );
        let (m, o) = qualify::restore(Arm::OnePass, &at, device)?;
        (m, o, meta.update, meta.history, meta.wall_seconds)
    } else {
        ensure!(!dir.exists(), "fresh run cannot overwrite");
        std::fs::create_dir_all(dir)?;
        let m = load_initial::<B>(initial, &frozen.plan, device)?;
        let o = recur64_model::train::adamw::<B, Reader<B>>();
        qualify::save(&m, &o, &dir.join("update-000"))?;
        let mut matrix = probe_eval::evaluate(
            &m,
            base,
            data,
            &frozen.p0,
            &frozen.maps,
            &frozen.plan.maps_digest,
            arm,
            0,
            device,
        )?;
        bind_provenance(&mut matrix, &frozen.plan, &dir.join("update-000/model.mpk"))?;
        std::fs::write(matrix_at(0), serde_json::to_vec(&matrix)?)?;
        (m, o, 0, vec![], 0.)
    };
    let start = Instant::now();
    while update < 200 && start.elapsed().as_secs_f64() < max_minutes * 60. {
        ensure!(
            wall + start.elapsed().as_secs_f64() < 7200.,
            "per arm fitting budget exceeded"
        );
        let tick = Instant::now();
        let mut accumulator = GradientsAccumulator::new();
        let (mut policy_total, mut aux_total) = (0., 0.);
        for k in 0..12 {
            let (index, second, policy) = episode(update, k);
            let v = make_views::<B>(
                base,
                data,
                &frozen.p0,
                bank,
                [index, second],
                policy,
                device,
            )?;
            let (p, a, _) = microbatch(&m, objective, &v, &mut |_| {});
            policy_total += scalar(p.clone()) / 12.;
            aux_total += scalar(a.clone()) / 12.;
            let loss = (p + a.mul_scalar(AUX_WEIGHT as f32)).div_scalar(12.);
            let g = GradientsParams::from_grads(loss.backward(), &m);
            let _ = qualify::gradients(&m, &g)?;
            accumulator.accumulate(&m, g);
        }
        ensure!(
            policy_total.is_finite() && aux_total.is_finite(),
            "nonfinite fitting loss"
        );
        let lr = recur64_runtime::learner::lr_at(update as u64, 1e-3, 20, 200);
        m = opt.step(lr, m, accumulator.grads());
        B::sync(device).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        update += 1;
        let device_used = qualify::memory()?;
        ensure!(
            device_used as f64 <= 3.2 * 1024.,
            "device-wide memory cap exceeded"
        );
        history.push(
            json!({"update": update, "policy_loss": policy_total, "auxiliary_loss": aux_total,
                            "lr": lr, "examples": 24, "seconds": tick.elapsed().as_secs_f64(), "device_used_mib": device_used}),
        );
        if update % 10 == 0 || update == 200 || start.elapsed().as_secs_f64() >= max_minutes * 60. {
            let at = dir.join(format!("update-{update:03}"));
            ensure!(!at.exists(), "checkpoint collision");
            qualify::save(&m, &opt, &at)?;
            let meta = Checkpoint {
                source: crate::SOURCE.into(),
                objective: OBJECTIVE.into(),
                arm: arm.into(),
                config: crate::packet::config_digest()?,
                plan: frozen.plan.digest.clone(),
                update,
                model_sha: recur64_v5::stage::hash_file(&at.join("model.mpk"))?,
                optimizer_sha: recur64_v5::stage::hash_file(&at.join("optimizer.mpk"))?,
                wall_seconds: wall + start.elapsed().as_secs_f64(),
                precision: "fp32".into(),
                backend: "cuda".into(),
                microbatch: 2,
                effective_batch: 24,
                seed: 6300,
                episode_digest: frozen.plan.episode_digest.clone(),
                history: history.clone(),
            };
            std::fs::write(at.join("state.json"), serde_json::to_vec(&meta)?)?;
            std::fs::write(dir.join("latest.json"), serde_json::to_vec(&meta)?)?;
            eprintln!("{arm} update {update}/200 policy={policy_total} aux={aux_total}");
        }
    }
    ensure!(
        before == base.baseline_parameter_digest()?,
        "baseline mutated during fitting"
    );
    if update < 200 {
        return Ok(json!({"bounded_stop": true, "update": update, "resume_required": true}));
    }
    let mut last = probe_eval::evaluate(
        &m,
        base,
        data,
        &frozen.p0,
        &frozen.maps,
        &frozen.plan.maps_digest,
        arm,
        200,
        device,
    )?;
    bind_provenance(&mut last, &frozen.plan, &dir.join("update-200/model.mpk"))?;
    std::fs::write(matrix_at(200), serde_json::to_vec(&last)?)?;
    let first: probe_eval::Matrix = serde_json::from_slice(&std::fs::read(matrix_at(0))?)?;
    let analysis = probe_eval::analyze(&first, &last)?;
    std::fs::write(
        dir.join("analysis.json"),
        serde_json::to_vec_pretty(&analysis)?,
    )?;
    let secs: Vec<f64> = history
        .iter()
        .filter_map(|h| h["seconds"].as_f64())
        .collect();
    let views = if objective == Objective::Treatment {
        2
    } else {
        1
    };
    Ok(json!({
        "schema": "v6_objective_probe_arm_result_v2",
        "source_sha": crate::SOURCE, "objective": OBJECTIVE, "arm": arm,
        "config_digest": crate::packet::config_digest()?, "plan_digest": frozen.plan.digest,
        "update": 200, "baseline_exact": true, "disposable": true, "weights_reused": false,
        "initial_model_sha256": frozen.plan.initial_model_sha256,
        "model_update0_sha256": recur64_v5::stage::hash_file(&dir.join("update-000/model.mpk"))?,
        "optimizer_update0_sha256": recur64_v5::stage::hash_file(&dir.join("update-000/optimizer.mpk"))?,
        "model_update200_sha256": recur64_v5::stage::hash_file(&dir.join("update-200/model.mpk"))?,
        "optimizer_update200_sha256": recur64_v5::stage::hash_file(&dir.join("update-200/optimizer.mpk"))?,
        "checkpoint_metadata_sha256": recur64_v5::stage::hash_file(&dir.join("latest.json"))?,
        "endpoint_000_sha256": recur64_v5::stage::hash_file(&matrix_at(0))?,
        "endpoint_200_sha256": recur64_v5::stage::hash_file(&matrix_at(200))?,
        "analysis_sha256": recur64_v5::stage::hash_file(&dir.join("analysis.json"))?,
        "compute": {"root_episodes": 4800, "gradient_bearing_factual_views": 4800 * views,
                    "null_streams": 4800 * views, "extra_gradient_views_vs_control": 4800 * (views - 1),
                    "compute_matched": false,
                    "fit_wall_seconds": wall + start.elapsed().as_secs_f64(),
                    "update_seconds_mean": secs.iter().sum::<f64>() / secs.len().max(1) as f64,
                    "update_seconds_max": secs.iter().cloned().fold(0., f64::max),
                    "device_used_peak_sampled_mib": history.iter().filter_map(|h| h["device_used_mib"].as_u64()).max()},
        "history": history,
    }))
}

/// Bind an endpoint matrix to this launch plan and to the exact checkpoint it evaluated.
fn bind_provenance(m: &mut probe_eval::Matrix, plan: &ProbePlan, model: &Path) -> Result<()> {
    m.probe_plan_digest = plan.digest.clone();
    m.model_sha256 = recur64_v5::stage::hash_file(model)?;
    Ok(())
}
