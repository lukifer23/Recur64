//! TRAIN-only frozen-forward investigation. No optimizer or training capability.
#[path = "support/forward_trace.rs"]
mod trace;
use anyhow::{Result, ensure};
use burn::module::{AutodiffModule, Module};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::TensorData;
use clap::Parser;
use recur64_v6::{
    model::{Arm, Reader},
    p0::{Entry, Plan},
    packet::Packet,
};
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    time::Instant,
};
const PRODUCER: &str = "42f47b852451d59645dcc50120e4a426b034f493";
#[derive(Parser)]
struct Args {
    #[arg(long)]
    arm: String,
    #[arg(long)]
    update: usize,
    #[arg(long)]
    mode: String,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    dev_custody: PathBuf,
    #[arg(long)]
    confirm_custody: PathBuf,
    #[arg(long)]
    stage_a: PathBuf,
}
fn file(p: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&std::fs::read(p)?)?)
}
fn hash(p: &Path) -> Result<String> {
    recur64_v5::stage::hash_file(p)
}
fn data<const D: usize, B: Backend>(t: Tensor<B, D>) -> Result<Vec<f32>> {
    t.into_data()
        .to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("{e:?}"))
}
fn choose(z: &[f32]) -> usize {
    (0..z.len())
        .max_by(|&a, &b| z[a].total_cmp(&z[b]).then(b.cmp(&a)))
        .unwrap()
}
fn structure(p: &Packet) -> Value {
    let mut v = serde_json::to_value(p).unwrap();
    v.as_object_mut().unwrap().remove("digest");
    for n in v["nodes"].as_array_mut().unwrap() {
        n.as_object_mut().unwrap().remove("payload");
    }
    v
}
fn control(e: &Entry, plan: &Plan, policy: usize, name: &str) -> Result<(Packet, Vec<Value>)> {
    let mut p = e.packets[policy].clone();
    let before = structure(&p);
    let mut mapping = Vec::new();
    let seed = match name {
        "shuffle_original" => Some(0x7A60_E002),
        "shuffle_e102" => Some(0x7A60_E102),
        "shuffle_e202" => Some(0x7A60_E202),
        "shuffle_e302" => Some(0x7A60_E302),
        _ => None,
    };
    if let Some(seed) = seed {
        for n in &mut p.nodes {
            let mut donors: Vec<_> = plan
                .entries
                .iter()
                .filter(|d| d.id != e.id && d.cell == e.cell)
                .flat_map(|d| {
                    d.packets[policy]
                        .nodes
                        .iter()
                        .filter(|v| v.root_to_move == n.root_to_move)
                        .map(move |v| (&d.id, v))
                })
                .collect();
            ensure!(!donors.is_empty(), "no same-turn different-root donor");
            let dist = donors
                .iter()
                .map(|(_, v)| v.depth.abs_diff(n.depth))
                .min()
                .unwrap();
            donors.retain(|(_, v)| v.depth.abs_diff(n.depth) == dist);
            donors.sort_by_key(|(id, v)| ((*id).clone(), v.depth, v.path.clone(), v.storage_id));
            let h = recur64_v6::packet::key(&e.id, &n.path, n.depth as u16, seed);
            let at = u64::from_le_bytes(h[..8].try_into().unwrap()) as usize % donors.len();
            let (id, v) = donors[at];
            mapping.push(json!({"seed":seed,"recipient_id":e.id,"recipient_path":n.path,"recipient_depth":n.depth,"recipient_turn":n.root_to_move,"donor_id":id,"donor_path":v.path,"donor_depth":v.depth,"donor_turn":v.root_to_move,"depth_delta":dist,"pool":donors.len()}));
            n.payload = v.payload.clone();
        }
    }
    // Tensor interventions have no claim of being valid chess states.
    for n in &mut p.nodes {
        if matches!(name, "successor_frames_zero" | "root_history_frames_zero") {
            for square in 0..64 {
                for frame in 0..8 {
                    let successor = frame < n.depth as usize;
                    if (name == "successor_frames_zero" && successor)
                        || (name == "root_history_frames_zero" && !successor)
                    {
                        n.payload.observation
                            [square * 119 + frame * 14..square * 119 + (frame + 1) * 14]
                            .fill(0.);
                    }
                }
            }
        }
        if matches!(name, "board_zero" | "board_flags_zero") {
            n.payload.observation.fill(0.);
        }
        if matches!(
            name,
            "flags_zero" | "board_flags_zero" | "carrier_flags_zero"
        ) {
            n.payload.flags.fill(0.);
        }
    }
    p.digest = p.content_digest()?;
    ensure!(structure(&p) == before, "intervention changed structure");
    Ok((p, mapping))
}
fn carrier<B: Backend>(b: usize, q: usize, device: &B::Device) -> Tensor<B, 3> {
    Tensor::from_data(
        TensorData::new(
            (0..b * q * 256)
                .map(|i| if i % 2 == 0 { 0.1f32 } else { -0.1 })
                .collect::<Vec<_>>(),
            [b, q, 256],
        ),
        device,
    )
}
fn source_guard() -> Result<()> {
    let git = |args: &[&str]| -> Result<String> {
        let o = std::process::Command::new("git").args(args).output()?;
        ensure!(o.status.success(), "git failed");
        Ok(String::from_utf8(o.stdout)?.trim().into())
    };
    ensure!(
        git(&["branch", "--show-current"])? == "experiment/hp-v6-branch-backup",
        "wrong branch"
    );
    ensure!(
        git(&[
            "log",
            "-1",
            "--format=%H",
            "--",
            "crates",
            "Cargo.toml",
            "Cargo.lock",
            "configs"
        ])? == recur64_v6::SOURCE,
        "binary source mismatch"
    );
    ensure!(
        git(&[
            "status",
            "--porcelain",
            "--",
            "crates",
            "Cargo.toml",
            "Cargo.lock",
            "configs"
        ])?
        .is_empty(),
        "dirty scientific source"
    );
    Ok(())
}
#[cfg(feature = "cuda")]
fn run(a: &Args) -> Result<Value> {
    type AD = burn::backend::Autodiff<burn::backend::Cuda>;
    type B = burn::backend::Cuda;
    let start = Instant::now();
    let device = Default::default();
    ensure!(
        matches!(a.arm.as_str(), "principal" | "one-pass")
            && matches!(a.update, 0 | 200)
            && matches!(a.mode.as_str(), "reproduce" | "diagnose"),
        "exact frozen capability only"
    );
    let custody =
        recur64_v5::data::verify_local_boundaries(&a.data, &a.dev_custody, &a.confirm_custody)?;
    let data_set = recur64_v5::data::V5Data::load_train(&a.data)?;
    let base = recur64_v6::packet::import_base::<AD>(&a.stage_a, &device, "cuda")?;
    let before = base.baseline_parameter_digest()?;
    let binding = file(Path::new("docs/evidence/v6-p0/plan-binding-42f47b8.json"))?;
    let plan_path = Path::new("runs/v6-p0/frozen-plan-42f47b8.json");
    ensure!(
        hash(plan_path)? == binding["raw_sha256"].as_str().unwrap(),
        "raw plan mismatch"
    );
    let plan: Plan = serde_json::from_slice(&std::fs::read(plan_path)?)?;
    ensure!(
        plan.source_sha == PRODUCER
            && plan.config_digest == recur64_v6::packet::config_digest()?
            && plan.digest == binding["plan_digest"]
            && plan.entries.len() == 96,
        "plan/source/config mismatch"
    );
    let mut checked = plan.clone();
    checked.digest.clear();
    ensure!(
        recur64_v6::digest(&checked)? == plan.digest,
        "plan scientific content tampered"
    );
    let arm = if a.arm == "principal" {
        Arm::SharedBackup
    } else {
        Arm::OnePass
    };
    let r = if arm == Arm::SharedBackup { 4 } else { 1 };
    let dir = PathBuf::from(format!("runs/v6-p0/seed-6300/{}", a.arm));
    let at = dir.join(format!("update-{:03}", a.update));
    let receipt = file(&PathBuf::from(format!(
        "docs/evidence/v6-p0/{}-result.json",
        a.arm
    )))?;
    let key = if a.update == 0 {
        "checkpoint_update0_sha256"
    } else {
        "checkpoint_update200_sha256"
    };
    ensure!(
        hash(&at.join("model.mpk"))? == receipt[key],
        "frozen model hash mismatch"
    );
    ensure!(
        hash(&dir.join("latest.json"))? == receipt["checkpoint_metadata_sha256"]
            && hash(&dir.join("update-200/optimizer.mpk"))?
                == receipt["optimizer_update200_sha256"],
        "metadata/optimizer mismatch"
    );
    let inv = file(Path::new(
        "docs/evidence/v6-p0-diagnostics/artifact-verification.json",
    ))?;
    for c in inv["checks"].as_array().unwrap() {
        if c["arm"] == a.arm {
            ensure!(
                hash(Path::new(c["path"].as_str().unwrap()))? == c["sha256"],
                "frozen inventory mismatch"
            );
        }
    }
    let rec = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let reader = Reader::<AD>::new(arm, &device).load_record(rec.load(at.join("model"), &device)?);
    let old_path = dir.join(format!("endpoint-{:03}.json", a.update));
    let old = file(&old_path)?;
    if a.mode == "reproduce" {
        let rows = recur64_v6::p0::endpoint(&reader, &base, &data_set, &plan, &device)?;
        let new = serde_json::to_value(&rows)?;
        ensure!(
            new == old,
            "original endpoint not exact; stop interpretation"
        );
        ensure!(
            before == base.baseline_parameter_digest()?,
            "baseline changed"
        );
        return Ok(
            json!({"schema":"v6_frozen_endpoint_reproduction_v1","diagnostic_source":recur64_v6::SOURCE,"producer_source":PRODUCER,"arm":a.arm,"update":a.update,"rows":192,"all_original_fields_exact":true,"baseline_exact":true,"model_sha256":hash(&at.join("model.mpk"))?,"original_endpoint_sha256":hash(&old_path)?,"cuda_fp32_microbatch":2,"custody":custody,"pass":true,"seconds":start.elapsed().as_secs_f64()}),
        );
    }
    let parent = a.output_dir.parent().unwrap();
    for arm in ["principal", "one-pass"] {
        for u in [0, 200] {
            let p = parent.join(format!("{arm}-{u:03}/reproduction.json"));
            let v = file(&p)?;
            ensure!(
                v["pass"] == true && v["diagnostic_source"] == recur64_v6::SOURCE,
                "all four reproduction gates required first"
            );
        }
    }
    let manifest_path = Path::new("docs/evidence/v6-p0-diagnostics/intervention-manifest.json");
    let manifest = file(manifest_path)?;
    ensure!(
        manifest["producer_source"] == PRODUCER && manifest["plan_digest"] == plan.digest,
        "manifest mismatch"
    );
    let conditions: Vec<_> = manifest["conditions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    let m = reader.valid();
    let mut rows = Vec::new();
    let mut parity_count = 0;
    for es in plan.entries.chunks(2) {
        let roots = data_set.roots(&es.iter().map(|e| e.index).collect::<Vec<_>>())?;
        let refs = roots.iter().collect::<Vec<_>>();
        let root = base
            .valid()
            .base_root(&recur64_v5::model::RootInputs::<B>::from_roots(
                &refs, &device,
            )?);
        for policy in 0..2 {
            for (e, state) in es.iter().zip(&roots) {
                ensure!(
                    e.id == data_set.position(e.index).id
                        && e.packets[policy].legal == data_set.position(e.index).legal
                        && e.packets[policy].source_sha == PRODUCER,
                    "packet identity"
                );
                e.packets[policy].verify_against_root(state)?;
            }
            for name in &conditions {
                ensure!(
                    start.elapsed().as_secs_f64() < 29. * 60.,
                    "bounded diagnostic stop"
                );
                let changed = es
                    .iter()
                    .map(|e| control(e, &plan, policy, name))
                    .collect::<Result<Vec<_>>>()?;
                let packets = changed.iter().map(|(p, _)| p.clone()).collect::<Vec<_>>();
                let mut i = recur64_v6::model::inputs::<B>(
                    &packets,
                    root.hypotheses.clone(),
                    root.z0.clone(),
                    &device,
                )?;
                if *name == "owner_zero" {
                    i.baseline_owner = Tensor::zeros(i.baseline_owner.dims(), &device);
                }
                let [b, q, _, _] = i.states.dims();
                let w = i.width;
                let encoded = trace::rows(
                    &m.slots,
                    m.encoder
                        .forward(i.states.clone(), i.flags.clone(), i.node_mask.clone())
                        .reshape([b, q, 1024]),
                );
                let anchor = if name.starts_with("carrier_") {
                    carrier(b, q, &device).mask_fill(
                        i.node_mask
                            .clone()
                            .bool_not()
                            .reshape([b, q, 1])
                            .expand([b, q, 256]),
                        0.,
                    )
                } else {
                    encoded
                };
                let f = trace::stream(
                    &m,
                    &i,
                    anchor.clone(),
                    r,
                    *name != "all_null",
                    *name == "turn_blind",
                    &mut |_| {},
                );
                let n = trace::stream(
                    &m,
                    &i,
                    anchor.clone(),
                    r,
                    false,
                    *name == "turn_blind",
                    &mut |_| {},
                );
                let iteration_delta = f
                    .iter()
                    .zip(&n)
                    .map(|(f, n)| (f.clone() - n.clone()).mask_fill(i.legal.clone().bool_not(), 0.))
                    .collect::<Vec<_>>();
                let raw = iteration_delta.last().unwrap().clone();
                let mean = (raw.clone().sum_dim(1) / i.legal.clone().float().sum_dim(1))
                    .expand(raw.dims());
                let centered = (raw.clone() - mean).mask_fill(i.legal.clone().bool_not(), 0.);
                let logits = i.z0.clone() + centered.clone();
                if !name.starts_with("carrier_") {
                    let original = m.forward(
                        &i,
                        r,
                        *name == "all_null",
                        *name == "turn_blind",
                        &mut |_| {},
                    );
                    ensure!(
                        data(original.logits)? == data(logits.clone())?
                            && data(original.raw_delta)? == data(raw.clone())?
                            && data(original.centered)? == data(centered.clone())?,
                        "diagnostic mirror changed production output"
                    );
                    for (old, new) in original.iteration_delta.into_iter().zip(&iteration_delta) {
                        ensure!(data(old)? == data(new.clone())?, "iteration trace mismatch");
                    }
                    parity_count += 1;
                }
                let z = data(logits)?;
                let bz = data(i.z0.clone())?;
                let rv = data(raw)?;
                let cv = data(centered)?;
                let fs = f.into_iter().map(data).collect::<Result<Vec<_>>>()?;
                let ns = n.into_iter().map(data).collect::<Result<Vec<_>>>()?;
                let ds = iteration_delta
                    .into_iter()
                    .map(data)
                    .collect::<Result<Vec<_>>>()?;
                let av = data(anchor)?;
                for (j, e) in es.iter().enumerate() {
                    let p = data_set.position(e.index);
                    let count = p.legal.len();
                    let range = j * w..j * w + count;
                    let zz = &z[range.clone()];
                    let bzz = &bz[range.clone()];
                    let rr = &rv[range.clone()];
                    let correct = (0..count)
                        .map(|a| p.correct.contains(&(a as u32)))
                        .collect::<Vec<_>>();
                    let elig = e.packets[policy].eligibility();
                    let positive = correct
                        .iter()
                        .zip(&elig)
                        .filter(|(c, e)| **c && **e)
                        .count();
                    let negative = correct
                        .iter()
                        .zip(&elig)
                        .filter(|(c, e)| !**c && **e)
                        .count();
                    let best_correct = zz
                        .iter()
                        .zip(&correct)
                        .filter(|(_, c)| **c)
                        .map(|(v, _)| *v as f64)
                        .max_by(f64::total_cmp)
                        .unwrap();
                    let best_other = zz
                        .iter()
                        .zip(&correct)
                        .filter(|(_, c)| !**c)
                        .map(|(v, _)| *v as f64)
                        .max_by(f64::total_cmp);
                    let candidate=(0..count).map(|a|json!({"index":a,"action_id":p.legal[a],"minimum_mate_correct":correct[a],"eligible":elig[a],"baseline_logit":bzz[a],"final_logit":zz[a],"factual_score":fs.last().unwrap()[j*w+a],"null_score":ns.last().unwrap()[j*w+a],"raw_delta":rr[a],"centered_delta":cv[j*w+a],"iteration_factual":fs.iter().map(|v|v[j*w+a]).collect::<Vec<_>>(),"iteration_null":ns.iter().map(|v|v[j*w+a]).collect::<Vec<_>>(),"iteration_delta":ds.iter().map(|v|v[j*w+a]).collect::<Vec<_>>()})).collect::<Vec<_>>();
                    let original = &e.packets[policy];
                    let node_rows=original.nodes.iter().enumerate().map(|(k,v)|json!({"root_candidate":v.root_candidate,"path":v.path,"depth":v.depth,"root_to_move":v.root_to_move,"legal_replies":original.legal_counts[k],"observed_replies":original.children(k).len(),"unknown":original.unknown(k),"original_flags":v.payload.flags,"effective_flags":packets[j].nodes[k].payload.flags,"anchor_rms":(av[(j*q+k)*256..(j*q+k+1)*256].iter().map(|v|(*v as f64).powi(2)).sum::<f64>()/256.).sqrt()})).collect::<Vec<_>>();
                    rows.push(json!({"id":e.id,"cell":e.cell,"policy":policy,"arm":a.arm,"checkpoint":a.update,"condition":name,"baseline_correct":e.baseline_correct,"baseline_selected_index":choose(bzz),"baseline_selected_action":p.legal[choose(bzz)],"selected_index":choose(zz),"selected_action":p.legal[choose(zz)],"correct":correct[choose(zz)],"set_loss":recur64_v5::loss::reference_set_loss(zz,&vec![true;count],&correct),"margin":best_other.map(|v|best_correct-v),"correction_range":rr.iter().copied().fold(f32::NEG_INFINITY,f32::max) as f64-rr.iter().copied().fold(f32::INFINITY,f32::min) as f64,"eligible_positive":positive,"eligible_negative":negative,"packet_digest":original.digest,"donor_mapping":changed[j].1,"nodes":node_rows,"candidates":candidate}));
                }
            }
        }
    }
    ensure!(
        before == base.baseline_parameter_digest()?,
        "baseline changed"
    );
    Ok(
        json!({"schema":"v6_frozen_forward_diagnostics_v1","diagnostic_source":recur64_v6::SOURCE,"producer_source":PRODUCER,"config_digest":plan.config_digest,"plan_digest":plan.digest,"manifest_sha256":hash(manifest_path)?,"arm":a.arm,"update":a.update,"graph_free":true,"optimizer_steps":0,"cuda_fp32_microbatch":2,"production_parity_batches":parity_count,"baseline_exact":true,"rows":rows,"seconds":start.elapsed().as_secs_f64()}),
    )
}
fn main() -> Result<()> {
    source_guard()?;
    let a = Args::parse();
    std::fs::create_dir_all(&a.output_dir)?;
    let path = a.output_dir.join(if a.mode == "reproduce" {
        "reproduction.json"
    } else {
        "diagnostics.json"
    });
    ensure!(!path.exists(), "no overwrite");
    std::thread::Builder::new().stack_size(64*1024*1024).spawn(move||{
        #[cfg(feature="cuda")]let result=run(&a);
        #[cfg(not(feature="cuda"))]let result:Result<Value>=Err(anyhow::anyhow!("CUDA required; no CPU fallback"));
        let value=match &result{Ok(v)=>v.clone(),Err(e)=>json!({"schema":"v6_frozen_diagnostic_failure_v1","source":recur64_v6::SOURCE,"error":format!("{e:#}"),"pass":false,"optimizer_steps":0})};
        std::fs::write(&path,serde_json::to_vec(&value)?)?;result.map(|_|())
    })?.join().map_err(|_|anyhow::anyhow!("native worker panic; STOP"))?
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tensor_interventions_preserve_packet_structure_and_targets_absent() {
        let raw =
            include_bytes!("../../../docs/evidence/v6-p0-diagnostics/intervention-manifest.json");
        let manifest: Value = serde_json::from_slice(raw).unwrap();
        assert_eq!(manifest["optimizer_steps"], 0);
        let plan: Plan = serde_json::from_slice(
            &std::fs::read("../../runs/v6-p0/frozen-plan-42f47b8.json").unwrap(),
        )
        .unwrap();
        let e = &plan.entries[0];
        for name in manifest["conditions"].as_array().unwrap() {
            let name = name.as_str().unwrap();
            let (p, m) = control(e, &plan, 0, name).unwrap();
            assert_eq!(structure(&p), structure(&e.packets[0]));
            if matches!(name, "successor_frames_zero" | "root_history_frames_zero") {
                for (new, old) in p.nodes.iter().zip(&e.packets[0].nodes) {
                    assert_eq!(new.payload.flags, old.payload.flags);
                    for square in 0..64 {
                        for feature in 0..119 {
                            let successor = feature / 14 < old.depth as usize;
                            let removed = feature < 112
                                && ((name == "successor_frames_zero" && successor)
                                    || (name == "root_history_frames_zero" && !successor));
                            assert_eq!(
                                new.payload.observation[square * 119 + feature],
                                if removed {
                                    0.
                                } else {
                                    old.payload.observation[square * 119 + feature]
                                }
                            );
                        }
                    }
                }
            }
            for d in m {
                assert_ne!(d["donor_id"], d["recipient_id"]);
                assert_eq!(d["donor_turn"], d["recipient_turn"]);
            }
        }
    }
    #[test]
    fn action_ties_choose_first_legal_index() {
        assert_eq!(choose(&[1., 1., 0.]), 0);
    }
    #[test]
    fn mirror_matches_production_and_zero_carrier_cancels() {
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                type B = burn::backend::Flex;
                let d = Default::default();
                let root =
                    recur64_core::GameState::from_fen("7k/8/8/8/8/8/3Q4/K1R5 w - - 0 1").unwrap();
                let w = root.legal_actions().len();
                let p = recur64_v6::packet::acquire(
                    &root,
                    "diagnostic-test",
                    &vec![0.; w],
                    recur64_v6::packet::Policy::ExploitTwo,
                    PRODUCER,
                )
                .unwrap();
                let mut i = recur64_v6::model::inputs::<B>(
                    &[p],
                    Tensor::zeros([1, w, 256], &d),
                    Tensor::zeros([1, w], &d),
                    &d,
                )
                .unwrap();
                let q = i.q;
                for arm in [Arm::SharedBackup, Arm::OnePass] {
                    let m = Reader::<B>::new(arm, &d);
                    let r = if arm == Arm::SharedBackup { 4 } else { 1 };
                    let x = trace::rows(
                        &m.slots,
                        m.encoder
                            .forward(i.states.clone(), i.flags.clone(), i.node_mask.clone())
                            .reshape([1, q, 1024]),
                    );
                    for blind in [false, true] {
                        let f = trace::stream(&m, &i, x.clone(), r, true, blind, &mut |_| {});
                        let n = trace::stream(&m, &i, x.clone(), r, false, blind, &mut |_| {});
                        let out = m.forward(&i, r, false, blind, &mut |_| {});
                        for (reference, (f, n)) in
                            out.iteration_delta.into_iter().zip(f.into_iter().zip(n))
                        {
                            assert_eq!(
                                data(reference).unwrap(),
                                data((f - n).mask_fill(i.legal.clone().bool_not(), 0.)).unwrap()
                            );
                        }
                    }
                    // Carrier-with-flags-zero stopping is false in both streams.
                    i.terminals = i.terminals.clone().bool_and(i.terminals.clone().bool_not());
                    let f = trace::stream(
                        &m,
                        &i,
                        Tensor::zeros([1, q, 256], &d),
                        r,
                        true,
                        false,
                        &mut |_| {},
                    );
                    let n = trace::stream(
                        &m,
                        &i,
                        Tensor::zeros([1, q, 256], &d),
                        r,
                        false,
                        false,
                        &mut |_| {},
                    );
                    assert!(
                        data(f.last().unwrap().clone() - n.last().unwrap().clone())
                            .unwrap()
                            .iter()
                            .all(|v| *v == 0.)
                    );
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
