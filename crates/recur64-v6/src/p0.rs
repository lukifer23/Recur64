//! Fixed TRAIN coverage, paired disposable plans and resumable learnability.
use crate::{
    model::{Arm, Inputs, Reader, inputs},
    packet::{Packet, Policy},
    qualify::scalar,
};
use burn::module::AutodiffModule;
use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::{Bool, TensorData, backend::AutodiffBackend};
use recur64_v5::{data::V5Data, model::RootInputs};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path, time::Instant};

pub type Prepared<B> = (
    Vec<Packet>,
    Inputs<B>,
    Tensor<B, 2, Bool>,
    serde_json::Value,
);
pub fn prepare<B: AutodiffBackend>(
    base: &crate::baseline::FrozenBase<B>,
    data: &V5Data,
    indices: &[usize],
    policy: Policy,
    device: &B::Device,
) -> anyhow::Result<Prepared<B>> {
    data.require_role(recur64_v5::native_data_v2::Role::Train)?;
    let roots = data.roots(indices)?;
    let references: Vec<_> = roots.iter().collect();
    let start = Instant::now();
    let root = RootInputs::<B::InnerBackend>::from_roots(&references, device)?;
    let out = base.valid().base_root(&root);
    B::sync(device).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let root_seconds = start.elapsed().as_secs_f64();
    let z = out
        .z0
        .clone()
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let w = out.z0.dims()[1];
    let start = Instant::now();
    let mut packets = Vec::new();
    for (j, (&idx, r)) in indices.iter().zip(&roots).enumerate() {
        data.validate_root_alignment(idx, r)?;
        packets.push(crate::packet::acquire(
            r,
            &data.position(idx).id,
            &z[j * w..j * w + data.position(idx).legal.len()],
            policy,
            crate::SOURCE,
        )?);
    }
    let query_seconds = start.elapsed().as_secs_f64();
    let input = inputs(
        &packets,
        Tensor::from_inner(out.hypotheses),
        Tensor::from_inner(out.z0),
        device,
    )?;
    let correct = targets(data, indices, w, device)?;
    Ok((
        packets,
        input,
        correct,
        serde_json::json!({"frozen_root_seconds":root_seconds,"acquisition_seconds":query_seconds}),
    ))
}
pub fn targets<B: Backend>(
    data: &V5Data,
    indices: &[usize],
    w: usize,
    device: &B::Device,
) -> anyhow::Result<Tensor<B, 2, Bool>> {
    data.require_role(recur64_v5::native_data_v2::Role::Train)?;
    let mut v = vec![false; indices.len() * w];
    for (j, &i) in indices.iter().enumerate() {
        let p = data.position(i);
        anyhow::ensure!(
            !p.correct.is_empty() && p.legal.len() <= w,
            "target alignment"
        );
        for &a in &p.correct {
            anyhow::ensure!((a as usize) < p.legal.len(), "correct index alignment");
            v[j * w + a as usize] = true;
        }
    }
    Ok(Tensor::from_data(
        TensorData::new(v, [indices.len(), w]),
        device,
    ))
}
fn choose(z: &[f32]) -> usize {
    z.iter()
        .enumerate()
        .max_by(|(i, a), (j, b)| a.total_cmp(b).then(j.cmp(i)))
        .unwrap()
        .0
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    pub index: usize,
    pub id: String,
    pub cell: String,
    pub baseline_correct: bool,
    pub packets: Vec<Packet>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    pub schema: String,
    pub source_sha: String,
    pub config_digest: String,
    pub train_digest: String,
    pub seed: u64,
    pub updates: u64,
    pub entries: Vec<Entry>,
    pub strata_support: BTreeMap<String, [usize; 2]>,
    pub initial_shapes_match: bool,
    pub episode_digest: String,
    pub digest: String,
}
pub fn episode(update: usize, k: usize) -> (usize, usize, usize) {
    let cell = k % 4;
    let local = (update * 3 + k / 4) % 12;
    (
        cell * 24 + local,
        cell * 24 + 12 + local,
        (k + update + update / 4) % 2,
    )
}
fn plan_digest(p: &Plan) -> anyhow::Result<String> {
    let mut p = p.clone();
    p.digest.clear();
    crate::digest(&p)
}
pub fn validate_plan(p: &Plan) -> anyhow::Result<()> {
    anyhow::ensure!(
        p.schema == "v6_competent_base_plan_v2"
            && p.source_sha == crate::SOURCE
            && p.config_digest == crate::packet::config_digest()?
            && p.train_digest == recur64_v5::data::TRAIN_DIGEST
            && p.seed == 6300
            && p.updates == 200
            && p.entries.len() == 96
            && p.initial_shapes_match
            && p.digest == plan_digest(p)?,
        "plan/source/data tamper mismatch"
    );
    let mut counts = BTreeMap::new();
    let mut ids = std::collections::BTreeSet::new();
    for e in &p.entries {
        anyhow::ensure!(
            ids.insert(e.id.clone())
                && e.packets.len() == 2
                && e.packets
                    .iter()
                    .enumerate()
                    .all(|(i, g)| g.generation_role == "p0_acquisition"
                        && g.root_id == e.id
                        && g.policy == Policy::all()[i]
                        && g.verify(crate::SOURCE).is_ok()),
            "plan packet/ID mismatch"
        );
        *counts
            .entry((e.cell.clone(), e.baseline_correct))
            .or_insert(0usize) += 1;
    }
    anyhow::ensure!(
        counts.len() == 8 && counts.values().all(|&c| c == 12),
        "exact panel strata required"
    );
    let cells = ["KQRvK M2", "KQRvK M3", "KRRvK M2", "KRRvK M3"];
    for (cell, entries) in cells.iter().zip(p.entries.chunks(24)) {
        anyhow::ensure!(
            entries.iter().all(|e| &e.cell == cell)
                && entries[..12].iter().all(|e| !e.baseline_correct)
                && entries[12..].iter().all(|e| e.baseline_correct),
            "episode cell/stratum order mismatch"
        );
    }
    let episodes: Vec<_> = (0..200)
        .map(|u| (0..12).map(|k| episode(u, k)).collect::<Vec<_>>())
        .collect();
    anyhow::ensure!(
        p.episode_digest == crate::digest(&episodes)?,
        "episode digest mismatch"
    );
    Ok(())
}
pub fn coverage<B: AutodiffBackend>(
    base: &crate::baseline::FrozenBase<B>,
    data: &V5Data,
    device: &B::Device,
) -> anyhow::Result<serde_json::Value> {
    let old: serde_json::Value =
        serde_json::from_str(include_str!("../../../docs/evidence/v6/train-panel.json"))?;
    let rows = old["positions"].as_array().unwrap();
    let mut groups: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    let mut ids = Vec::new();
    anyhow::ensure!(rows.len() == 216, "frozen coverage panel count");
    for r in rows {
        let idx = r["index"]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("panel index"))? as usize;
        anyhow::ensure!(
            r["position_id"] == data.position(idx).id,
            "coverage panel ID mismatch"
        );
    }
    for pair in rows.chunks(2) {
        let indices: Vec<_> = pair
            .iter()
            .map(|r| r["index"].as_u64().unwrap() as usize)
            .collect();
        for &i in &indices {
            ids.push(data.position(i).id.clone());
        }
        for policy in Policy::all() {
            let (packets, input, _, _) = prepare(base, data, &indices, policy, device)?;
            let z = input
                .z0
                .into_data()
                .to_vec::<f32>()
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            for (j, (&idx, p)) in indices.iter().zip(packets).enumerate() {
                let label = data.position(idx);
                let w = z.len() / indices.len();
                let right = label
                    .correct
                    .contains(&(choose(&z[j * w..j * w + label.legal.len()]) as u32));
                let cell = format!(
                    "{} M{}|{policy:?}|{}",
                    label.family,
                    label.mate_depth,
                    if right { "B0_right" } else { "B0_wrong" }
                );
                let correct_seen = label.correct.iter().any(|a| p.eligibility()[*a as usize]);
                let pos = label
                    .correct
                    .iter()
                    .filter(|a| p.eligibility()[**a as usize])
                    .count();
                let neg = p.eligibility().iter().filter(|b| **b).count() - pos;
                let mut legal = 0;
                let mut observed = 0;
                let mut unknown = 0;
                for (n, v) in p.nodes.iter().enumerate() {
                    unknown += p.unknown(n);
                    if !v.root_to_move {
                        legal += p.legal_counts[n];
                        observed += p.children(n).len();
                    }
                }
                let g=groups.entry(cell).or_insert(serde_json::json!({"roots":0,"correct_branch_present":0,"eligible_positive":0,"eligible_negative":0,"defender_legal":0,"defender_observed":0,"unknown_replies":0,"root_branches":0,"actual_q":0,"maximum_depth_sum":0}));
                for (key, v) in [
                    ("roots", 1),
                    ("correct_branch_present", usize::from(correct_seen)),
                    ("eligible_positive", pos),
                    ("eligible_negative", neg),
                    ("defender_legal", legal),
                    ("defender_observed", observed),
                    ("unknown_replies", unknown),
                    (
                        "root_branches",
                        p.eligibility().iter().filter(|b| **b).count(),
                    ),
                    ("actual_q", p.nodes.len()),
                    (
                        "maximum_depth_sum",
                        p.nodes.iter().map(|v| v.depth as usize).max().unwrap_or(0),
                    ),
                ] {
                    g[key] = serde_json::json!(g[key].as_u64().unwrap() + v as u64);
                }
            }
        }
    }
    Ok(
        serde_json::json!({"schema":"v6_p0_acquisition_census_v1","source_sha":crate::SOURCE,"config_digest":crate::packet::config_digest()?,"old_frozen_panel_id_digest":old["selected_id_digest"],"positions":ids.len(),"groups":groups,"reader_invocations":0,"policy_tuning":false}),
    )
}
fn selection(id: &str) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"v6_competent_base_drill_select_v1\0");
    h.update(0x7A60_D101u64.to_le_bytes());
    h.update(id.as_bytes());
    h.finalize().into()
}
pub fn freeze<B: AutodiffBackend>(
    base: &crate::baseline::FrozenBase<B>,
    data: &V5Data,
    device: &B::Device,
) -> anyhow::Result<Plan> {
    data.require_role(recur64_v5::native_data_v2::Role::Train)?;
    let mut classes: BTreeMap<(String, bool), Vec<usize>> = BTreeMap::new();
    for pair in data.fit.chunks(2) {
        let wanted: Vec<_> = pair
            .iter()
            .copied()
            .filter(|&i| {
                let p = data.position(i);
                matches!(p.family.as_str(), "KQRvK" | "KRRvK") && matches!(p.mate_depth, 2 | 3)
            })
            .collect();
        if wanted.is_empty() {
            continue;
        }
        let roots = data.roots(&wanted)?;
        let references: Vec<_> = roots.iter().collect();
        let out = base
            .valid()
            .base_root(&RootInputs::<B::InnerBackend>::from_roots(
                &references,
                device,
            )?);
        let w = out.z0.dims()[1];
        let z = out
            .z0
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        for (j, &i) in wanted.iter().enumerate() {
            let p = data.position(i);
            let right = p
                .correct
                .contains(&(choose(&z[j * w..j * w + p.legal.len()]) as u32));
            classes
                .entry((format!("{} M{}", p.family, p.mate_depth), right))
                .or_default()
                .push(i);
        }
    }
    anyhow::ensure!(
        classes.len() == 8 && classes.values().all(|v| v.len() >= 12),
        "insufficient fixed panel support: {:?}",
        classes
            .iter()
            .map(|(k, v)| (k, v.len()))
            .collect::<Vec<_>>()
    );
    let support = classes.iter().fold(
        BTreeMap::<String, [usize; 2]>::new(),
        |mut m, ((c, r), v)| {
            m.entry(c.clone()).or_default()[usize::from(*r)] = v.len();
            m
        },
    );
    let mut entries = Vec::new();
    for ((cell, right), mut indices) in classes {
        indices.sort_by_key(|&i| (selection(&data.position(i).id), data.position(i).id.clone()));
        for i in indices.into_iter().take(12) {
            let mut packets = Vec::new();
            for policy in Policy::all() {
                packets.push(prepare(base, data, &[i], policy, device)?.0.remove(0));
            }
            entries.push(Entry {
                index: i,
                id: data.position(i).id.clone(),
                cell: cell.clone(),
                baseline_correct: right,
                packets,
            });
        }
    }
    B::seed(device, 6300);
    let a = Reader::<B>::new(Arm::SharedBackup, device);
    let ai = crate::qualify::inventory(&a)?;
    B::seed(device, 6300);
    let b = Reader::<B>::new(Arm::OnePass, device);
    let bi = crate::qualify::inventory(&b)?;
    anyhow::ensure!(
        ai.iter()
            .filter(|v| !v.name.starts_with("backup."))
            .cloned()
            .collect::<Vec<_>>()
            == bi,
        "shared-shape initial tensors mismatch"
    );
    let episodes: Vec<_> = (0..200)
        .map(|u| (0..12).map(|k| episode(u, k)).collect::<Vec<_>>())
        .collect();
    let mut p = Plan {
        schema: "v6_competent_base_plan_v2".into(),
        source_sha: crate::SOURCE.into(),
        config_digest: crate::packet::config_digest()?,
        train_digest: recur64_v5::data::TRAIN_DIGEST.into(),
        seed: 6300,
        updates: 200,
        entries,
        strata_support: support,
        initial_shapes_match: true,
        episode_digest: crate::digest(&episodes)?,
        digest: String::new(),
    };
    p.digest = plan_digest(&p)?;
    validate_plan(&p)?;
    for entries in p.entries.chunks(2) {
        for policy in 0..2 {
            let refs = entries.iter().collect::<Vec<_>>();
            let _ = cached_inputs(base, data, &refs, policy, device, Some(&p))?;
        }
    }
    Ok(p)
}
fn cached_inputs<B: AutodiffBackend>(
    base: &crate::baseline::FrozenBase<B>,
    data: &V5Data,
    entries: &[&Entry],
    policy: usize,
    device: &B::Device,
    shuffle: Option<&Plan>,
) -> anyhow::Result<(Inputs<B>, Tensor<B, 2, Bool>)> {
    data.require_role(recur64_v5::native_data_v2::Role::Train)?;
    let indices: Vec<_> = entries.iter().map(|e| e.index).collect();
    for e in entries {
        anyhow::ensure!(
            data.position(e.index).id == e.id
                && data.position(e.index).legal == e.packets[policy].legal,
            "frozen packet/data action alignment"
        );
    }
    let roots = data.roots(&indices)?;
    let references: Vec<_> = roots.iter().collect();
    let out = base
        .valid()
        .base_root(&RootInputs::<B::InnerBackend>::from_roots(
            &references,
            device,
        )?);
    let w = out.z0.dims()[1];
    let mut packets: Vec<_> = entries.iter().map(|e| e.packets[policy].clone()).collect();
    if let Some(plan) = shuffle {
        for (p, e) in packets.iter_mut().zip(entries) {
            for n in &mut p.nodes {
                let turn = n.root_to_move;
                let mut donors: Vec<_> = plan
                    .entries
                    .iter()
                    .filter(|d| d.id != e.id && d.cell == e.cell)
                    .flat_map(|d| {
                        d.packets[policy]
                            .nodes
                            .iter()
                            .filter(move |v| v.root_to_move == turn)
                            .map(move |v| (&d.id, v))
                    })
                    .collect();
                anyhow::ensure!(
                    !donors.is_empty(),
                    "unresolved TRAIN same-turn shuffle donor"
                );
                let distance = donors
                    .iter()
                    .map(|(_, v)| v.depth.abs_diff(n.depth))
                    .min()
                    .unwrap();
                donors.retain(|(_, v)| v.depth.abs_diff(n.depth) == distance);
                donors
                    .sort_by_key(|(id, v)| ((*id).clone(), v.depth, v.path.clone(), v.storage_id));
                let hash = crate::packet::key(&e.id, &n.path, n.depth as u16, 0x7A60_E002);
                let k = u64::from_le_bytes(hash[..8].try_into().unwrap()) as usize % donors.len();
                n.payload = donors[k].1.payload.clone();
            }
            p.digest = p.content_digest()?;
        }
    }
    for (p, root) in packets.iter().zip(&roots) {
        if shuffle.is_none() {
            p.verify_against_root(root)?;
        }
    }
    let i = inputs(
        &packets,
        Tensor::from_inner(out.hypotheses),
        Tensor::from_inner(out.z0),
        device,
    )?;
    Ok((i, targets(data, &indices, w, device)?))
}
#[derive(Serialize, Deserialize, Clone)]
pub struct EndpointRow {
    pub id: String,
    pub policy: usize,
    pub baseline_correct: bool,
    pub real_correct: bool,
    pub shuffle_correct: bool,
    pub policy_loss: f64,
    pub shuffle_loss: f64,
    pub turn_blind_loss: f64,
    pub turn_blind_correct: bool,
    pub auxiliary_loss: f64,
    pub margin: Option<f64>,
    pub correction_range: f64,
    pub eligible_positive: usize,
    pub eligible_negative: usize,
}
pub fn endpoint<B: AutodiffBackend>(
    m: &Reader<B>,
    base: &crate::baseline::FrozenBase<B>,
    data: &V5Data,
    plan: &Plan,
    device: &B::Device,
) -> anyhow::Result<Vec<EndpointRow>> {
    data.require_role(recur64_v5::native_data_v2::Role::Train)?;
    let mut rows = Vec::new();
    let r = if m.arm() == Arm::SharedBackup { 4 } else { 1 };
    for entries in plan.entries.chunks(2) {
        let references: Vec<_> = entries.iter().collect();
        for policy in 0..2 {
            let (i, c) = cached_inputs(base, data, &references, policy, device, None)?;
            let (si, _) = cached_inputs(base, data, &references, policy, device, Some(plan))?;
            let valid = m.valid(); // All endpoint math is graph-free.
            let lift = |i: &Inputs<B>| Inputs::<B::InnerBackend> {
                states: i.states.clone().inner(),
                flags: i.flags.clone().inner(),
                node_mask: i.node_mask.clone().inner(),
                node_structure: i.node_structure.clone().inner(),
                root_structure: i.root_structure.clone().inner(),
                owners: i.owners.clone().inner(),
                baseline_owner: i.baseline_owner.clone().inner(),
                z0: i.z0.clone().inner(),
                legal: i.legal.clone().inner(),
                eligible: i.eligible.clone().inner(),
                pool_attacker: i.pool_attacker.clone().inner(),
                pool_defender: i.pool_defender.clone().inner(),
                local_allow: i.local_allow.clone().inner(),
                local_sign: i.local_sign.clone().inner(),
                local_allow_turn_blind: i.local_allow_turn_blind.clone().inner(),
                terminals: i.terminals.clone().inner(),
                q: i.q,
                width: i.width,
            };
            let vi = lift(&i);
            let vs = lift(&si);
            let o = valid.forward(&vi, r, false, false, &mut |_| {});
            let s = valid.forward(&vs, r, false, false, &mut |_| {});
            let turn_blind = valid.forward(&vi, r, false, true, &mut |_| {});
            let tz = turn_blind
                .logits
                .into_data()
                .to_vec::<f32>()
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let null = valid.forward(&vi, r, true, false, &mut |_| {});
            anyhow::ensure!(
                null.raw_delta
                    .into_data()
                    .to_vec::<f32>()
                    .map_err(|e| anyhow::anyhow!("{e:?}"))?
                    .iter()
                    .all(|v| *v == 0.),
                "endpoint null invariant"
            );
            let z = o
                .logits
                .into_data()
                .to_vec::<f32>()
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let sz = s
                .logits
                .into_data()
                .to_vec::<f32>()
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let iteration_raw: Vec<Vec<f32>> = o
                .iteration_delta
                .iter()
                .cloned()
                .map(|t| {
                    t.into_data()
                        .to_vec::<f32>()
                        .map_err(|e| anyhow::anyhow!("{e:?}"))
                })
                .collect::<anyhow::Result<_>>()?;
            let raw = o
                .raw_delta
                .into_data()
                .to_vec::<f32>()
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let _ = c;
            for (j, e) in entries.iter().enumerate() {
                let p = data.position(e.index);
                let zz = &z[j * i.width..j * i.width + p.legal.len()];
                let ss = &sz[j * i.width..j * i.width + p.legal.len()];
                let tt = &tz[j * i.width..j * i.width + p.legal.len()];
                let rr = &raw[j * i.width..j * i.width + p.legal.len()];
                let correct: Vec<_> = (0..p.legal.len())
                    .map(|a| p.correct.contains(&(a as u32)))
                    .collect();
                let eligible = e.packets[policy].eligibility();
                let pos = correct
                    .iter()
                    .zip(&eligible)
                    .filter(|(a, b)| **a && **b)
                    .count();
                let neg = correct
                    .iter()
                    .zip(&eligible)
                    .filter(|(a, b)| !**a && **b)
                    .count();
                let mut aux = 0.;
                let mut classes = 0;
                for iteration in &iteration_raw {
                    for target in [true, false] {
                        let vv: Vec<_> = iteration[j * i.width..j * i.width + p.legal.len()]
                            .iter()
                            .zip(&correct)
                            .zip(&eligible)
                            .filter(|((_, c), e)| **e && **c == target)
                            .map(|((v, _), _)| {
                                let x = f64::from(*v);
                                if target {
                                    (-x).max(0.) + (-x.abs()).exp().ln_1p()
                                } else {
                                    x.max(0.) + (-x.abs()).exp().ln_1p()
                                }
                            })
                            .collect();
                        if !vv.is_empty() {
                            aux += vv.iter().sum::<f64>() / vv.len() as f64;
                            classes += 1;
                        }
                    }
                }
                let bc = zz
                    .iter()
                    .zip(&correct)
                    .filter(|(_, c)| **c)
                    .map(|(v, _)| *v as f64)
                    .max_by(f64::total_cmp)
                    .unwrap();
                let bi = zz
                    .iter()
                    .zip(&correct)
                    .filter(|(_, c)| !**c)
                    .map(|(v, _)| *v as f64)
                    .max_by(f64::total_cmp);
                rows.push(EndpointRow {
                    id: e.id.clone(),
                    policy,
                    baseline_correct: e.baseline_correct,
                    real_correct: correct[choose(zz)],
                    shuffle_correct: correct[choose(ss)],
                    turn_blind_correct: correct[choose(tt)],
                    turn_blind_loss: recur64_v5::loss::reference_set_loss(
                        tt,
                        &vec![true; tt.len()],
                        &correct,
                    ),
                    policy_loss: recur64_v5::loss::reference_set_loss(
                        zz,
                        &vec![true; zz.len()],
                        &correct,
                    ),
                    shuffle_loss: recur64_v5::loss::reference_set_loss(
                        ss,
                        &vec![true; ss.len()],
                        &correct,
                    ),
                    auxiliary_loss: if classes > 0 {
                        aux / classes as f64
                    } else {
                        0.
                    },
                    margin: bi.map(|v| bc - v),
                    correction_range: rr.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64
                        - rr.iter().copied().fold(f32::INFINITY, f32::min) as f64,
                    eligible_positive: pos,
                    eligible_negative: neg,
                });
            }
        }
    }
    Ok(rows)
}
pub fn gates(
    initial: &[EndpointRow],
    final_rows: &[EndpointRow],
) -> anyhow::Result<serde_json::Value> {
    anyhow::ensure!(
        initial.len() == 192 && final_rows.len() == 192,
        "endpoint support"
    );
    let initial_ids: std::collections::BTreeSet<_> = initial
        .iter()
        .map(|r| (r.id.clone(), r.policy, r.baseline_correct))
        .collect();
    let final_ids: std::collections::BTreeSet<_> = final_rows
        .iter()
        .map(|r| (r.id.clone(), r.policy, r.baseline_correct))
        .collect();
    anyhow::ensure!(
        initial_ids.len() == 192 && initial_ids == final_ids,
        "endpoint ID/stratum/policy mismatch"
    );
    anyhow::ensure!(
        final_rows
            .iter()
            .filter(|r| r.policy == 0 && !r.baseline_correct)
            .count()
            == 48
            && final_rows
                .iter()
                .filter(|r| r.policy == 0 && r.baseline_correct)
                .count()
                == 48,
        "48 wrong/right roots required"
    );
    let mut by: BTreeMap<&str, Vec<&EndpointRow>> = BTreeMap::new();
    for r in final_rows {
        by.entry(&r.id).or_default().push(r);
    }
    anyhow::ensure!(
        by.len() == 96
            && by
                .values()
                .all(|v| v.len() == 2 && v[0].policy != v[1].policy),
        "paired root counting mismatch"
    );
    let corrected = by
        .values()
        .filter(|v| !v[0].baseline_correct && v.iter().all(|r| r.real_correct))
        .count();
    let harmed = by
        .values()
        .filter(|v| v[0].baseline_correct && v.iter().any(|r| !r.real_correct))
        .count();
    let specificity = by
        .values()
        .filter(|v| {
            !v[0].baseline_correct && v.iter().all(|r| r.real_correct && !r.shuffle_correct)
        })
        .count();
    let first = initial.iter().map(|r| r.policy_loss).sum::<f64>() / 192.;
    let last = final_rows.iter().map(|r| r.policy_loss).sum::<f64>() / 192.;
    let contrast = final_rows
        .iter()
        .map(|r| r.shuffle_loss - r.policy_loss)
        .sum::<f64>()
        / 192.;
    let pass = (first < 0.05 || last <= first * 0.8)
        && corrected >= 12
        && harmed <= 2
        && contrast >= 0.05
        && specificity >= 6;
    Ok(
        serde_json::json!({"unit":"96 paired roots, correction both policies, harm either policy","initial_policy_loss":first,"final_policy_loss":last,"relative_reduction":1.-last/first,"corrected_wrong_roots":corrected,"harmed_right_roots":harmed,"corrected_then_shuffle_wrong_both_policies":specificity,"shuffle_minus_real_loss":contrast,"per_policy":(0..2).map(|p|serde_json::json!({"policy":p,"corrected":final_rows.iter().filter(|r|r.policy==p&&!r.baseline_correct&&r.real_correct).count(),"harmed":final_rows.iter().filter(|r|r.policy==p&&r.baseline_correct&&!r.real_correct).count()})).collect::<Vec<_>>(),"pass":pass,"held_out_result":false}),
    )
}
#[derive(Serialize, Deserialize)]
struct Checkpoint {
    source: String,
    config: String,
    plan: String,
    arm: Arm,
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
    history: Vec<serde_json::Value>,
}
pub struct LearnOptions {
    pub resume: bool,
    pub max_minutes: f64,
}
pub fn learn<B: AutodiffBackend>(
    base: &crate::baseline::FrozenBase<B>,
    data: &V5Data,
    plan: &Plan,
    arm: Arm,
    dir: &Path,
    device: &B::Device,
    options: LearnOptions,
) -> anyhow::Result<serde_json::Value> {
    let LearnOptions {
        resume,
        max_minutes,
    } = options;
    data.require_role(recur64_v5::native_data_v2::Role::Train)?;
    validate_plan(plan)?;
    anyhow::ensure!(
        max_minutes > 0. && max_minutes <= 45.,
        "bounded invocation required"
    );
    let before = base.baseline_parameter_digest()?;
    let (mut m, mut opt, mut update, mut history, wall) = if resume {
        let meta: Checkpoint = serde_json::from_slice(&std::fs::read(dir.join("latest.json"))?)?;
        let at = dir.join(format!("update-{:03}", meta.update));
        anyhow::ensure!(
            meta.source == crate::SOURCE
                && meta.config == crate::packet::config_digest()?
                && meta.plan == plan.digest
                && meta.arm == arm
                && meta.precision == "fp32"
                && meta.backend == "cuda"
                && meta.microbatch == 2
                && meta.effective_batch == 24
                && meta.seed == 6300
                && meta.episode_digest == plan.episode_digest
                && meta.history.len() == meta.update
                && meta.update < 200
                && meta.model_sha == recur64_v5::stage::hash_file(&at.join("model.mpk"))?
                && meta.optimizer_sha == recur64_v5::stage::hash_file(&at.join("optimizer.mpk"))?,
            "resume identity/refusal"
        );
        let (m, o) = crate::qualify::restore(arm, &at, device)?;
        (m, o, meta.update, meta.history, meta.wall_seconds)
    } else {
        anyhow::ensure!(!dir.exists(), "fresh run cannot overwrite");
        std::fs::create_dir_all(dir)?;
        B::seed(device, 6300);
        let m = Reader::<B>::new(arm, device);
        let _ = crate::qualify::inventory(&m)?;
        let o = recur64_model::train::adamw::<B, Reader<B>>();
        crate::qualify::save(&m, &o, &dir.join("update-000"))?;
        let initial = endpoint(&m, base, data, plan, device)?;
        std::fs::write(dir.join("endpoint-000.json"), serde_json::to_vec(&initial)?)?;
        (m, o, 0, vec![], 0.)
    };
    let start = Instant::now();
    while update < 200 && start.elapsed().as_secs_f64() < max_minutes * 60. {
        anyhow::ensure!(
            wall + start.elapsed().as_secs_f64() < 7200.,
            "per arm fitting budget exceeded"
        );
        let mut accumulator = GradientsAccumulator::new();
        let mut policy_total = 0.;
        let mut aux_total = 0.;
        for k in 0..12 {
            let (index, second, policy) = episode(update, k);
            let refs = vec![&plan.entries[index], &plan.entries[second]];
            let (i, c) = cached_inputs(base, data, &refs, policy, device, None)?;
            let out = m.forward(
                &i,
                if arm == Arm::SharedBackup { 4 } else { 1 },
                false,
                false,
                &mut |_| {},
            );
            let (p, a) = crate::loss::components(&out, &i, c);
            policy_total += scalar(p.clone()) / 12.;
            aux_total += scalar(a.clone()) / 12.;
            let loss = (p + a.mul_scalar(0.5)).div_scalar(12.);
            let g = GradientsParams::from_grads(loss.backward(), &m);
            let _ = crate::qualify::gradients(&m, &g)?;
            accumulator.accumulate(&m, g);
        }
        anyhow::ensure!(
            policy_total.is_finite() && aux_total.is_finite(),
            "nonfinite fitting loss"
        );
        let lr = recur64_runtime::learner::lr_at(update as u64, 1e-3, 20, 200);
        m = opt.step(lr, m, accumulator.grads());
        update += 1;
        history.push(serde_json::json!({"update":update,"policy_loss":policy_total,"auxiliary_loss":aux_total,"lr":lr,"examples":24}));
        if update % 10 == 0 || update == 200 || start.elapsed().as_secs_f64() >= max_minutes * 60. {
            let at = dir.join(format!("update-{update:03}"));
            anyhow::ensure!(!at.exists(), "checkpoint collision");
            crate::qualify::save(&m, &opt, &at)?;
            let meta = Checkpoint {
                source: crate::SOURCE.into(),
                config: crate::packet::config_digest()?,
                plan: plan.digest.clone(),
                arm,
                update,
                model_sha: recur64_v5::stage::hash_file(&at.join("model.mpk"))?,
                optimizer_sha: recur64_v5::stage::hash_file(&at.join("optimizer.mpk"))?,
                wall_seconds: wall + start.elapsed().as_secs_f64(),
                precision: "fp32".into(),
                backend: "cuda".into(),
                microbatch: 2,
                effective_batch: 24,
                seed: 6300,
                episode_digest: plan.episode_digest.clone(),
                history: history.clone(),
            };
            std::fs::write(at.join("state.json"), serde_json::to_vec(&meta)?)?;
            std::fs::write(dir.join("latest.json"), serde_json::to_vec(&meta)?)?;
            eprintln!("{arm:?} update {update}/200 policy={policy_total}");
        }
    }
    anyhow::ensure!(
        before == base.baseline_parameter_digest()?,
        "baseline mutated during fitting"
    );
    if update == 200 {
        let final_rows = endpoint(&m, base, data, plan, device)?;
        std::fs::write(
            dir.join("endpoint-200.json"),
            serde_json::to_vec(&final_rows)?,
        )?;
        let initial: Vec<EndpointRow> =
            serde_json::from_slice(&std::fs::read(dir.join("endpoint-000.json"))?)?;
        let g = gates(&initial, &final_rows)?;
        Ok(
            serde_json::json!({"schema":"v6_p0_learnability_v1","source_sha":crate::SOURCE,"config_digest":crate::packet::config_digest()?,"arm":arm,"update":200,"plan_digest":plan.digest,"gates":g,"baseline_exact":true,"null_exact":true,"policy_auxiliary_history":history,"wall_seconds":wall+start.elapsed().as_secs_f64(),"disposable":true,"weights_reused":false}),
        )
    } else {
        Ok(serde_json::json!({"bounded_stop":true,"update":update,"resume_required":true}))
    }
}

#[cfg(test)]
mod counting_tests {
    use super::*;
    #[test]
    fn episode_bags_match_and_each_policy_gets_twelve_examples() {
        let mut exposures = [[0; 2]; 96];
        for u in 0..200 {
            let mut counts = [0; 2];
            for k in 0..12 {
                let (i, j, p) = episode(u, k);
                for j in [i, j] {
                    counts[p] += 1;
                    exposures[j][p] += 1;
                }
            }
            assert_eq!(counts, [12, 12]);
        }
        assert!(exposures.iter().all(|v| *v == [25, 25]));
    }
    #[test]
    fn roots_not_policy_rows_define_correction_harm_and_specificity() {
        let mut initial = Vec::new();
        let mut rows = Vec::new();
        for root in 0..96 {
            for policy in 0..2 {
                let baseline = root >= 48;
                let initial_row = EndpointRow {
                    id: root.to_string(),
                    policy,
                    baseline_correct: baseline,
                    real_correct: baseline,
                    shuffle_correct: baseline,
                    policy_loss: 1.,
                    shuffle_loss: 1.,
                    turn_blind_loss: 0.,
                    turn_blind_correct: true,
                    auxiliary_loss: 0.,
                    margin: Some(0.),
                    correction_range: 0.,
                    eligible_positive: 1,
                    eligible_negative: 1,
                };
                initial.push(initial_row.clone());
                let mut row = initial_row;
                row.policy_loss = 0.5;
                row.shuffle_loss = 0.6;
                row.real_correct = if baseline {
                    !(root == 48 && policy == 0)
                } else {
                    root < 12 || (root == 12 && policy == 0)
                };
                row.shuffle_correct = if root < 6 { false } else { row.real_correct };
                rows.push(row);
            }
        }
        let g = gates(&initial, &rows).unwrap();
        assert_eq!(g["corrected_wrong_roots"], 12);
        assert_eq!(g["harmed_right_roots"], 1);
        assert_eq!(g["corrected_then_shuffle_wrong_both_policies"], 6);
        assert_eq!(g["pass"], true);
        rows.pop();
        assert!(gates(&initial, &rows).is_err());
    }
}
