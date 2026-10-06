//! Source-bound real-device P0 gates; failed gates never authorize fitting.
use crate::model::{Arm, Inputs, Reader};
use burn::module::{AutodiffModule, Module, ModuleVisitor, Param};
use burn::optim::{AdamW, GradientsParams, Optimizer, adaptor::OptimizerAdaptor};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Record, Recorder};
use burn::tensor::backend::AutodiffBackend;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path, time::Instant};
pub type Opt<B> = OptimizerAdaptor<AdamW, Reader<B>, B>;
#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct TensorReceipt {
    pub name: String,
    pub shape: Vec<usize>,
    pub elements: usize,
    pub digest: String,
    pub gradient_present: Option<bool>,
    pub finite: bool,
    pub l2: Option<f64>,
}
pub fn inventory<B: Backend>(m: &Reader<B>) -> anyhow::Result<Vec<TensorReceipt>> {
    struct V {
        path: Vec<String>,
        rows: Vec<TensorReceipt>,
    }
    impl<B: Backend> ModuleVisitor<B> for V {
        fn enter_module(&mut self, n: &str, _: &str) {
            self.path.push(n.into());
        }
        fn exit_module(&mut self, _: &str, _: &str) {
            self.path.pop();
        }
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<B, D>>) {
            let x = p.val();
            let shape = x.dims().to_vec();
            let values = x.into_data().to_vec::<f32>().expect("FP32 parameter data");
            let mut h = Sha256::new();
            for v in &values {
                h.update(v.to_bits().to_le_bytes());
            }
            self.rows.push(TensorReceipt {
                name: self.path.join("."),
                shape,
                elements: values.len(),
                digest: format!("{:x}", h.finalize()),
                gradient_present: None,
                finite: values.iter().all(|v| v.is_finite()),
                l2: None,
            });
        }
    }
    let mut v = V {
        path: vec![],
        rows: vec![],
    };
    m.visit(&mut v);
    anyhow::ensure!(v.rows.iter().all(|r| r.finite), "nonfinite parameter");
    let names: std::collections::BTreeSet<_> = v.rows.iter().map(|x| x.name.clone()).collect();
    anyhow::ensure!(
        names.len() == v.rows.len()
            && names.contains("correction_out.weight")
            && !names.contains("correction_out.bias")
            && names.contains("slots.weight")
            && names.iter().any(|n| n.starts_with("encoder.blocks.1.")),
        "named parameter preflight missing present / unexpected intentionally absent tensor: {names:?}"
    );
    Ok(v.rows)
}
pub fn gradients<B: AutodiffBackend>(
    m: &Reader<B>,
    g: &GradientsParams,
) -> anyhow::Result<Vec<TensorReceipt>> {
    struct V<'a, B: AutodiffBackend> {
        path: Vec<String>,
        rows: Vec<TensorReceipt>,
        g: &'a GradientsParams,
        marker: std::marker::PhantomData<B>,
    }
    impl<B: AutodiffBackend> ModuleVisitor<B> for V<'_, B> {
        fn enter_module(&mut self, n: &str, _: &str) {
            self.path.push(n.into());
        }
        fn exit_module(&mut self, _: &str, _: &str) {
            self.path.pop();
        }
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<B, D>>) {
            let values = self
                .g
                .get::<B::InnerBackend, D>(p.id)
                .map(|v| v.into_data().to_vec::<f32>().expect("gradient readback"));
            let mut h = Sha256::new();
            h.update([u8::from(values.is_some())]);
            if let Some(v) = &values {
                for a in v {
                    h.update(a.to_bits().to_le_bytes());
                }
            }
            self.rows.push(TensorReceipt {
                name: self.path.join("."),
                shape: p.val().dims().to_vec(),
                elements: p.val().dims().iter().product(),
                digest: format!("{:x}", h.finalize()),
                gradient_present: Some(values.is_some()),
                finite: values
                    .as_ref()
                    .is_none_or(|v| v.iter().all(|x| x.is_finite())),
                l2: values
                    .as_ref()
                    .map(|v| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt()),
            });
        }
    }
    let mut v = V::<B> {
        path: vec![],
        rows: vec![],
        g,
        marker: std::marker::PhantomData,
    };
    m.visit(&mut v);
    anyhow::ensure!(
        v.rows
            .iter()
            .all(|r| r.finite && r.gradient_present == Some(true)),
        "missing/nonfinite reader gradient: {:?}",
        v.rows
            .iter()
            .filter(|r| !r.finite || r.gradient_present != Some(true))
            .collect::<Vec<_>>()
    );
    Ok(v.rows)
}
pub fn parameter_digest<B: Backend>(m: &Reader<B>) -> anyhow::Result<String> {
    crate::digest(&inventory(m)?)
}
pub fn optimizer_digest<B: AutodiffBackend>(o: &Opt<B>) -> anyhow::Result<String> {
    let item = o.to_record().into_item::<FullPrecisionSettings>();
    let sorted: BTreeMap<_, _> = item.into_iter().collect();
    struct W(Sha256);
    impl std::io::Write for W {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.update(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut w = W(Sha256::new());
    serde_json::to_writer(&mut w, &sorted)?;
    Ok(format!("{:x}", w.0.finalize()))
}
pub fn scalar<B: Backend>(t: Tensor<B, 1>) -> f64 {
    t.into_scalar().elem::<f32>() as f64
}
pub fn step<B: AutodiffBackend>(
    m: Reader<B>,
    o: &mut Opt<B>,
    i: &Inputs<B>,
    correct: Tensor<B, 2, burn::tensor::Bool>,
    r: usize,
    lr: f64,
    profile: bool,
) -> anyhow::Result<(Reader<B>, serde_json::Value)> {
    let d = i.states.device();
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
    let out = m.forward(i, r, false, false, &mut phase);
    let logits = out
        .logits
        .clone()
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let raw = out
        .raw_delta
        .clone()
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let (policy, aux) = crate::loss::components(&out, i, correct);
    let pv = scalar(policy.clone());
    let av = scalar(aux.clone());
    let loss = policy.clone() + aux.clone().mul_scalar(0.5);
    let lv = scalar(loss.clone());
    anyhow::ensure!(lv.is_finite(), "nonfinite loss");
    phase("policy_and_eligible_auxiliary");
    let g = GradientsParams::from_grads(loss.backward(), &m);
    phase("backward");
    let gs = gradients(&m, &g)?;
    phase("gradient_readback");
    let m = o.step(lr, m, g);
    phase("AdamW");
    anyhow::ensure!(failure.is_none(), "CUDA fence failure: {failure:?}");
    Ok((
        m,
        serde_json::json!({"logits":crate::digest(&logits)?,"raw_delta":crate::digest(&raw)?,"loss":lv,"policy_loss":pv,"auxiliary_loss":av,"gradient_digest":crate::digest(&gs)?,"gradients":gs,"phases":phases}),
    ))
}
pub fn save<B: AutodiffBackend>(m: &Reader<B>, o: &Opt<B>, dir: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    let rec = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    rec.record(m.clone().into_record(), dir.join("model"))?;
    rec.record(o.to_record(), dir.join("optimizer"))?;
    Ok(())
}
pub fn restore<B: AutodiffBackend>(
    arm: Arm,
    dir: &Path,
    device: &B::Device,
) -> anyhow::Result<(Reader<B>, Opt<B>)> {
    let rec = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let m = Reader::<B>::new(arm, device).load_record(rec.load(dir.join("model"), device)?);
    let o = recur64_model::train::adamw::<B, Reader<B>>()
        .load_record(rec.load(dir.join("optimizer"), device)?);
    Ok((m, o))
}
pub fn memory() -> anyhow::Result<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()?;
    anyhow::ensure!(out.status.success(), "GPU memory sample unavailable");
    Ok(String::from_utf8(out.stdout)?
        .lines()
        .next()
        .ok_or_else(|| anyhow::anyhow!("empty memory"))?
        .trim()
        .parse()?)
}
pub fn run<B: AutodiffBackend>(
    base: &crate::baseline::FrozenBase<B>,
    data: &recur64_v5::data::V5Data,
    device: &B::Device,
    backend: &str,
    dir: &Path,
) -> anyhow::Result<serde_json::Value> {
    data.require_role(recur64_v5::native_data_v2::Role::Train)?;
    anyhow::ensure!(
        !dir.exists(),
        "qualification output directory already exists"
    );
    std::fs::create_dir_all(dir)?;
    let before = base.baseline_parameter_digest()?;
    let mut arms = Vec::new();
    let mut peak = if backend == "cuda" { memory()? } else { 0 };
    for arm in [Arm::SharedBackup, Arm::OnePass] {
        B::seed(device, 6300);
        let original = Reader::<B>::new(arm, device);
        let inv = inventory(&original)?;
        let root_count = base.param_breakdown()[0].1;
        anyhow::ensure!(
            original.num_params() + root_count <= 8_000_000,
            "parameter cap"
        );

        // Explicit full-information path from a real TRAIN depth-five state to R1.
        let root = data.roots(&[0])?.remove(0);
        let chain = crate::packet::diagnostic_chain(root.clone(), crate::SOURCE)?;
        chain.verify_against_root(&root)?;
        let inner = base.valid();
        let refs = vec![&root];
        let b = inner.base_root(
            &recur64_v5::model::RootInputs::<B::InnerBackend>::from_roots(&refs, device)?,
        );
        let build = |p: &crate::packet::Packet| {
            crate::model::inputs(
                std::slice::from_ref(p),
                Tensor::from_inner(b.hypotheses.clone()),
                Tensor::from_inner(b.z0.clone()),
                device,
            )
        };
        let mut di = build(&chain)?;
        di.states = di.states.clone().require_grad();
        let tracked = di.states.clone();
        let deep = original.forward(&di, 1, false, false, &mut |_| {});
        let before_logits = deep
            .raw_delta
            .clone()
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let raw = deep.raw_delta.square().sum().backward();
        let dg = tracked
            .grad(&raw)
            .ok_or_else(|| anyhow::anyhow!("deep payload gradient absent"))?
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        anyhow::ensure!(
            dg[4 * 64 * 119..5 * 64 * 119].iter().any(|v| *v != 0.)
                && dg.iter().all(|v| v.is_finite()),
            "depth5 payload R1 gradient failed"
        );
        struct BaselineAbsent<'a, B: AutodiffBackend> {
            raw: &'a B::Gradients,
            absent: bool,
            marker: std::marker::PhantomData<B>,
        }
        impl<B: AutodiffBackend> ModuleVisitor<B> for BaselineAbsent<'_, B> {
            fn visit_float<const N: usize>(&mut self, p: &Param<Tensor<B, N>>) {
                self.absent &= p.val().grad(self.raw).is_none();
            }
        }
        let mut absent = BaselineAbsent::<B> {
            raw: &raw,
            absent: true,
            marker: std::marker::PhantomData,
        };
        base.visit(&mut absent);
        anyhow::ensure!(absent.absent, "baseline gradient present");
        let mut changed = chain.clone();
        changed.nodes[4].payload.observation[0] += 0.5;
        changed.digest = changed.content_digest()?;
        let after = original
            .forward(&build(&changed)?, 1, false, false, &mut |_| {})
            .raw_delta
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        anyhow::ensure!(
            before_logits[chain.nodes[4].root_candidate].to_bits()
                != after[chain.nodes[4].root_candidate].to_bits(),
            "depth5 payload R1 decision insensitive"
        );
        let mut opt = recur64_model::train::adamw::<B, Reader<B>>();
        let snapshot = dir.join(format!("{arm:?}-initial"));
        save(&original, &opt, &snapshot)?;
        let initial = parameter_digest(&original)?;
        let oo = optimizer_digest(&opt)?;
        let mut checks = Vec::new();
        for policy in crate::packet::Policy::all() {
            let (packets, input, correct, timing) =
                crate::p0::prepare(base, data, &[0, 1], policy, device)?;
            let null = original.forward(&input, 4, true, false, &mut |_| {});
            anyhow::ensure!(
                null.raw_delta
                    .into_data()
                    .to_vec::<f32>()
                    .map_err(|e| anyhow::anyhow!("{e:?}"))?
                    .iter()
                    .all(|v| *v == 0.),
                "null cancellation failed"
            );
            let _ = original
                .clone()
                .forward(&input, 1, false, false, &mut |_| {})
                .logits
                .into_data();
            anyhow::ensure!(
                parameter_digest(&original)? == initial,
                "clone forward contaminated original"
            );
            let mut cloned_opt = opt.clone();
            let _ = step(
                original.clone(),
                &mut cloned_opt,
                &input,
                correct.clone(),
                1,
                1e-3,
                false,
            )?;
            anyhow::ensure!(
                parameter_digest(&original)? == initial && optimizer_digest(&opt)? == oo,
                "clone backward/AdamW contaminated original"
            );
            let rs = if arm == Arm::SharedBackup {
                vec![1, 4]
            } else {
                vec![1]
            };
            for r in rs {
                let (a, mut ao) = restore(arm, &snapshot, device)?;
                anyhow::ensure!(
                    parameter_digest(&a)? == initial && optimizer_digest(&ao)? == oo,
                    "normal initial snapshot mismatch"
                );
                let (a, ar) = step(a, &mut ao, &input, correct.clone(), r, 1e-3, false)?;
                anyhow::ensure!(
                    parameter_digest(&original)? == initial && optimizer_digest(&opt)? == oo,
                    "clone/backward original purity failed"
                );
                let (b, mut bo) = restore(arm, &snapshot, device)?;
                anyhow::ensure!(
                    parameter_digest(&b)? == initial && optimizer_digest(&bo)? == oo,
                    "profile initial snapshot mismatch"
                );
                let (b, br) = step(b, &mut bo, &input, correct.clone(), r, 1e-3, true)?;
                std::fs::write(
                    dir.join(format!("{arm:?}-{policy:?}-R{r}-parity.json")),
                    serde_json::to_vec_pretty(
                        &serde_json::json!({"normal":ar,"profile":br,"normal_parameters":inventory(&a)?,"profile_parameters":inventory(&b)?,"normal_moments":optimizer_digest(&ao)?,"profile_moments":optimizer_digest(&bo)?}),
                    )?,
                )?;
                anyhow::ensure!(
                    ar["logits"] == br["logits"]
                        && ar["raw_delta"] == br["raw_delta"]
                        && ar["loss"] == br["loss"]
                        && ar["gradient_digest"] == br["gradient_digest"]
                        && parameter_digest(&a)? == parameter_digest(&b)?
                        && optimizer_digest(&ao)? == optimizer_digest(&bo)?,
                    "exact normal/profile parity failed"
                );
                checks.push(serde_json::json!({"policy":policy,"r":r,"normal_profile_exact":true,"numeric":br,"prepare_seconds":timing,"packet_digests":packets.iter().map(|p|&p.digest).collect::<Vec<_>>()}));
            }
        }
        let (_, input, correct, _) = crate::p0::prepare(
            base,
            data,
            &[0, 1],
            crate::packet::Policy::BroadRankedHash,
            device,
        )?;
        let mut m = original;
        let r = if arm == Arm::SharedBackup { 4 } else { 1 };
        let mut times = Vec::new();
        for _ in 0..50 {
            let start = Instant::now();
            let (next, _) = step(m, &mut opt, &input, correct.clone(), r, 1e-3, false)?;
            m = next;
            B::sync(device).map_err(|e| anyhow::anyhow!("{e:?}"))?;
            times.push(start.elapsed().as_secs_f64());
            if backend == "cuda" {
                peak = peak.max(memory()?);
                anyhow::ensure!(
                    peak as f64 <= 3.2 * 1024.,
                    "device-wide memory cap exceeded"
                );
            }
        }
        // Purity with populated moments, not just a fresh empty optimizer.
        let warm_model = parameter_digest(&m)?;
        let warm_optim = optimizer_digest(&opt)?;
        let mut disposable = opt.clone();
        let _ = step(
            m.clone(),
            &mut disposable,
            &input,
            correct.clone(),
            r,
            1e-3,
            true,
        )?;
        anyhow::ensure!(
            parameter_digest(&m)? == warm_model && optimizer_digest(&opt)? == warm_optim,
            "populated-moment clone contamination"
        );
        let checkpoint = dir.join(format!("{arm:?}-resident"));
        save(&m, &opt, &checkpoint)?;
        let (loaded, mut lo) = restore(arm, &checkpoint, device)?;
        anyhow::ensure!(
            parameter_digest(&m)? == parameter_digest(&loaded)?
                && optimizer_digest(&opt)? == optimizer_digest(&lo)?,
            "model/moment restore failed"
        );
        let (next, _) = step(m, &mut opt, &input, correct.clone(), r, 1e-3, false)?;
        let (resumed, _) = step(loaded, &mut lo, &input, correct, r, 1e-3, false)?;
        anyhow::ensure!(
            parameter_digest(&next)? == parameter_digest(&resumed)?
                && optimizer_digest(&opt)? == optimizer_digest(&lo)?,
            "exact continuation failed"
        );
        arms.push(serde_json::json!({"arm":arm,"reader_parameters":inv.iter().map(|x|x.elements).sum::<usize>(),"frozen_root_parameters":root_count,"inventory":inv,"intentionally_absent":["correction_out.bias"],"deep_depth5_r1_dependency":true,"baseline_gradients_absent":true,"populated_moment_clone_purity":true,"checks":checks,"resident_updates":50,"resident_seconds":times,"checkpoint_restore_exact":true,"moment_restore_exact":true,"continuation_exact":true}));
    }
    anyhow::ensure!(
        before == base.baseline_parameter_digest()?,
        "frozen baseline mutated"
    );
    Ok(
        serde_json::json!({"schema":"v6_p0_qualification_v1","source_sha":crate::SOURCE,"architecture":crate::ARCHITECTURE,"config_digest":crate::packet::config_digest()?,"backend":backend,"precision":"fp32","microbatch":2,"arms":arms,"device_used_peak_sampled_mib":if backend=="cuda"{Some(peak)}else{None},"baseline_parameters_exact":true,"baseline_optimizer_registered":false,"pass":true}),
    )
}
