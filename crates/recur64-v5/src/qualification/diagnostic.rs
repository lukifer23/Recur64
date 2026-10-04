//! Disposable same-snapshot diagnostic. No qualification or training authority.
use super::*;
use burn::module::{AutodiffModule, ModuleVisitor, Param};
use serde_json::{Value, json};
use std::collections::BTreeMap;

type Tensors = BTreeMap<String, Option<TensorData>>;
fn tensors<B: AutodiffBackend>(
    model: &CounterfactualRelationalLoop<B>,
    grads: Option<&GradientsParams>,
) -> anyhow::Result<(Tensors, BTreeMap<String, String>)> {
    struct Visitor<'a, B: AutodiffBackend> {
        path: Vec<String>,
        rows: Tensors,
        ids: BTreeMap<String, String>,
        grads: Option<&'a GradientsParams>,
        marker: std::marker::PhantomData<B>,
    }
    impl<B: AutodiffBackend> ModuleVisitor<B> for Visitor<'_, B> {
        fn enter_module(&mut self, n: &str, _: &str) {
            self.path.push(n.into());
        }
        fn exit_module(&mut self, _: &str, _: &str) {
            self.path.pop();
        }
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<B, D>>) {
            let name = self.path.join(".");
            self.ids.insert(p.id.to_string(), name.clone());
            let value = match self.grads {
                Some(g) => g.get::<B::InnerBackend, D>(p.id).map(|v| v.into_data()),
                None => Some(p.val().into_data()),
            };
            self.rows.insert(name, value);
        }
    }
    let mut v = Visitor::<B> {
        path: vec![],
        rows: BTreeMap::new(),
        ids: BTreeMap::new(),
        grads,
        marker: std::marker::PhantomData,
    };
    model.visit(&mut v);
    Ok((v.rows, v.ids))
}
fn tensor_digest(data: &Option<TensorData>) -> anyhow::Result<String> {
    let mut h = Sha256::new();
    match data {
        None => h.update(b"absent"),
        Some(d) => {
            h.update(serde_json::to_vec(&d.shape.as_slice())?);
            for v in d.to_vec::<f32>()? {
                h.update(v.to_bits().to_le_bytes());
            }
        }
    }
    Ok(format!("{:x}", h.finalize()))
}
fn diff(a: &Option<TensorData>, b: &Option<TensorData>) -> anyhow::Result<Value> {
    let mut out = json!({"exact":a==b,"shape_a":a.as_ref().map(|v|v.shape.as_slice()),"shape_b":b.as_ref().map(|v|v.shape.as_slice()),"digest_a":tensor_digest(a)?,"digest_b":tensor_digest(b)?,"absent_a":a.is_none(),"absent_b":b.is_none()});
    if let (Some(a), Some(b)) = (a, b) {
        let av = a.to_vec::<f32>()?;
        let bv = b.to_vec::<f32>()?;
        anyhow::ensure!(av.len() == bv.len(), "tensor length mismatch");
        let mut count = 0;
        let mut max_abs = 0.0_f64;
        let mut sum = 0.0_f64;
        let mut max_rel = 0.0_f64;
        let mut ulp = 0_u64;
        let mut first = None;
        let ordered = |v: f32| {
            let x = v.to_bits();
            if x & 0x80000000 != 0 {
                !x
            } else {
                x | 0x80000000
            }
        };
        for (i, (&x, &y)) in av.iter().zip(&bv).enumerate() {
            anyhow::ensure!(x.is_finite() && y.is_finite(), "nonfinite diagnostic value");
            if x.to_bits() != y.to_bits() {
                count += 1;
                if first.is_none() {
                    first = Some(json!({"flat_index":i,"a":x,"b":y}));
                }
            }
            let d = (f64::from(x) - f64::from(y)).abs();
            max_abs = max_abs.max(d);
            sum += d * d;
            max_rel = max_rel.max(d / f64::from(x.abs().max(y.abs())).max(1e-12));
            ulp = ulp.max(u64::from(ordered(x).abs_diff(ordered(y))));
        }
        out["differing_elements"] = json!(count);
        out["max_abs"] = json!(max_abs);
        out["rms"] = json!((sum / av.len().max(1) as f64).sqrt());
        out["max_relative"] = json!(max_rel);
        out["relative_denominator_floor"] = json!(1e-12);
        out["max_ulp"] = json!(ulp);
        out["first_difference"] = json!(first);
    }
    Ok(out)
}
fn compare(a: &Tensors, b: &Tensors) -> anyhow::Result<Value> {
    anyhow::ensure!(a.keys().eq(b.keys()), "named tensor sets differ");
    let mut rows = BTreeMap::new();
    let mut first = None;
    let mut ha = Sha256::new();
    let mut hb = Sha256::new();
    for (n, x) in a {
        let d = diff(x, &b[n])?;
        ha.update(n.as_bytes());
        ha.update(tensor_digest(x)?.as_bytes());
        hb.update(n.as_bytes());
        hb.update(tensor_digest(&b[n])?.as_bytes());
        if first.is_none() && d["exact"] == false {
            first = Some(n.clone());
        }
        rows.insert(n, d);
    }
    Ok(
        json!({"exact":first.is_none(),"first_differing_tensor":first,"aggregate_digest_a":format!("{:x}",ha.finalize()),"aggregate_digest_b":format!("{:x}",hb.finalize()),"tensors":rows}),
    )
}
fn moments<B: AutodiffBackend>(
    opt: &Opt<B>,
    ids: &BTreeMap<String, String>,
) -> anyhow::Result<(Tensors, BTreeMap<String, Value>)> {
    fn walk(
        v: &Value,
        path: &str,
        t: &mut Tensors,
        c: &mut BTreeMap<String, Value>,
    ) -> anyhow::Result<()> {
        if v.get("bytes").is_some() && v.get("dtype").is_some() {
            let d: TensorData = serde_json::from_value(v.clone())?;
            t.insert(path.into(), Some(d));
        } else {
            match v {
                Value::Object(m) => {
                    for (k, v) in m {
                        walk(v, &format!("{path}.{k}"), t, c)?
                    }
                }
                Value::Array(a) => {
                    for (i, v) in a.iter().enumerate() {
                        walk(v, &format!("{path}.{i}"), t, c)?
                    }
                }
                _ => {
                    c.insert(path.into(), v.clone());
                }
            }
        }
        Ok(())
    }
    let value = serde_json::to_value(opt.to_record().into_item::<FullPrecisionSettings>())?;
    let mut t = BTreeMap::new();
    let mut c = BTreeMap::new();
    for (id, v) in value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("optimizer record not map"))?
    {
        let n = ids
            .get(id)
            .ok_or_else(|| anyhow::anyhow!("moment without parameter"))?;
        walk(v, n, &mut t, &mut c)?;
    }
    Ok((t, c))
}
struct Replay {
    baseline: Tensors,
    forward: Tensors,
    gradients: Tensors,
    parameters: Tensors,
    moments: Tensors,
    counters: BTreeMap<String, Value>,
    phases: Vec<String>,
}
fn replay<B: AutodiffBackend>(
    model: CounterfactualRelationalLoop<B>,
    opt: Opt<B>,
    roots: &[recur64_core::GameState],
    graphs: &[AcquiredGraph],
    device: &B::Device,
    fences: Option<&str>,
) -> anyhow::Result<Replay> {
    replay_retained(model, opt, roots, graphs, device, fences, 0)
}

fn replay_retained<B: AutodiffBackend>(
    model: CounterfactualRelationalLoop<B>,
    mut opt: Opt<B>,
    roots: &[recur64_core::GameState],
    graphs: &[AcquiredGraph],
    device: &B::Device,
    fences: Option<&str>,
    retained: u8,
) -> anyhow::Result<Replay> {
    let mut profile = SynchronizedProfile::<B>::new(device)?;
    let mut phases = vec![];
    let mut phase = |n: &str| {
        phases.push(n.to_string());
        if fences == Some("all") || fences == Some(n) {
            profile.mark(n);
        }
    };
    let ex: Vec<_> = roots.iter().zip(graphs).collect();
    let mut input = V5Inputs::<B>::from_examples_profiled(&ex, device, &mut phase)?;
    let tracked = input.states.clone().require_grad();
    input.states = tracked.clone();
    phase("input_gradient_tracking");
    let mut held_frozen_inputs = None;
    let mut root_only_inputs = None;
    let base = if retained & (16 | 32) != 0 {
        let inner = model.valid();
        phase("frozen_model_view");
        let inner_base = if retained & 32 != 0 {
            let root_refs: Vec<_> = roots.iter().collect();
            root_only_inputs = Some(
                crate::model::RootInputs::<B::InnerBackend>::from_roots_profiled(
                    &root_refs, device, &mut phase,
                )?,
            );
            inner.base_root(root_only_inputs.as_ref().expect("root input probe"))
        } else {
            let frozen_inputs =
                V5Inputs::<B::InnerBackend>::from_examples_profiled(&ex, device, &mut phase)?;
            let value = inner.base(&frozen_inputs);
            held_frozen_inputs = Some(frozen_inputs);
            value
        };
        phase("frozen_root_encoder_and_candidate_path");
        let value = crate::model::BaseOutput {
            context: Tensor::from_inner(inner_base.context),
            pooled: Tensor::from_inner(inner_base.pooled),
            hypotheses: Tensor::from_inner(inner_base.hypotheses),
            z0: Tensor::from_inner(inner_base.z0),
        };
        phase("frozen_base_lift");
        value
    } else {
        model.base_frozen_profiled(&ex, device, &mut phase)?
    };
    drop(root_only_inputs);
    // Diagnostic references only: no readback/fence until the replay completes.
    let context = (retained & 1 != 0).then(|| base.context.clone());
    let pooled = (retained & 2 != 0).then(|| base.pooled.clone());
    let hypotheses = (retained & 4 != 0).then(|| base.hypotheses.clone());
    let z0 = (retained & 8 != 0).then(|| base.z0.clone());
    let out = model.paired_profiled(&input, base, 4, Treatment::Normal, None, &mut phase);
    let mut forward = BTreeMap::new();
    forward.insert("logits".into(), Some(out.logits.clone().into_data()));
    forward.insert(
        "centered_delta".into(),
        Some(out.centered_delta.into_data()),
    );
    let loss = correct_set_loss(
        out.logits,
        input.cands.mask.clone(),
        first_legal_correct(&input, device),
    );
    forward.insert("correct_set_loss".into(), Some(loss.clone().into_data()));
    phase("target_loss_and_scalar_readback");
    let raw = loss.backward();
    phase("backward_both_streams");
    let payload = tracked
        .grad(&raw)
        .ok_or_else(|| anyhow::anyhow!("missing payload gradient"))?
        .into_data();
    let grads = GradientsParams::from_grads(raw, &model);
    let (mut gradients, ids) = tensors(&model, Some(&grads))?;
    gradients.insert("returned_payload_input".into(), Some(payload));
    phase("gradient_health_and_input_readback");
    phase("pre_adamw");
    let next = opt.step(3e-4, model, grads);
    B::sync(device).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    phase("adamw_and_completion_fence");
    profile.finish()?;
    let parameters = tensors(&next, None)?.0;
    let (moments, counters) = moments(&opt, &ids)?;
    let mut baseline = BTreeMap::new();
    if let Some(v) = context {
        baseline.insert("context".into(), Some(v.into_data()));
    }
    if let Some(v) = pooled {
        baseline.insert("pooled".into(), Some(v.into_data()));
    }
    if let Some(v) = hypotheses {
        baseline.insert("hypotheses".into(), Some(v.into_data()));
    }
    if let Some(v) = z0 {
        baseline.insert("z0".into(), Some(v.into_data()));
    }
    if let Some(v) = held_frozen_inputs {
        baseline.insert("unused_frozen_states".into(), Some(v.states.into_data()));
        baseline.insert("unused_frozen_flags".into(), Some(v.flags.into_data()));
    }
    Ok(Replay {
        baseline,
        forward,
        gradients,
        parameters,
        moments,
        counters,
        phases,
    })
}
fn pair(a: &Replay, b: &Replay) -> anyhow::Result<Value> {
    Ok(
        json!({"baseline":compare(&a.baseline,&b.baseline)?,"forward":compare(&a.forward,&b.forward)?,"gradients":compare(&a.gradients,&b.gradients)?,"post_adamw_parameters":compare(&a.parameters,&b.parameters)?,"optimizer_moments":compare(&a.moments,&b.moments)?,"optimizer_counters_exact":a.counters==b.counters,"optimizer_counters_a":a.counters,"optimizer_counters_b":b.counters}),
    )
}
fn exact(v: &Value) -> bool {
    [
        "forward",
        "gradients",
        "post_adamw_parameters",
        "optimizer_moments",
    ]
    .iter()
    .all(|k| v[k]["exact"] == true)
        && v["optimizer_counters_exact"] == true
}

pub fn run<B: AutodiffBackend>(
    source: &str,
    label: &str,
    device: &B::Device,
) -> anyhow::Result<Value> {
    B::seed(device, 5301);
    let roots = roots(2);
    let graphs = graphs(&roots, 8)?;
    let model = CounterfactualRelationalLoop::<B>::new(V5Config::default(), device);
    let mut opt = adamw::<B, CounterfactualRelationalLoop<B>>();
    // Populate real moments/counters with the unchanged ordinary update.
    let (model, _, _, _) = update(model, &mut opt, &roots, &graphs, 4, device, false)?;
    let dir = PathBuf::from("runs/v5/profile-diagnostic").join(format!("{source}-{label}"));
    std::fs::create_dir_all(&dir)?;
    anyhow::ensure!(
        !dir.join("model.mpk").exists(),
        "canonical snapshot already exists"
    );
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model.clone().save_file(dir.join("model"), &recorder)?;
    recorder.record(opt.to_record(), dir.join("optimizer"))?;
    let before = (model.parameter_digest()?, optimizer_digest(&opt)?);
    let manifest = json!({"schema":"v5_profile_snapshot_v1","source_sha":source,"architecture":crate::config::ARCHITECTURE,"config_digest":V5Config::default().scientific_digest()?,"model_digest":before.0,"optimizer_digest":before.1,"graphs":graphs,"root_fens":roots.iter().map(|r|r.to_fen()).collect::<Vec<_>>(),"targets":roots.iter().map(|r|json!({"legal_action_ids":r.legal_actions().iter().map(|a|a.index()).collect::<Vec<_>>(),"correct_root_indices":[0],"role":"first-legal execution fixture, not chess-learning label"})).collect::<Vec<_>>(),"r":4,"precision":"fp32"});
    std::fs::write(
        dir.join("snapshot.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    let fresh = || -> anyhow::Result<_> {
        let m = CounterfactualRelationalLoop::<B>::new(V5Config::default(), device).load_file(
            dir.join("model"),
            &recorder,
            device,
        )?;
        let o = adamw::<B, CounterfactualRelationalLoop<B>>()
            .load_record(recorder.load(dir.join("optimizer"), device)?);
        anyhow::ensure!(
            (m.parameter_digest()?, optimizer_digest(&o)?) == before,
            "starting replica differs from canonical snapshot"
        );
        Ok((m, o))
    };
    let fingerprint =
        || -> anyhow::Result<_> { Ok((model.parameter_digest()?, optimizer_digest(&opt)?)) };
    let mut purity = vec![];
    let pre = fingerprint()?;
    let _ = logits(&model.clone(), &roots, &graphs, 4, device)?;
    let post = fingerprint()?;
    purity.push(json!({"operation":"clone_forward","before":pre,"after":post,"original_unchanged":pre==post}));
    for observed in [false, true, true, false] {
        let pre = fingerprint()?;
        let _ = replay(
            model.clone(),
            opt.clone(),
            &roots,
            &graphs,
            device,
            observed.then_some("all"),
        )?;
        let post = fingerprint()?;
        purity.push(json!({"operation":"clone_backward_adamw","profiled":observed,"before":pre,"after":post,"original_unchanged":pre==post}));
    }
    let mut pairs = vec![];
    let mut nn = true;
    let mut pp = true;
    let mut cross = true;
    let mut phases = vec![];
    for rep in 0..3 {
        for (name, left, right) in [
            ("NORMAL_A/NORMAL_B", false, false),
            ("PROFILE_A/PROFILE_B", true, true),
            ("NORMAL_A/PROFILE_A", false, true),
            ("PROFILE_A/NORMAL_A", true, false),
        ] {
            let (m, o) = fresh()?;
            let a = replay(m, o, &roots, &graphs, device, left.then_some("all"))?;
            let (m, o) = fresh()?;
            let b = replay(m, o, &roots, &graphs, device, right.then_some("all"))?;
            let v = pair(&a, &b)?;
            match name {
                "NORMAL_A/NORMAL_B" => nn &= exact(&v),
                "PROFILE_A/PROFILE_B" => pp &= exact(&v),
                _ => cross &= exact(&v),
            }
            phases = a.phases.clone();
            pairs.push(json!({"repeat":rep,"ordering":name,"comparison":v}));
        }
    }
    let mut boundaries = vec![];
    if nn && pp && !cross {
        let (m, o) = fresh()?;
        let normal = replay(m, o, &roots, &graphs, device, None)?;
        let forward_exact = pairs
            .iter()
            .filter(|v| v["ordering"] == "NORMAL_A/PROFILE_A")
            .all(|v| v["comparison"]["forward"]["exact"] == true);
        for n in &phases {
            if forward_exact
                && !matches!(
                    n.as_str(),
                    "target_loss_and_scalar_readback"
                        | "backward_both_streams"
                        | "gradient_health_and_input_readback"
                        | "pre_adamw"
                        | "adamw_and_completion_fence"
                )
            {
                continue;
            }
            let (m, o) = fresh()?;
            let fenced = replay(m, o, &roots, &graphs, device, Some(n))?;
            boundaries.push(json!({"single_fence":n,"comparison":pair(&normal,&fenced)?}));
        }
    }
    let mut lifetime_probes = vec![];
    if nn && pp && !cross {
        for retained in [1, 2, 4, 8, 15, 16, 32, 47] {
            for rep in 0..3 {
                let (m, o) = fresh()?;
                let a = replay_retained(m, o, &roots, &graphs, device, None, retained)?;
                let (m, o) = fresh()?;
                let b = replay_retained(m, o, &roots, &graphs, device, Some("all"), retained)?;
                lifetime_probes
                    .push(json!({"retained_mask":retained,"repeat":rep,"comparison":pair(&a,&b)?}));
            }
        }
    }
    Ok(
        json!({"schema":"v5_profile_parity_diagnostic_v3","training_authorized":false,"source_sha":source,"device":label,"precision":"fp32","config_digest":V5Config::default().scientific_digest()?,"canonical_snapshot":manifest,"clone_purity":purity,"normal_normal_exact":nn,"profile_profile_exact":pp,"cross_mode_exact":cross,"classification":if !nn {"CASE_A"}else if !pp {"CASE_B"}else if !cross {"CASE_C"}else{"ALL_EXACT"},"pairs":pairs,"single_fence_localization":boundaries,"baseline_lifetime_probes":lifetime_probes,"environment":{"CUBLAS_WORKSPACE_CONFIG":std::env::var("CUBLAS_WORKSPACE_CONFIG").ok(),"CUDA_LAUNCH_BLOCKING":std::env::var("CUDA_LAUNCH_BLOCKING").ok(),"CUDA_PATH":std::env::var("CUDA_PATH").ok(),"backend":"Burn 0.21.0 / CubeCL 0.10.0","gpu":std::process::Command::new("nvidia-smi").args(["--query-gpu=name,driver_version","--format=csv,noheader"]).output().ok().map(|v|String::from_utf8_lossy(&v.stdout).into_owned())}}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn numeric_reports_keep_bit_differences_and_absence_visible() {
        let a = Some(TensorData::new(vec![1.0_f32, 0.0], [2]));
        let b = Some(TensorData::new(
            vec![f32::from_bits(1.0_f32.to_bits() + 1), -0.0],
            [2],
        ));
        let d = diff(&a, &b).unwrap();
        assert_eq!(d["exact"], false);
        assert_eq!(d["differing_elements"], 2);
        assert_eq!(d["max_ulp"], 1);
        assert_eq!(d["first_difference"]["flat_index"], 0);
        assert_eq!(diff(&None, &a).unwrap()["exact"], false);
    }
}
