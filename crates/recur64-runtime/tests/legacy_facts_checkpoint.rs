//! `legacy_facts_v25`: exact checkpoint round trip, the full cross-architecture
//! refusal matrix (probe_v1, candidate_v25, legacy_facts_v25, every ordered pair, by
//! architecture id and not by tensor shape), and the inference path.

use burn::prelude::*;

use recur64_core::{GameState, candidate_facts, encode_observation_v1};
use recur64_model::candidate::{CandidateInputs, CandidateV25Model};
use recur64_model::checkpoint::{CheckpointMeta, load_training, save_training};
use recur64_model::config::{CandidateConfig, LegacyFactsConfig, ModelConfig};
use recur64_model::legacy_facts::LegacyFactsModel;
use recur64_model::loss::Targets;
use recur64_model::model::ProbeModel;
use recur64_model::train::{CpuTrainBackend, adamw, train_step_any};
use recur64_runtime::evaluator::SyncEvaluator;
use recur64_runtime::inference::{BatchedModel, InferenceConfig, InferenceOwner};
use recur64_runtime::model_io::load_as;
use recur64_search::{EvalError, EvalRequest, Evaluator};

type B = CpuTrainBackend;
type F = burn::backend::Flex;

fn lf_cfg() -> ModelConfig {
    let mut c = ModelConfig::legacy_facts_v25();
    c.width = 32;
    c.heads = 4;
    c.ffn = 64;
    c.core_blocks = 2;
    c.policy_dim = 16;
    c.legacy_facts = Some(LegacyFactsConfig { facts_hidden: 8 });
    c
}

fn cand_cfg() -> ModelConfig {
    let mut c = ModelConfig::candidate_v25(true);
    c.width = 32;
    c.heads = 4;
    c.ffn = 64;
    c.core_blocks = 2;
    c.candidate = Some(CandidateConfig {
        dim: 16,
        heads: 2,
        ffn: 32,
        blocks: 1,
        facts_hidden: 8,
        policy_hidden: 16,
        facts_enabled: true,
    });
    c
}

fn probe_cfg() -> ModelConfig {
    serde_json::from_str(
        r#"{"width":32,"heads":4,"ffn":64,"input_blocks":0,"core_blocks":2,"output_blocks":0,"policy_dim":16}"#,
    )
    .unwrap()
}

fn states() -> Vec<GameState> {
    let mut g = GameState::startpos();
    let mut out = vec![g.clone()];
    for m in ["e2e4", "e7e5", "g1f3", "b8c6"] {
        g.apply_uci(m).unwrap();
        out.push(g.clone());
    }
    out
}

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("recur64-lf-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn bits(model: &LegacyFactsModel<B>, states: &[GameState]) -> Vec<u32> {
    let device = Default::default();
    let inp = CandidateInputs::<B>::from_states(states, &device).unwrap();
    let out = model.forward(inp.board, &inp.cands, inp.facts);
    let r = &out.readouts[0];
    let mut v: Vec<u32> = r
        .policy
        .log_probs
        .clone()
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect();
    v.extend(
        r.wdl_logits
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|x| x.to_bits()),
    );
    v
}

fn train_two_steps(
    model: LegacyFactsModel<B>,
    optim: &mut impl burn::optim::Optimizer<LegacyFactsModel<B>, B>,
    sts: &[GameState],
) -> LegacyFactsModel<B> {
    let device = Default::default();
    let mut model = model;
    for _ in 0..2 {
        let inp = CandidateInputs::<B>::from_states(sts, &device).unwrap();
        let (b, w) = (sts.len(), inp.cands.width);
        let mut t = vec![0.0f32; b * w];
        for i in 0..b {
            t[i * w] = 1.0;
        }
        let targets = Targets {
            policy_target: Tensor::<B, 2>::from_data(
                burn::tensor::TensorData::new(t, [b, w]),
                &device,
            ),
            wdl_target: Tensor::<B, 1, Int>::from_data(
                burn::tensor::TensorData::new(vec![1i32; b], [b]),
                &device,
            ),
        };
        let (m, _) = train_step_any(
            model,
            optim,
            inp.board,
            &inp.cands,
            Some(inp.facts),
            &targets,
            1,
            false,
            1e-3,
        );
        model = m;
    }
    model
}

#[test]
fn lf_checkpoint_round_trips_exactly() {
    let device = Default::default();
    let cfg = lf_cfg();
    let sts = states();
    let mut optim = adamw::<B, LegacyFactsModel<B>>();
    let model = train_two_steps(
        LegacyFactsModel::<B>::new(cfg.clone(), &device),
        &mut optim,
        &sts,
    );
    let dir = tmp("rt");
    let meta = CheckpointMeta::new(cfg.clone(), 1, false, 2, 1e-3, 7, 0, "flex", "fp32");
    assert_eq!(meta.architecture, "legacy_facts_v25");
    assert_eq!(meta.fact_delta_contract, 2);
    save_training(&dir, &model, &optim, &meta).unwrap();
    let (loaded, _o, m2) = load_training(
        &dir,
        LegacyFactsModel::<B>::new(cfg.clone(), &device),
        adamw::<B, LegacyFactsModel<B>>(),
        &device,
    )
    .unwrap();
    assert_eq!(m2.seed, 7);
    assert!(!m2.model_id.is_empty());
    assert_eq!(bits(&model, &sts), bits(&loaded, &sts));
    let via_io = load_as::<B, LegacyFactsModel<B>>(&dir, &cfg, &device).unwrap();
    assert_eq!(bits(&model, &sts), bits(&via_io, &sts));
    let _ = std::fs::remove_dir_all(&dir);
}

/// Save a fresh checkpoint of each architecture, then try every loader on every dir.
#[test]
fn every_ordered_pair_of_architectures_is_refused_explicitly() {
    let device = Default::default();
    let mk = |name: &str, cfg: &ModelConfig| -> std::path::PathBuf {
        let dir = tmp(name);
        let meta = CheckpointMeta::new(cfg.clone(), 1, false, 0, 1e-3, 1, 0, "flex", "fp32");
        match cfg.architecture.id() {
            "probe_v1" => save_training(
                &dir,
                &ProbeModel::<B>::new(cfg.clone(), &device),
                &adamw::<B, ProbeModel<B>>(),
                &meta,
            )
            .unwrap(),
            "candidate_v25" => save_training(
                &dir,
                &CandidateV25Model::<B>::new(cfg.clone(), &device),
                &adamw::<B, CandidateV25Model<B>>(),
                &meta,
            )
            .unwrap(),
            _ => save_training(
                &dir,
                &LegacyFactsModel::<B>::new(cfg.clone(), &device),
                &adamw::<B, LegacyFactsModel<B>>(),
                &meta,
            )
            .unwrap(),
        }
        dir
    };
    let cfgs = [probe_cfg(), cand_cfg(), lf_cfg()];
    let dirs: Vec<_> = cfgs
        .iter()
        .enumerate()
        .map(|(i, c)| mk(&format!("m{i}"), c))
        .collect();
    let load = |loader: usize, dir: &std::path::Path, cfg: &ModelConfig| -> anyhow::Result<()> {
        match loader {
            0 => load_as::<B, ProbeModel<B>>(dir, cfg, &device).map(|_| ()),
            1 => load_as::<B, CandidateV25Model<B>>(dir, cfg, &device).map(|_| ()),
            _ => load_as::<B, LegacyFactsModel<B>>(dir, cfg, &device).map(|_| ()),
        }
    };
    for loader in 0..3 {
        for ckpt in 0..3 {
            if loader == ckpt {
                load(loader, &dirs[ckpt], &cfgs[loader]).expect("the matching loader must load");
                continue;
            }
            // Right config for the loader, wrong checkpoint: refused by architecture id.
            let e = load(loader, &dirs[ckpt], &cfgs[loader])
                .unwrap_err()
                .to_string();
            assert!(
                e.contains("cross-architecture"),
                "loader {loader} ckpt {ckpt}: {e}"
            );
            // Config of the checkpoint's architecture handed to the wrong loader.
            let e = load(loader, &dirs[ckpt], &cfgs[ckpt])
                .unwrap_err()
                .to_string();
            assert!(e.contains("cannot load"), "loader {loader} cfg {ckpt}: {e}");
        }
    }
    for d in dirs {
        let _ = std::fs::remove_dir_all(d);
    }
}

#[test]
fn lf_inference_paths_agree_and_facts_are_required() {
    let device: <F as burn::tensor::backend::BackendTypes>::Device = Default::default();
    let model = LegacyFactsModel::<F>::new(lf_cfg(), &device);
    let sync = SyncEvaluator::new(model.clone(), 1, device);
    let owner = InferenceOwner::spawn(
        BatchedModel::new(model, 1, device),
        InferenceConfig::default(),
    );
    let batched = owner.evaluator();
    assert!(batched.needs_candidate_facts() && sync.needs_candidate_facts());
    for s in states() {
        let obs = encode_observation_v1(&s);
        let legal = s.legal_actions();
        let facts = candidate_facts(&s);
        let req = |f| EvalRequest {
            observation: &obs,
            legal: &legal,
            side_to_move: s.side_to_move(),
            facts: f,
        };
        let a = sync.evaluate(req(Some(&facts))).unwrap();
        let b = batched.evaluate(req(Some(&facts))).unwrap();
        for (x, y) in a.policy.iter().zip(&b.policy) {
            assert!((x - y).abs() < 1e-5);
        }
        assert!(matches!(
            batched.evaluate(req(None)),
            Err(EvalError::Invalid(_))
        ));
        assert!(matches!(
            sync.evaluate(req(None)),
            Err(EvalError::Invalid(_))
        ));
    }
    owner.shutdown();
}

#[test]
fn probe_identity_is_unchanged_by_the_lf_config_field() {
    let legacy =
        r#"{"width":384,"heads":12,"ffn":768,"input_blocks":0,"core_blocks":8,"output_blocks":0}"#;
    let m: ModelConfig = serde_json::from_str(legacy).unwrap();
    let v = serde_json::to_value(&m).unwrap();
    let keys: Vec<&String> = v.as_object().unwrap().keys().collect();
    assert_eq!(
        keys.len(),
        12,
        "exactly the twelve historical keys: {keys:?}"
    );
    assert!(
        !keys
            .iter()
            .any(|k| *k == "legacy_facts" || *k == "architecture")
    );
}
