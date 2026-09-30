//! `candidate_v25` checkpoint round trip and cross-architecture refusal, using
//! real checkpoint directories (not only in-memory metadata).

use burn::prelude::*;

use recur64_core::{GameState, StandardMove};
use recur64_model::candidate::{CandidateInputs, CandidateV25Model};
use recur64_model::checkpoint::{CheckpointMeta, load_training, save_training};
use recur64_model::config::{CandidateConfig, ModelConfig};
use recur64_model::loss::Targets;
use recur64_model::model::ProbeModel;
use recur64_model::net::NeuralModel;
use recur64_model::train::{CpuTrainBackend, adamw, train_step_any};
use recur64_runtime::model_io::{load, load_as};

type B = CpuTrainBackend;

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
    let mut out = vec![GameState::startpos()];
    let mut g = GameState::startpos();
    for m in ["e2e4", "e7e5", "g1f3", "b8c6"] {
        g.apply_uci(m).unwrap();
        out.push(g.clone());
    }
    let _ = StandardMove::new;
    out
}

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("recur64-v25-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn policy_bits(model: &CandidateV25Model<B>, states: &[GameState]) -> Vec<u32> {
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

#[test]
fn candidate_checkpoint_round_trips_exactly_and_refuses_the_probe_loader() {
    let device = Default::default();
    let cfg = cand_cfg();
    let sts = states();
    let mut model = CandidateV25Model::<B>::new(cfg.clone(), &device);
    let mut optim = adamw::<B, CandidateV25Model<B>>();

    // Two real optimizer steps so the optimizer state is non-trivial.
    for _ in 0..2 {
        let inp = CandidateInputs::<B>::from_states(&sts, &device).unwrap();
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
        let (m, _loss) = train_step_any(
            model,
            &mut optim,
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

    let dir = tmp("cand");
    let meta = CheckpointMeta::new(cfg.clone(), 1, false, 2, 1e-3, 1, 0, "flex", "fp32");
    assert_eq!(meta.architecture, "candidate_v25");
    save_training(&dir, &model, &optim, &meta).unwrap();

    // Exact restore.
    let fresh = CandidateV25Model::<B>::new(cfg.clone(), &device);
    let (loaded, _o, m2) =
        load_training(&dir, fresh, adamw::<B, CandidateV25Model<B>>(), &device).unwrap();
    assert_eq!(m2.architecture, "candidate_v25");
    assert!(!m2.model_id.is_empty());
    assert_eq!(policy_bits(&model, &sts), policy_bits(&loaded, &sts));

    // The weights-only loader path is exact too.
    let via_io = load_as::<B, CandidateV25Model<B>>(&dir, &cfg, &device).unwrap();
    assert_eq!(policy_bits(&model, &sts), policy_bits(&via_io, &sts));

    // V2.5 checkpoint -> Probe loader: refused explicitly, both with a probe
    // config and with the candidate config.
    let e = load(&dir, &probe_cfg(), &device)
        .map(|_: ProbeModel<B>| ())
        .unwrap_err();
    assert!(e.to_string().contains("cross-architecture"), "{e}");
    let e = load(&dir, &cfg, &device)
        .map(|_: ProbeModel<B>| ())
        .unwrap_err();
    assert!(e.to_string().contains("cannot load"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn probe_checkpoint_refuses_the_candidate_loader() {
    let device = Default::default();
    let pcfg = probe_cfg();
    let model = ProbeModel::<B>::new(pcfg.clone(), &device);
    let optim = adamw::<B, ProbeModel<B>>();
    let dir = tmp("probe");
    let meta = CheckpointMeta::new(pcfg.clone(), 1, false, 0, 1e-3, 1, 0, "flex", "fp32");
    assert_eq!(meta.architecture, "probe_v1");
    save_training(&dir, &model, &optim, &meta).unwrap();

    // The Probe loader still works on it.
    load(&dir, &pcfg, &device)
        .map(|_: ProbeModel<B>| ())
        .unwrap();
    // Candidate loader: refused with a candidate config...
    let e = load_as::<B, CandidateV25Model<B>>(&dir, &cand_cfg(), &device)
        .map(|_| ())
        .unwrap_err();
    assert!(e.to_string().contains("cross-architecture"), "{e}");
    // ...and with a probe config (wrong loader for the config).
    let e = load_as::<B, CandidateV25Model<B>>(&dir, &pcfg, &device)
        .map(|_| ())
        .unwrap_err();
    assert!(e.to_string().contains("cannot load"), "{e}");
    // Saving with mismatched metadata is refused too.
    let bad = CheckpointMeta::new(cand_cfg(), 1, false, 0, 1e-3, 1, 0, "flex", "fp32");
    let e = save_training(&tmp("bad"), &model, &optim, &bad).unwrap_err();
    assert!(e.to_string().contains("refusing to save"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
    let _ = <ProbeModel<B> as NeuralModel<B>>::ARCHITECTURE;
}
