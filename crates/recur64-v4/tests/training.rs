//! Plumbing tests for the V4 stage machinery (CPU, tiny geometry, synthetic labels). These test
//! that the training and evaluation functions do what the stage contracts say (which parameter
//! group may move, that labels are deterministic, that replays reproduce runs). The mechanism
//! questions A-G are answered by the TRAIN studies (`recur64 v4 ...`), not here.

use burn::optim::{GradientsParams, Optimizer};
use burn::prelude::*;

use recur64_core::GameState;
use recur64_model::active::coverage::gradient_coverage;
use recur64_model::config::{CandidateConfig, ModelConfig};
use recur64_model::train::{CpuTrainBackend, adamw};
use recur64_runtime::proof::targets::{ProofPosition, ProofTargets, Split};

use recur64_v4::data::V4Data;
use recur64_v4::model::EvidenceBeliefModel;
use recur64_v4::session::{ContentMode, Freeze, RunOptions};
use recur64_v4::train::{
    EvalSel, Sel, UtilityLoss, ce_update, eval_policy, probe_batch, utility_loss, utility_report,
};

type B = burn::backend::Flex;
type TB = CpuTrainBackend;

static RNG: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    RNG.lock().unwrap_or_else(|e| e.into_inner())
}

fn tiny() -> ModelConfig {
    let mut c = ModelConfig::evidence_belief_v4();
    c.width = 32;
    c.heads = 4;
    c.ffn = 64;
    c.core_blocks = 2;
    let e = c.evidence.as_mut().unwrap();
    e.candidate = CandidateConfig {
        dim: 16,
        heads: 2,
        ffn: 32,
        blocks: 1,
        facts_hidden: 8,
        policy_hidden: 16,
        facts_enabled: true,
    };
    e.base_hidden = 16;
    e.query_heads = 2;
    e.query_ffn = 32;
    e.query_blocks = 1;
    e.content_dim = 16;
    e.content_heads = 2;
    e.content_ffn = 32;
    e.content_blocks = 1;
    e.message_dim = 16;
    e.pair_dim = 16;
    e.key_dim = 8;
    e.trust_hidden = 8;
    e.utility_hidden = 16;
    c
}

const FENS: [&str; 6] = [
    "4k3/8/8/8/8/8/3Q4/R3K3 w - - 0 1",
    "8/8/8/4k3/8/8/4K3/1Q5R w - - 0 1",
    "k7/8/1K6/8/8/8/8/1Q5R w - - 0 1",
    "7k/8/5K2/8/8/8/R7/1R6 w - - 0 1",
    "4k3/8/8/8/8/8/R7/R3K3 w - - 0 1",
    "8/3k4/8/8/8/3K4/8/Q6R w - - 0 1",
];

fn data() -> V4Data {
    let positions: Vec<ProofPosition> = FENS
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let s = GameState::from_fen(f).unwrap();
            let legal: Vec<u16> = s.legal_actions().iter().map(|a| a.index() as u16).collect();
            ProofPosition {
                id: format!("syn-{i}"),
                fen: (*f).to_string(),
                split: Split::Train,
                family: "KQRvK".into(),
                mate_depth: 1,
                canon: format!("syn-canon-{i}"),
                chance_top1: 1.0 / legal.len() as f32,
                legal,
                // Synthetic label: the move at legal index i. Plumbing only.
                correct: vec![i as u32],
                generator_seed: 0,
            }
        })
        .collect();
    let t = ProofTargets::new(Split::Train, 1, serde_json::json!({"synthetic": true}), positions);
    V4Data::from_targets(t).unwrap()
}

fn all(d: &V4Data) -> Vec<usize> {
    (0..d.targets.positions.len()).collect()
}

fn v<Bk: Backend, const D: usize>(t: &Tensor<Bk, D>) -> Vec<f32> {
    t.clone().into_data().to_vec::<f32>().unwrap()
}

#[test]
fn stage_a_updates_reduce_the_base_loss_on_a_tiny_set() {
    let _g = lock();
    let device = Default::default();
    let d = data();
    let mut m = EvidenceBeliefModel::<TB>::new(tiny(), &device);
    let mut optim = adamw::<TB, EvidenceBeliefModel<TB>>();
    let idx = all(&d);
    let opts = RunOptions::new(0);
    let mut losses = Vec::new();
    for _ in 0..40 {
        let (g, l) = ce_update(&m, &d, &idx, 3, &opts, Sel::Fixed, &device).unwrap();
        assert!(l.is_finite());
        losses.push(l);
        m = optim.step(2e-3, m, g);
    }
    assert!(
        losses.last().unwrap() < &(losses[0] * 0.7),
        "stage A did not fit a six-position set: {:.3} -> {:.3}",
        losses[0],
        losses.last().unwrap()
    );
}

#[test]
fn stage_b_updates_move_only_the_evidence_path_and_keep_b0_bit_identical() {
    let _g = lock();
    let device = Default::default();
    let d = data();
    let mut m = EvidenceBeliefModel::<TB>::new(tiny(), &device);
    let mut optim = adamw::<TB, EvidenceBeliefModel<TB>>();
    let idx = all(&d);
    let b0 = |m: &EvidenceBeliefModel<TB>| {
        let e = eval_policy(m, &d, &idx, 0, EvalSel::Fixed, 3, &device).unwrap();
        e.ce
    };
    let before = b0(&m);
    let opts = RunOptions::new(4).with_freeze(Freeze::BASE);
    for u in 0..4 {
        let sel = if u % 2 == 0 { Sel::Fixed } else { Sel::Random(u) };
        let (g, l) = ce_update(&m, &d, &idx, 3, &opts, sel, &device).unwrap();
        assert!(l.is_finite());
        m = optim.step(5e-3, m, g);
    }
    assert_eq!(before, b0(&m), "Stage B changed B0");
    let b4 = eval_policy(&m, &d, &idx, 4, EvalSel::Fixed, 3, &device).unwrap();
    assert!(b4.delta_norm.iter().any(|&n| n > 0.0), "no evidence was produced at B4");
}

#[test]
fn replays_reproduce_runs_and_zero_content_replay_equals_b0() {
    let _g = lock();
    let device = Default::default();
    let d = data();
    let m = EvidenceBeliefModel::<B>::new(tiny(), &device);
    let idx = all(&d);
    let run = eval_policy(&m, &d, &idx, 4, EvalSel::Random(9), 3, &device).unwrap();
    let again = eval_policy(&m, &d, &idx, 4, EvalSel::Random(9), 3, &device).unwrap();
    assert_eq!(run.ce, again.ce, "a seeded random schedule is not reproducible");
    let normal = eval_policy(
        &m,
        &d,
        &idx,
        4,
        EvalSel::Replay {
            chosen: &run.chosen,
            content: ContentMode::Normal,
        },
        3,
        &device,
    )
    .unwrap();
    assert_eq!(normal.ce, run.ce, "replaying the recorded path did not reproduce the run");
    let zero = eval_policy(
        &m,
        &d,
        &idx,
        4,
        EvalSel::Replay {
            chosen: &run.chosen,
            content: ContentMode::Zero,
        },
        3,
        &device,
    )
    .unwrap();
    let b0 = eval_policy(&m, &d, &idx, 0, EvalSel::Fixed, 3, &device).unwrap();
    assert_eq!(zero.ce, b0.ce, "zero-content replay is not exactly B0");
    let shuffled = eval_policy(
        &m,
        &d,
        &idx,
        4,
        EvalSel::Replay {
            chosen: &run.chosen,
            content: ContentMode::Shuffled,
        },
        3,
        &device,
    )
    .unwrap();
    assert_ne!(shuffled.ce, run.ce, "shuffled content gave the identical result");
}

#[test]
fn probe_labels_are_deterministic_and_realised_utilities_are_real_numbers() {
    let _g = lock();
    let device = Default::default();
    let d = data();
    let m = EvidenceBeliefModel::<B>::new(tiny(), &device);
    let idx = all(&d);
    let (_, a) = probe_batch(&m, &d, &idx, 2, 5, 7, &device).unwrap();
    let (_, b) = probe_batch(&m, &d, &idx, 2, 5, 7, &device).unwrap();
    assert_eq!(a.len(), idx.len());
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.picks, y.picks);
        assert_eq!(x.u, y.u, "repeating an identical probe changed its label (label noise must be 0)");
        assert!(x.u.iter().all(|u| u.is_finite()));
        assert_eq!(x.picks[0], x.preferred);
        assert_eq!(x.u.len(), x.picks.len());
        assert!(x.picks.len() <= 5);
    }
    let r = utility_report(&a);
    assert_eq!(r.states, 6);
    assert!(r.frac_positive + r.frac_negative + r.frac_zero > 0.999);
    // A different seed probes a different set of edges.
    let (_, c) = probe_batch(&m, &d, &idx, 2, 5, 8, &device).unwrap();
    assert!(a.iter().zip(&c).any(|(x, y)| x.picks != y.picks));
}

#[test]
fn utility_losses_train_only_the_utility_path() {
    let _g = lock();
    let device = Default::default();
    let d = data();
    let m = EvidenceBeliefModel::<TB>::new(tiny(), &device);
    let idx = all(&d);
    for kind in [UtilityLoss::Ranking, UtilityLoss::Regression] {
        let (scores, mut samples) = probe_batch(&m, &d, &idx, 1, 6, 11, &device).unwrap();
        // An untrained tiny model produces utilities below the ranking margin; give the loss real
        // gaps so its plumbing (pairs, gather, gradient scope) is exercised.
        for s in &mut samples {
            for (a, u) in s.u.iter_mut().enumerate() {
                *u = a as f64 * 0.3 - 0.5;
            }
        }
        let loss = utility_loss(scores, &samples, kind);
        assert!(v(&loss)[0].is_finite());
        let grads = GradientsParams::from_grads(loss.backward(), &m);
        let cov = gradient_coverage::<TB, _>(&m, &grads);
        for row in &cov {
            let top = row.name.split('.').next().unwrap_or("");
            match top {
                "base" | "encoder" | "update" => assert!(
                    !row.has_grad,
                    "{kind:?}: frozen parameter {} received a gradient",
                    row.name
                ),
                "utility" => assert!(row.finite, "{kind:?}: {row:?}"),
                _ => {}
            }
        }
        assert!(
            cov.iter().any(|r| r.name.starts_with("utility.") && r.nonzero),
            "{kind:?}: the utility head received no gradient"
        );
    }
}
