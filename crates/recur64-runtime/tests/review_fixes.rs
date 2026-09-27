//! Core-review fixes (2026-09-27).
//!
//! * D54: in an arena each player must search its own tree with its own
//!   network. The historical per-node router sent every tree node to the
//!   network of that node's side to move, mixing both networks in each search.
//! * Non-finite model output must fail visibly on both the synchronous and
//!   the batched inference paths (no uniform-prior / zero-value fallback).

use std::sync::Mutex;

use burn::backend::Flex;
use burn::prelude::Backend;

use recur64_core::{Color, GameState, encode_observation_v1};
use recur64_eval::{ArenaConfig, ArenaTreePolicy, run_arena};
use recur64_model::config::ModelConfig;
use recur64_model::model::ProbeModel;
use recur64_runtime::{BatchEvaluator, BatchedModel, SyncEvaluator};
use recur64_search::{EvalError, EvalRequest, EvalResult, Evaluator, FixedEvaluator};

/// Counts the side to move of every request it evaluates.
struct Counting {
    inner: FixedEvaluator,
    seen: Mutex<[u64; 2]>,
}

impl Counting {
    fn new() -> Self {
        Self {
            inner: FixedEvaluator::uniform(0.0),
            seen: Mutex::new([0, 0]),
        }
    }
    fn seen(&self) -> [u64; 2] {
        *self.seen.lock().unwrap()
    }
}

impl Evaluator for Counting {
    fn evaluate(&self, request: EvalRequest<'_>) -> Result<EvalResult, EvalError> {
        let i = match request.side_to_move {
            Color::White => 0,
            Color::Black => 1,
        };
        self.seen.lock().unwrap()[i] += 1;
        self.inner.evaluate(request)
    }
}

fn one_game(policy: ArenaTreePolicy) -> ([u64; 2], [u64; 2]) {
    // One game: the candidate plays White (game index 0).
    let cfg = ArenaConfig {
        games: 1,
        simulations: 8,
        ply_cap: 6,
        tree_policy: policy,
        ..ArenaConfig::default()
    };
    let (reference, candidate) = (Counting::new(), Counting::new());
    run_arena(&reference, &candidate, "ref", "cand", &cfg).unwrap();
    (candidate.seen(), reference.seen())
}

#[test]
fn root_player_trees_use_one_network_per_side() {
    // Historical routing: the White candidate's evaluator only ever sees
    // White-to-move nodes, because Black-to-move nodes inside its own tree
    // are sent to the reference.
    let (cand, reference) = one_game(ArenaTreePolicy::PerNodeSideV1);
    assert!(
        cand[0] > 0 && cand[1] == 0,
        "per-node: candidate saw {cand:?}"
    );
    assert!(
        reference[0] == 0 && reference[1] > 0,
        "per-node: ref saw {reference:?}"
    );

    // D54: each side's evaluator sees every node of its own tree, at both
    // parities, and never a node of the opponent's tree.
    let (cand, reference) = one_game(ArenaTreePolicy::RootPlayerV1);
    assert!(
        cand[0] > 0 && cand[1] > 0,
        "root-player: candidate saw {cand:?}"
    );
    assert!(
        reference[0] > 0 && reference[1] > 0,
        "root-player: ref saw {reference:?}"
    );
    assert_eq!(ArenaTreePolicy::default(), ArenaTreePolicy::PerNodeSideV1);
}

fn poisoned_model() -> ProbeModel<Flex> {
    let device = Default::default();
    <Flex as Backend>::seed(&device, 3);
    let cfg = ModelConfig {
        width: 32,
        heads: 4,
        ffn: 64,
        input_blocks: 0,
        core_blocks: 1,
        output_blocks: 0,
        squares: 64,
        in_features: 119,
        policy_dim: 16,
        wdl_classes: 3,
        promo_codes: 5,
        rms_eps: 1e-5,
    };
    ProbeModel::<Flex>::new(cfg, &device).with_core_weight_scalar(f32::NAN)
}

#[test]
fn non_finite_model_output_fails_visibly_on_both_inference_paths() {
    let state = GameState::startpos();
    let observation = encode_observation_v1(&state);
    let legal = state.legal_actions();

    let sync = SyncEvaluator::new(poisoned_model(), 1, Default::default());
    let r = sync.evaluate(EvalRequest {
        observation: &observation,
        legal: &legal,
        side_to_move: Color::White,
    });
    assert!(r.is_err(), "sync path must refuse NaN output, got {r:?}");

    let batched = BatchedModel::new(poisoned_model(), 1, Default::default());
    let r = batched.evaluate_batch(
        std::slice::from_ref(&observation),
        std::slice::from_ref(&legal),
    );
    assert!(r.is_err(), "batched path must refuse NaN output");
}
