//! Recur64 Phase 0 model-shaped probe graph.
//!
//! Square-token bidirectional transformer with pre-RMSNorm, GeLU FFNs,
//! residual paths, a learned relative-displacement bias, a sparse
//! legal-candidate policy path, a promotion head, and a pooled WDL head.
//!
//! This is a systems probe, not a chess model.

use burn::module::{Module, ModuleVisitor, Param};
use burn::nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::{Distribution, Int, TensorData, activation};

use crate::config::ModelConfig;

/// Number of relative displacement buckets for an 8x8 board: `(2*7+1)^2 = 225`.
const REL_BUCKETS: usize = 225;

/// Row-major relative-displacement bucket index for each `(from, to)` square
/// pair, repeated for every attention head. Shape `[heads, squares*squares]`.
fn rel_index_data(heads: usize, squares: usize) -> Vec<i32> {
    assert_eq!(squares, 64, "Phase 0 probe assumes an 8x8 board");
    let mut v = Vec::with_capacity(heads * squares * squares);
    for _h in 0..heads {
        for i in 0..squares {
            let (ri, fi) = (i / 8, i % 8);
            for j in 0..squares {
                let (rj, fj) = (j / 8, j % 8);
                let dr = rj as i32 - ri as i32;
                let df = fj as i32 - fi as i32;
                v.push((dr + 7) * 15 + (df + 7));
            }
        }
    }
    v
}

/// Learned relative-displacement bias shared over the 64x64 attention grid.
#[derive(Module, Debug)]
pub struct RelPosBias<B: Backend> {
    /// `[heads, REL_BUCKETS]`
    pub table: Param<Tensor<B, 2>>,
}

impl<B: Backend> RelPosBias<B> {
    pub fn new(heads: usize, device: &B::Device) -> Self {
        let table = Param::from_tensor(Tensor::random(
            [heads, REL_BUCKETS],
            Distribution::Normal(0.0, 0.02),
            device,
        ));
        Self { table }
    }

    /// Produce `[1, heads, squares, squares]` by gathering the displacement
    /// bucket for every `(from, to)` pair.
    pub fn bias(&self, rel_idx: Tensor<B, 2, Int>, squares: usize, heads: usize) -> Tensor<B, 4> {
        let g = self.table.val().gather(1, rel_idx); // [heads, squares*squares]
        g.reshape([heads, squares, squares]).unsqueeze_dim::<4>(0)
    }
}

/// One bidirectional transformer block: pre-RMSNorm attention + GeLU FFN.
#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    norm1: RmsNorm<B>,
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    out_proj: Linear<B>,
    rel: RelPosBias<B>,
    norm2: RmsNorm<B>,
    ffn1: Linear<B>,
    ffn2: Linear<B>,
    heads: usize,
    squares: usize,
    head_dim: usize,
}

impl<B: Backend> Block<B> {
    pub fn new(cfg: &ModelConfig, device: &B::Device) -> Self {
        let d = cfg.width;
        let h = cfg.heads;
        let hd = cfg.head_dim();
        let eps = cfg.rms_eps;
        Self {
            norm1: RmsNormConfig::new(d).with_epsilon(eps).init(device),
            q_proj: LinearConfig::new(d, d).with_bias(true).init(device),
            k_proj: LinearConfig::new(d, d).with_bias(true).init(device),
            v_proj: LinearConfig::new(d, d).with_bias(true).init(device),
            out_proj: LinearConfig::new(d, d).with_bias(true).init(device),
            rel: RelPosBias::<B>::new(h, device),
            norm2: RmsNormConfig::new(d).with_epsilon(eps).init(device),
            ffn1: LinearConfig::new(d, cfg.ffn).with_bias(true).init(device),
            ffn2: LinearConfig::new(cfg.ffn, d).with_bias(true).init(device),
            heads: h,
            squares: cfg.squares,
            head_dim: hd,
        }
    }

    fn attention(&self, x: Tensor<B, 3>, rel_idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [b, s, d] = x.dims();
        let h = self.heads;
        let hd = self.head_dim;

        let q = self
            .q_proj
            .forward(x.clone())
            .reshape([b, s, h, hd])
            .swap_dims(1, 2); // [b, h, s, hd]
        let k = self
            .k_proj
            .forward(x.clone())
            .reshape([b, s, h, hd])
            .swap_dims(1, 2);
        let v = self
            .v_proj
            .forward(x)
            .reshape([b, s, h, hd])
            .swap_dims(1, 2);

        let scale = 1.0 / (hd as f32).sqrt();
        let logits = q.matmul(k.swap_dims(2, 3)).mul_scalar(scale);
        let bias = self.rel.bias(rel_idx, self.squares, h);
        let logits = logits + bias; // broadcast [1,h,s,s]
        let attn = activation::softmax(logits, 3);
        let out = attn.matmul(v); // [b,h,s,hd]
        let out = out.swap_dims(1, 2).reshape([b, s, d]);
        self.out_proj.forward(out)
    }

    fn ffn(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = self.ffn1.forward(x);
        self.ffn2.forward(activation::gelu(h))
    }

    pub fn forward(&self, x: Tensor<B, 3>, rel_idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let a = x.clone() + self.attention(self.norm1.forward(x), rel_idx);
        a.clone() + self.ffn(self.norm2.forward(a))
    }
}

/// Candidate-index tensors for the sparse policy path.
pub struct CandidateTensors<B: Backend> {
    pub base_idx: Tensor<B, 2, Int>,
    pub from_idx: Tensor<B, 2, Int>,
    pub to_idx: Tensor<B, 2, Int>,
    pub promo_idx: Tensor<B, 2, Int>,
    pub mask: Tensor<B, 2, Bool>,
    pub valid: Tensor<B, 1, Bool>,
    pub width: usize,
}

impl<B: Backend> CandidateTensors<B> {
    pub fn from_batch(cb: &crate::action::CandidateBatch, device: &B::Device) -> Self {
        let b = cb.batch;
        let w = cb.width;
        let base_idx =
            Tensor::<B, 2, Int>::from_data(TensorData::new(cb.base_index(), [b, w]), device);
        let from_idx =
            Tensor::<B, 2, Int>::from_data(TensorData::new(cb.from.clone(), [b, w]), device);
        let to_idx = Tensor::<B, 2, Int>::from_data(TensorData::new(cb.to.clone(), [b, w]), device);
        // Promotion column: promo code 1..4 maps to column 0..3; "none" maps to 0
        // and is neutralised by the `is_promo` factor.
        let promo_col: Vec<i32> = cb
            .promo
            .iter()
            .map(|&p| if p == 0 { 0 } else { p as i32 - 1 })
            .collect();
        let promo_idx = Tensor::<B, 2, Int>::from_data(TensorData::new(promo_col, [b, w]), device);
        let mask =
            Tensor::<B, 2, Bool>::from_data(TensorData::new(cb.mask.clone(), [b, w]), device);
        let valid = Tensor::<B, 1, Bool>::from_data(
            TensorData::new(cb.terminal.iter().map(|t| !t).collect::<Vec<_>>(), [b]),
            device,
        );
        Self {
            base_idx,
            from_idx,
            to_idx,
            promo_idx,
            mask,
            valid,
            width: w,
        }
    }
}

/// Sparse legal-candidate policy output.
pub struct PolicyOutput<B: Backend> {
    /// Log-probabilities over candidates; padded entries are exactly 0.
    pub log_probs: Tensor<B, 2>,
    /// Valid-candidate mask.
    pub mask: Tensor<B, 2, Bool>,
    /// Positions with at least one legal candidate.
    pub valid: Tensor<B, 1, Bool>,
    /// Full `[b, 64, 64]` base score grid (for tests / inspection).
    pub base_all: Tensor<B, 3>,
}

/// One supervised readout.
pub struct Readout<B: Backend> {
    pub policy: PolicyOutput<B>,
    pub wdl_logits: Tensor<B, 2>,
}

/// Result of a probe forward pass.
pub struct ModelOutput<B: Backend> {
    pub readouts: Vec<Readout<B>>,
    pub executed_blocks: usize,
}

/// The Recur64 Phase 0 probe model.
#[derive(Module, Debug)]
pub struct ProbeModel<B: Backend> {
    input_proj: Linear<B>,
    square_emb: Param<Tensor<B, 2>>,
    input_blocks: Vec<Block<B>>,
    core_blocks: Vec<Block<B>>,
    output_blocks: Vec<Block<B>>,
    inject_norm: RmsNorm<B>,
    alpha_logit: Param<Tensor<B, 1>>,
    source_proj: Linear<B>,
    dest_proj: Linear<B>,
    promo1: Linear<B>,
    promo2: Linear<B>,
    wdl: Linear<B>,
    cfg: ModelConfig,
}

impl<B: Backend> ProbeModel<B> {
    pub fn new(cfg: ModelConfig, device: &B::Device) -> Self {
        let d = cfg.width;
        let mk_blocks = |n: usize, dev: &B::Device| -> Vec<Block<B>> {
            (0..n).map(|_| Block::new(&cfg, dev)).collect()
        };
        let alpha_logit = Param::from_tensor(Tensor::from_data(
            TensorData::new(vec![(0.1f32 / 0.9).ln()], [1]),
            device,
        ));
        let square_emb = Param::from_tensor(Tensor::random(
            [cfg.squares, d],
            Distribution::Normal(0.0, 0.02),
            device,
        ));
        let model = Self {
            input_proj: LinearConfig::new(cfg.in_features, d)
                .with_bias(true)
                .init(device),
            square_emb,
            input_blocks: mk_blocks(cfg.input_blocks, device),
            core_blocks: mk_blocks(cfg.core_blocks, device),
            output_blocks: mk_blocks(cfg.output_blocks, device),
            inject_norm: RmsNormConfig::new(d).with_epsilon(cfg.rms_eps).init(device),
            alpha_logit,
            source_proj: LinearConfig::new(d, cfg.policy_dim)
                .with_bias(true)
                .init(device),
            dest_proj: LinearConfig::new(d, cfg.policy_dim)
                .with_bias(true)
                .init(device),
            promo1: LinearConfig::new(d * 3, cfg.policy_dim)
                .with_bias(true)
                .init(device),
            promo2: LinearConfig::new(cfg.policy_dim, cfg.promo_codes - 1)
                .with_bias(true)
                .init(device),
            wdl: LinearConfig::new(d, cfg.wdl_classes)
                .with_bias(true)
                .init(device),
            cfg,
        };
        // Burn 0.21 initializes parameters lazily. A clone of a module whose
        // params are still deferred copies the initializer and re-samples on
        // first access, which breaks value-preserving clones (and therefore
        // checkpoint/resume proofs). Force eager initialization.
        model.force_init();
        model
    }

    /// Force every parameter to materialize its value.
    pub fn force_init(&self) {
        struct Init;
        impl<B: Backend> ModuleVisitor<B> for Init {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                let _ = param.val();
            }
            fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<B, D, Int>>) {
                let _ = param.val();
            }
            fn visit_bool<const D: usize>(&mut self, param: &Param<Tensor<B, D, Bool>>) {
                let _ = param.val();
            }
        }
        let mut visitor = Init;
        self.visit(&mut visitor);
    }

    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    /// Learned input-injection scale `alpha = sigmoid(a)`, as a tensor so that
    /// gradients flow into it.
    fn alpha(&self) -> Tensor<B, 3> {
        activation::sigmoid(self.alpha_logit.val()).reshape([1, 1, 1])
    }

    fn rel_idx(&self, device: &B::Device) -> Tensor<B, 2, Int> {
        let data = rel_index_data(self.cfg.heads, self.cfg.squares);
        Tensor::<B, 2, Int>::from_data(
            TensorData::new(data, [self.cfg.heads, self.cfg.squares * self.cfg.squares]),
            device,
        )
    }

    fn embed(&self, board: Tensor<B, 3>) -> Tensor<B, 3> {
        let p = self.input_proj.forward(board); // [b, s, d]
        let [b, s, d] = p.dims();
        let emb = self.square_emb.val().reshape([1, s, d]).expand([b, s, d]);
        p + emb
    }

    fn run_blocks(
        &self,
        blocks: &[Block<B>],
        x: Tensor<B, 3>,
        rel_idx: &Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let mut h = x;
        for block in blocks {
            h = block.forward(h, rel_idx.clone());
        }
        h
    }

    fn readout(&self, y: Tensor<B, 3>, cands: &CandidateTensors<B>) -> Readout<B> {
        assert!(
            cands.width > 0,
            "candidate batch contains no legal candidates; terminal-only batches \
             have no policy path and must bypass neural evaluation"
        );
        let [b, _s, d] = y.dims();

        // Pooled WDL.
        let pooled = y.clone().mean_dim(1).squeeze_dim::<2>(1); // [b, d]
        let wdl_logits = self.wdl.forward(pooled.clone());

        // Base 64x64 source/destination score grid.
        let source = self.source_proj.forward(y.clone()); // [b, s, pd]
        let dest = self.dest_proj.forward(y.clone());
        let base_all = source.matmul(dest.swap_dims(1, 2)); // [b, s, s]
        let pd = base_all.dims()[2];
        let base_flat = base_all.clone().reshape([b, pd * pd]); // [b, 4096]
        let base = base_flat.gather(1, cands.base_idx.clone()); // [b, width]

        // Promotion deltas: gather h_from, h_to and concat with pooled board.
        let from3 = cands
            .from_idx
            .clone()
            .unsqueeze_dim::<3>(2)
            .expand([b, cands.width, d]);
        let to3 = cands
            .to_idx
            .clone()
            .unsqueeze_dim::<3>(2)
            .expand([b, cands.width, d]);
        let h_from = y.clone().gather(1, from3); // [b, width, d]
        let h_to = y.clone().gather(1, to3);
        let pooled3 = pooled.unsqueeze_dim::<3>(1).expand([b, cands.width, d]);
        let joined = Tensor::cat(vec![h_from, h_to, pooled3], 2); // [b, width, 3d]
        let promo_h = activation::gelu(self.promo1.forward(joined));
        let promo_delta = self.promo2.forward(promo_h); // [b, width, promo_codes-1]

        let sel = promo_delta
            .gather(2, cands.promo_idx.clone().unsqueeze_dim::<3>(2))
            .squeeze_dim::<2>(2); // [b, width]
        let is_promo = {
            // 1.0 where promo code > 0.
            let p = cands.promo_idx.clone(); // [b,width], 0 for none
            p.greater_elem(0).float()
        };
        let logits = base + sel * is_promo;

        // Mask padding with -inf; replace fully-terminal rows with 0 so the
        // softmax is never all-masked.
        let invalid = cands.mask.clone().bool_not(); // [b,width]
        let logits = logits.mask_fill(invalid.clone(), f32::NEG_INFINITY);
        let terminal_row = cands
            .valid
            .clone()
            .bool_not()
            .unsqueeze_dim::<2>(1)
            .expand([b, cands.width]);
        let logits = logits.mask_fill(terminal_row, 0.0);

        let log_probs = activation::log_softmax(logits, 1);
        let log_probs = log_probs.mask_fill(invalid, 0.0);

        Readout {
            policy: PolicyOutput {
                log_probs,
                mask: cands.mask.clone(),
                valid: cands.valid.clone(),
                base_all,
            },
            wdl_logits,
        }
    }

    /// Forward with a given recurrence count.
    ///
    /// * `deep_supervision = false`: one readout from the final hidden state.
    /// * `deep_supervision = true`: one readout per recurrent iteration.
    pub fn forward_r(
        &self,
        board: Tensor<B, 3>,
        cands: &CandidateTensors<B>,
        r: usize,
        deep_supervision: bool,
    ) -> ModelOutput<B> {
        assert!(r >= 1, "recurrence must be >= 1");
        let device = board.device();
        let rel_idx = self.rel_idx(&device);
        let x = self.embed(board);

        let mut h = self.run_blocks(&self.input_blocks, x.clone(), &rel_idx);
        let alpha = self.alpha();
        let inject = x * alpha;

        let mut readouts = Vec::new();
        for _t in 0..r {
            let normed = self.inject_norm.forward(h + inject.clone());
            h = self.run_blocks(&self.core_blocks, normed, &rel_idx);
            if deep_supervision {
                let y = self.run_blocks(&self.output_blocks, h.clone(), &rel_idx);
                readouts.push(self.readout(y, cands));
            }
        }
        if !deep_supervision {
            let y = self.run_blocks(&self.output_blocks, h, &rel_idx);
            readouts.push(self.readout(y, cands));
        }

        let executed = if deep_supervision {
            self.cfg.executed_blocks_deep_supervision(r)
        } else {
            self.cfg.executed_blocks_final(r)
        };
        ModelOutput {
            readouts,
            executed_blocks: executed,
        }
    }

    /// Explicit R=1 straight-line control graph. Must match `forward_r(..,1,false)`.
    pub fn forward_control(
        &self,
        board: Tensor<B, 3>,
        cands: &CandidateTensors<B>,
    ) -> ModelOutput<B> {
        let device = board.device();
        let rel_idx = self.rel_idx(&device);
        let x = self.embed(board);
        let h = self.run_blocks(&self.input_blocks, x.clone(), &rel_idx);
        let inject = x * self.alpha();
        let h = self.inject_norm.forward(h + inject);
        let h = self.run_blocks(&self.core_blocks, h, &rel_idx);
        let y = self.run_blocks(&self.output_blocks, h, &rel_idx);
        let readout = self.readout(y, cands);
        ModelOutput {
            readouts: vec![readout],
            executed_blocks: self.cfg.executed_blocks_final(1),
        }
    }

    /// Total unique parameter count (shared core counted once).
    pub fn num_params(&self) -> usize {
        Module::num_params(self)
    }

    /// Exact parameter count by submodule group. Shared core blocks are counted
    /// once, as they are stored once.
    pub fn param_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("input_proj", self.input_proj.num_params()),
            ("square_emb", self.square_emb.num_params()),
            ("input_blocks", self.input_blocks.num_params()),
            ("core_blocks (shared)", self.core_blocks.num_params()),
            ("output_blocks", self.output_blocks.num_params()),
            ("inject_norm", self.inject_norm.num_params()),
            ("alpha_logit", self.alpha_logit.num_params()),
            ("source_proj", self.source_proj.num_params()),
            ("dest_proj", self.dest_proj.num_params()),
            (
                "promotion_head",
                self.promo1.num_params() + self.promo2.num_params(),
            ),
            ("wdl_head", self.wdl.num_params()),
        ]
    }

    /// Number of uniquely stored core blocks.
    pub fn core_block_count(&self) -> usize {
        self.core_blocks.len()
    }

    /// The `(0,0)` element of the first core block's query projection weight.
    /// Used to prove gradient flow through repeated shared-core execution.
    pub fn core_weight_scalar(&self) -> f32 {
        self.core_blocks[0]
            .q_proj
            .weight
            .val()
            .into_data()
            .to_vec::<f32>()
            .expect("f32 core weight")[0]
    }

    /// Identity of the first core block's query projection weight.
    pub fn core_weight_id(&self) -> burn::module::ParamId {
        self.core_blocks[0].q_proj.weight.id
    }

    /// Identity of the promotion head's first weight (for gradient tests).
    pub fn promo_weight_id(&self) -> burn::module::ParamId {
        self.promo1.weight.id
    }

    /// Clone with the first core block's query weight `(0,0)` replaced.
    pub fn with_core_weight_scalar(&self, v: f32) -> Self {
        let mut m = self.clone();
        let w = m.core_blocks[0].q_proj.weight.val();
        let [o, i] = w.dims();
        let device = w.device();
        let mut data = w.into_data().to_vec::<f32>().expect("f32 core weight");
        data[0] = v;
        let t = Tensor::<B, 2>::from_data(TensorData::new(data, [o, i]), &device);
        m.core_blocks[0].q_proj.weight = Param::from_tensor(t);
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::CandidateBatch;

    type B = burn::backend::Flex;

    fn tiny_cfg() -> ModelConfig {
        ModelConfig {
            width: 32,
            heads: 4,
            ffn: 64,
            input_blocks: 1,
            core_blocks: 2,
            output_blocks: 1,
            squares: 64,
            in_features: 119,
            policy_dim: 16,
            wdl_classes: 3,
            promo_codes: 5,
            rms_eps: 1e-5,
        }
    }

    fn fixture(device: &Device<B>) -> (Tensor<B, 3>, CandidateTensors<B>) {
        let board = Tensor::<B, 3>::random([2, 64, 119], Distribution::Default, device);
        let lists = vec![
            vec![(0, 8, 0u8), (8, 16, 0), (48, 56, 4), (8, 16, 1)],
            vec![(1, 9, 0)],
        ];
        let cb = CandidateBatch::from_lists(&lists);
        let cands = CandidateTensors::from_batch(&cb, device);
        (board, cands)
    }

    #[test]
    fn forward_shapes_and_finiteness() {
        let device = Default::default();
        let cfg = tiny_cfg();
        let model = ProbeModel::<B>::new(cfg.clone(), &device);
        let (board, cands) = fixture(&device);
        let out = model.forward_r(board, &cands, 2, false);
        assert_eq!(out.readouts.len(), 1);
        assert_eq!(out.executed_blocks, cfg.executed_blocks_final(2));
        let vals = out.readouts[0]
            .policy
            .log_probs
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn clone_preserves_weights_flex() {
        let device = Default::default();
        let m = ProbeModel::<B>::new(tiny_cfg(), &device);
        let c = m.clone();
        println!(
            "flex clone: {} vs {}",
            m.core_weight_scalar(),
            c.core_weight_scalar()
        );
        assert_eq!(
            m.core_weight_scalar(),
            c.core_weight_scalar(),
            "Module::clone must preserve parameter values"
        );
    }

    #[test]
    fn flex_forward_is_repeatable() {
        let device = Default::default();
        let model = ProbeModel::<B>::new(tiny_cfg(), &device);
        let (board, cands) = fixture(&device);
        let a = model.forward_r(board.clone(), &cands, 2, false).readouts[0]
            .policy
            .log_probs
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let b = model.forward_r(board, &cands, 2, false).readouts[0]
            .policy
            .log_probs
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let max = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        println!("forward repeatability max abs diff = {max}");
        assert_eq!(a, b, "forward must be exactly repeatable on the same input");
    }

    #[test]
    fn padded_candidates_get_zero_probability() {
        let device = Default::default();
        let model = ProbeModel::<B>::new(tiny_cfg(), &device);
        let (board, cands) = fixture(&device);
        let out = model.forward_r(board, &cands, 1, false);
        let lp = out.readouts[0]
            .policy
            .log_probs
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        // row 1 has width 4, only candidate 0 valid => entries 1..4 are padding.
        let w = cands.width;
        for k in 1..w {
            assert_eq!(lp[w + k], 0.0, "padding must be exactly zero");
        }
    }
}
