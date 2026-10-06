//! Root-only immutable adapter; equations copied verbatim from qualified V5.
use crate::encoder::{SquareBlock, attention_softmax, rel_index_data, rows};
use burn::module::{AutodiffModule, Module, ModuleVisitor, Param};
use burn::nn::{Initializer, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, Record};
use burn::tensor::{Bool, Distribution, Int, TensorData, activation};
use recur64_model::model::CandidateTensors;
use recur64_v5::config::V5Config;
use recur64_v5::model::{BaseOutput, RootInputs};
use recur64_v5::{ACTION_GEOMETRY, FACT_FIELDS, IN_FEATURES, MASKED_LOGIT, SQUARES};
use sha2::{Digest, Sha256};
#[derive(Module, Debug)]
struct MaskedBlock<B: Backend> {
    norm1: RmsNorm<B>,
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    o: Linear<B>,
    norm2: RmsNorm<B>,
    f1: Linear<B>,
    f2: Linear<B>,
    heads: usize,
    head_dim: usize,
}

impl<B: Backend> MaskedBlock<B> {
    fn new(cfg: &V5Config, device: &B::Device) -> Self {
        let (d, h) = (cfg.width, cfg.heads);
        Self {
            norm1: RmsNormConfig::new(d).with_epsilon(cfg.rms_eps).init(device),
            q: LinearConfig::new(d, d).with_bias(true).init(device),
            k: LinearConfig::new(d, d).with_bias(true).init(device),
            v: LinearConfig::new(d, d).with_bias(true).init(device),
            o: LinearConfig::new(d, d).with_bias(true).init(device),
            norm2: RmsNormConfig::new(d).with_epsilon(cfg.rms_eps).init(device),
            f1: LinearConfig::new(d, cfg.ffn).with_bias(true).init(device),
            f2: LinearConfig::new(cfg.ffn, d).with_bias(true).init(device),
            heads: h,
            head_dim: d / h,
        }
    }

    fn forward(&self, x: Tensor<B, 3>, valid: Tensor<B, 2, Bool>) -> Tensor<B, 3> {
        let [b, s, d] = x.dims();
        let (h, hd) = (self.heads, self.head_dim);
        let n = self.norm1.forward(x.clone());
        let split = |t: Tensor<B, 3>| t.reshape([b, s, h, hd]).swap_dims(1, 2);
        let q = split(rows(&self.q, n.clone()));
        let k = split(rows(&self.k, n.clone()));
        let v = split(rows(&self.v, n));
        let logits = q
            .matmul(k.swap_dims(2, 3))
            .mul_scalar(1.0 / (hd as f32).sqrt());
        let pad = valid
            .clone()
            .bool_not()
            .unsqueeze_dim::<3>(1)
            .unsqueeze_dim::<4>(1)
            .expand([b, h, s, s]);
        let a = attention_softmax(logits.mask_fill(pad, MASKED_LOGIT), 3)
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([b, s, d]);
        let x = x + rows(&self.o, a);
        let f = rows(
            &self.f2,
            activation::gelu(rows(&self.f1, self.norm2.forward(x.clone()))),
        );
        (x + f).mask_fill(
            valid.bool_not().unsqueeze_dim::<3>(2).expand([b, s, d]),
            0.0,
        )
    }
}

#[derive(Module, Debug)]
struct RootTower<B: Backend> {
    input: Linear<B>,
    square: Param<Tensor<B, 2>>,
    blocks: Vec<SquareBlock<B>>,
    final_norm: RmsNorm<B>,
    candidate_input: Linear<B>,
    candidate_norm: RmsNorm<B>,
    candidate_blocks: Vec<MaskedBlock<B>>,
    baseline_hidden: Linear<B>,
    baseline_out: Linear<B>,
    cfg: V5Config,
}

impl<B: Backend> RootTower<B> {
    fn new(cfg: &V5Config, device: &B::Device) -> Self {
        let d = cfg.width;
        Self {
            input: LinearConfig::new(IN_FEATURES, d)
                .with_bias(true)
                .init(device),
            square: Param::from_tensor(Tensor::random(
                [SQUARES, d],
                Distribution::Normal(0.0, 0.02),
                device,
            )),
            blocks: (0..cfg.root_blocks)
                .map(|_| SquareBlock::new(cfg, device))
                .collect(),
            final_norm: RmsNormConfig::new(d).with_epsilon(cfg.rms_eps).init(device),
            candidate_input: LinearConfig::new(3 * d + ACTION_GEOMETRY + FACT_FIELDS, d)
                .with_bias(true)
                .init(device),
            candidate_norm: RmsNormConfig::new(d).with_epsilon(cfg.rms_eps).init(device),
            candidate_blocks: (0..cfg.candidate_blocks)
                .map(|_| MaskedBlock::new(cfg, device))
                .collect(),
            baseline_hidden: LinearConfig::new(2 * d, cfg.readout_hidden)
                .with_bias(true)
                .init(device),
            baseline_out: LinearConfig::new(cfg.readout_hidden, 1)
                .with_bias(false)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: 0.01,
                })
                .init(device),
            cfg: cfg.clone(),
        }
    }

    fn rel_idx(&self, device: &B::Device) -> Tensor<B, 2, Int> {
        Tensor::from_data(
            TensorData::new(
                rel_index_data(self.cfg.heads),
                [self.cfg.heads, SQUARES * SQUARES],
            ),
            device,
        )
    }

    fn forward(
        &self,
        board: Tensor<B, 3>,
        cands: &CandidateTensors<B>,
        geom: Tensor<B, 3>,
        facts: Tensor<B, 3>,
    ) -> BaseOutput<B> {
        let device = board.device();
        let [b, _, _] = board.dims();
        let w = cands.width;
        let d = self.cfg.width;
        let mut c = rows(&self.input, board)
            + self
                .square
                .val()
                .reshape([1, SQUARES, d])
                .expand([b, SQUARES, d]);
        let rel = self.rel_idx(&device);
        for block in &self.blocks {
            c = block.forward(c, rel.clone());
        }
        c = self.final_norm.forward(c);
        let pooled = c.clone().mean_dim(1).squeeze_dim::<2>(1);
        let gather = |idx: &Tensor<B, 2, Int>| {
            c.clone()
                .gather(1, idx.clone().unsqueeze_dim::<3>(2).expand([b, w, d]))
        };
        let from = gather(&cands.from_idx);
        let to = gather(&cands.to_idx);
        let global = pooled.clone().unsqueeze_dim::<3>(1).expand([b, w, d]);
        let mut h = rows(
            &self.candidate_input,
            Tensor::cat(vec![from, to, global, geom, facts], 2),
        );
        h = self.candidate_norm.forward(h);
        for block in &self.candidate_blocks {
            h = block.forward(h, cands.mask.clone());
        }
        let ctx = pooled.clone().unsqueeze_dim::<3>(1).expand([b, w, d]);
        let hidden = activation::gelu(rows(
            &self.baseline_hidden,
            Tensor::cat(vec![h.clone(), ctx], 2),
        ));
        let z0 = rows(&self.baseline_out, hidden)
            .squeeze_dim::<2>(2)
            .mask_fill(cands.mask.clone().bool_not(), MASKED_LOGIT);
        BaseOutput {
            context: c,
            pooled,
            hypotheses: h,
            z0,
        }
    }
}

#[derive(Module, Debug)]
pub struct FrozenBase<B: Backend> {
    root: RootTower<B>,
}
impl<B: Backend> FrozenBase<B> {
    pub fn base_root(&self, i: &RootInputs<B>) -> BaseOutput<B> {
        self.root.forward(
            i.root.clone(),
            &i.cands,
            i.candidate_geometry.clone(),
            i.facts.clone(),
        )
    }
    pub fn param_breakdown(&self) -> Vec<(&'static str, usize)> {
        vec![("root", self.root.num_params())]
    }
    pub fn baseline_parameter_digest(&self) -> anyhow::Result<String> {
        struct V(Sha256);
        impl<B: Backend> ModuleVisitor<B> for V {
            fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<B, D>>) {
                let x = p.val();
                self.0.update((D as u64).to_le_bytes());
                for d in x.dims() {
                    self.0.update((d as u64).to_le_bytes());
                }
                for v in x.into_data().to_vec::<f32>().expect("FP32 root") {
                    self.0.update(v.to_bits().to_le_bytes());
                }
            }
        }
        let mut v = V(Sha256::new());
        self.root.visit(&mut v);
        Ok(format!("{:x}", v.0.finalize()))
    }
}
impl<B: burn::tensor::backend::AutodiffBackend> FrozenBase<B> {
    pub fn from_legacy(
        m: recur64_v5::model::CounterfactualRelationalLoop<B>,
        device: &B::Device,
    ) -> anyhow::Result<Self> {
        let expected = m.baseline_parameter_digest()?;
        let fingerprint = recur64_v5::stage::baseline_fingerprint(&m, device)?;
        let item = serde_json::to_value(m.into_record().into_item::<FullPrecisionSettings>())?;
        let root_item = item
            .get("root")
            .ok_or_else(|| anyhow::anyhow!("root record absent"))?
            .clone();
        let record = RootTowerRecord::<B>::from_item::<FullPrecisionSettings>(
            serde_json::from_value(root_item)?,
            device,
        );
        let root = RootTower::new(&V5Config::default(), device).load_record(record);
        let base = Self { root };
        anyhow::ensure!(
            base.baseline_parameter_digest()? == expected
                && base.fingerprint(device)? == fingerprint,
            "root-only transfer parameters/forward differs from predecessor"
        );
        Ok(base)
    }
    pub fn fingerprint(&self, device: &B::Device) -> anyhow::Result<String> {
        let roots = [
            recur64_core::GameState::from_fen("6k1/8/8/8/8/8/4Q3/3RK3 w - - 0 1")?,
            recur64_core::GameState::from_fen("3rk3/4q3/8/8/8/8/8/6K1 b - - 0 1")?,
            recur64_core::GameState::from_fen("4k3/P7/8/8/8/8/7r/4K3 w - - 0 1")?,
        ];
        let refs = roots.iter().collect::<Vec<_>>();
        let out = self
            .valid()
            .base_root(&RootInputs::<B::InnerBackend>::from_roots(&refs, device)?);
        let mut h = Sha256::new();
        h.update(b"recur64.v5.baseline_fingerprint.v1\0");
        for tensor in [
            out.context.into_data(),
            out.hypotheses.into_data(),
            out.z0.into_data(),
        ] {
            for v in tensor
                .to_vec::<f32>()
                .map_err(|e| anyhow::anyhow!("{e:?}"))?
            {
                h.update(v.to_bits().to_le_bytes());
            }
        }
        Ok(format!("{:x}", h.finalize()))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn root_transfer_is_exact_and_discards_unused_legacy_reader() {
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                let _seed_guard = crate::SEEDED_TEST_LOCK.lock().unwrap();
                type B = recur64_model::train::CpuTrainBackend;
                let d = Default::default();
                let m = recur64_v5::model::CounterfactualRelationalLoop::<B>::new(
                    V5Config::default(),
                    &d,
                );
                let count = m.param_breakdown()[0].1;
                let b = FrozenBase::from_legacy(m, &d).unwrap();
                assert_eq!(b.num_params(), count);
                assert_eq!(count, 3_677_728);
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
