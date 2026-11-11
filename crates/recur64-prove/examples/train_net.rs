//! V500 P3: train the ordering network with burn on exact tablebase distances (TRAIN split only).
//!
//! cargo run --release -p recur64-prove --example train_net -- <out_dir> <seed> <variant> [epochs] [h1] [h2] [materials]
//!   variant: `full` (all depths) or `extrap` (positions deeper than the cap are withheld from training)
//!   materials: `both` (default), `kq` (KQvK only) or `kr` (KRvK only)
//!
//! Writes `<out_dir>/net.json` (deployable weights) and `<out_dir>/train-report.json` (losses and the exact ordering
//! accuracy by depth on the TEST split, computed with the deployed Rust forward pass, not the burn graph).

use std::time::Instant;

use burn::module::{AutodiffModule, Module};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation::relu;
use burn::tensor::{Int, TensorData};
use cozy_chess::{Board, Color, GameStatus, Piece};
use recur64_prove::net::{EXTRA, MAXP, Mlp, NET_SCHEMA, NetFile, TOKENS, extras, target, tokens};
use recur64_prove::split::{Part, part_of};
use recur64_prove::tb::{Tablebase, moves_of};
use serde_json::json;

type Ad = burn::backend::Autodiff<burn::backend::Flex>;

#[derive(Module, Debug)]
struct Net<B: Backend> {
    emb: Embedding<B>,
    extra: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
}

impl<B: Backend> Net<B> {
    fn new(h1: usize, h2: usize, device: &B::Device) -> Self {
        Self {
            emb: EmbeddingConfig::new(TOKENS, h1).init(device),
            extra: LinearConfig::new(EXTRA, h1).with_bias(true).init(device),
            l2: LinearConfig::new(h1, h2).with_bias(true).init(device),
            l3: LinearConfig::new(h2, 1).with_bias(true).init(device),
        }
    }

    fn forward(&self, toks: Tensor<B, 2, Int>, mask: Tensor<B, 2>, ex: Tensor<B, 2>) -> Tensor<B, 1> {
        let e = self.emb.forward(toks) * mask.unsqueeze_dim::<3>(2);
        let s = e.sum_dim(1).squeeze_dim::<2>(1) + self.extra.forward(ex);
        let h = relu(s);
        let z = relu(self.l2.forward(h));
        self.l3.forward(z).squeeze_dim::<1>(1)
    }
}

struct Data {
    toks: Vec<i32>,
    mask: Vec<f32>,
    ex: Vec<f32>,
    y: Vec<f32>,
    n: usize,
}

impl Data {
    fn new() -> Self {
        Self { toks: Vec::new(), mask: Vec::new(), ex: Vec::new(), y: Vec::new(), n: 0 }
    }

    fn push(&mut self, b: &Board, plies: Option<u8>) {
        let (t, n) = tokens(b);
        for i in 0..MAXP {
            self.toks.push(i32::from(t[i]));
            self.mask.push(f32::from(u8::from(i < n)));
        }
        self.ex.extend_from_slice(&extras(b));
        self.y.push(target(plies));
        self.n += 1;
    }

    fn batch<Bk: Backend>(&self, idx: &[usize], device: &Bk::Device) -> (Tensor<Bk, 2, Int>, Tensor<Bk, 2>, Tensor<Bk, 2>, Tensor<Bk, 1>) {
        let n = idx.len();
        let mut t = Vec::with_capacity(n * MAXP);
        let mut m = Vec::with_capacity(n * MAXP);
        let mut e = Vec::with_capacity(n * EXTRA);
        let mut y = Vec::with_capacity(n);
        for &i in idx {
            t.extend_from_slice(&self.toks[i * MAXP..(i + 1) * MAXP]);
            m.extend_from_slice(&self.mask[i * MAXP..(i + 1) * MAXP]);
            e.extend_from_slice(&self.ex[i * EXTRA..(i + 1) * EXTRA]);
            y.push(self.y[i]);
        }
        (
            Tensor::from_data(TensorData::new(t, [n, MAXP]), device),
            Tensor::from_data(TensorData::new(m, [n, MAXP]), device),
            Tensor::from_data(TensorData::new(e, [n, EXTRA]), device),
            Tensor::from_data(TensorData::new(y, [n]), device),
        )
    }
}

struct Sm(u64);
impl Sm {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn vec_of<Bk: Backend, const D: usize>(t: Tensor<Bk, D>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().expect("f32")
}

/// Exact ordering accuracy of the deployed forward pass on TEST white-to-move positions, by optimal depth.
fn ordering_accuracy(tb: &Tablebase, net: &Mlp, max_depth: u8) -> Vec<serde_json::Value> {
    let mut rows = Vec::new();
    let mut by: Vec<(u32, u32, u32, u32)> = vec![(0, 0, 0, 0); max_depth as usize + 1];
    for b in tb.positions.iter().filter(|b| b.side_to_move() == Color::White && b.status() == GameStatus::Ongoing) {
        if part_of(b) != Part::Test {
            continue;
        }
        let Some(p) = tb.plies_to_mate(b) else { continue };
        let d = p.div_ceil(2) as usize;
        let mut scored: Vec<(f32, bool)> = moves_of(b)
            .into_iter()
            .map(|m| {
                let mut a = b.clone();
                a.play_unchecked(m);
                let optimal = a.status() == GameStatus::Won || tb.plies_to_mate(&a) == Some(p - 1);
                let v = match a.status() {
                    GameStatus::Won => -1e9,
                    GameStatus::Drawn => 1e9,
                    GameStatus::Ongoing => net.eval(&a),
                };
                (v, optimal)
            })
            .collect();
        scored.sort_by(|x, y| x.0.total_cmp(&y.0));
        let e = &mut by[d];
        e.0 += 1;
        e.1 += u32::from(scored[0].1);
        e.2 += u32::from(scored.iter().take(3).any(|s| s.1));
        e.3 += scored.iter().filter(|s| s.1).count() as u32 * 1000 / scored.len() as u32;
    }
    for (d, e) in by.iter().enumerate().skip(1) {
        if e.0 > 0 {
            rows.push(json!({"depth": d, "n": e.0, "top1_optimal": f64::from(e.1) / f64::from(e.0),
                             "top3_contains_optimal": f64::from(e.2) / f64::from(e.0),
                             "chance_top1": f64::from(e.3) / 1000.0 / f64::from(e.0)}));
        }
    }
    rows
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 4, "usage: train_net <out_dir> <seed> <full|extrap> [epochs] [h1] [h2] [both|kq|kr]");
    let out = std::path::PathBuf::from(&a[1]);
    let seed: u64 = a[2].parse()?;
    let extrap = match a[3].as_str() {
        "full" => false,
        "extrap" => true,
        other => anyhow::bail!("variant {other}"),
    };
    let epochs: usize = a.get(4).map_or(Ok(20), |s| s.parse())?;
    let h1: usize = a.get(5).map_or(Ok(256), |s| s.parse())?;
    let h2: usize = a.get(6).map_or(Ok(32), |s| s.parse())?;
    let materials = a.get(7).map_or("both", String::as_str).to_string();
    std::fs::create_dir_all(&out)?;

    let kq = Tablebase::build("KQvK", &[Piece::Queen])?;
    let kr = Tablebase::build("KRvK", &[Piece::Rook])?;
    // Extrapolation caps (plies): withhold KRvK deeper than 8 moves and KQvK deeper than 6 moves from training.
    let caps = [("KQvK", 11u8), ("KRvK", 15u8)];
    let (mut train, mut val) = (Data::new(), Data::new());
    for (tb, &(name, cap)) in [&kq, &kr].into_iter().zip(&caps) {
        if (materials == "kq" && name == "KRvK") || (materials == "kr" && name == "KQvK") {
            continue;
        }
        for b in &tb.positions {
            let p = tb.plies_to_mate(b);
            match part_of(b) {
                Part::Train => {
                    if extrap && p.is_some_and(|x| x > cap) {
                        continue;
                    }
                    train.push(b, p);
                }
                Part::Val => val.push(b, p),
                Part::Test => {}
            }
        }
    }
    eprintln!("train {} val {} (variant {}, materials {materials})", train.n, val.n, a[3]);

    let device = Default::default();
    <Ad as burn::tensor::backend::Backend>::seed(&device, seed);
    let mut model = Net::<Ad>::new(h1, h2, &device);
    let mut optim = AdamWConfig::new().with_weight_decay(1e-4).init::<Ad, Net<Ad>>();
    let batch = 4096usize;
    let steps_total = (train.n / batch).max(1) * epochs;
    let mut rng = Sm(seed ^ 0xA5A5_5A5A);
    let mut order: Vec<usize> = (0..train.n).collect();
    let mut step = 0usize;
    let t0 = Instant::now();
    let mut log = Vec::new();
    for epoch in 0..epochs {
        for i in (1..order.len()).rev() {
            order.swap(i, (rng.next() % (i as u64 + 1)) as usize);
        }
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for chunk in order.chunks(batch).filter(|c| c.len() == batch) {
            let lr = 2e-3 * (1.0 - step as f64 / steps_total as f64).max(0.02);
            let (t, m, e, y) = train.batch::<Ad>(chunk, &device);
            let pred = model.forward(t, m, e);
            let loss = (pred - y).powf_scalar(2.0).mean();
            sum += f64::from(vec_of(loss.clone())[0]);
            count += 1;
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(lr, model, grads);
            step += 1;
        }
        // Validation MSE (inner backend, no graph).
        let inner = model.valid();
        let mut vs = 0.0f64;
        let mut vn = 0usize;
        let vidx: Vec<usize> = (0..val.n).collect();
        for chunk in vidx.chunks(8192) {
            let (t, m, e, y) = {
                let dv = Default::default();
                let n = chunk.len();
                let mut tt = Vec::new();
                let mut mm = Vec::new();
                let mut ee = Vec::new();
                let mut yy = Vec::new();
                for &i in chunk {
                    tt.extend_from_slice(&val.toks[i * MAXP..(i + 1) * MAXP]);
                    mm.extend_from_slice(&val.mask[i * MAXP..(i + 1) * MAXP]);
                    ee.extend_from_slice(&val.ex[i * EXTRA..(i + 1) * EXTRA]);
                    yy.push(val.y[i]);
                }
                type I = <Ad as burn::tensor::backend::AutodiffBackend>::InnerBackend;
                (
                    Tensor::<I, 2, Int>::from_data(TensorData::new(tt, [n, MAXP]), &dv),
                    Tensor::<I, 2>::from_data(TensorData::new(mm, [n, MAXP]), &dv),
                    Tensor::<I, 2>::from_data(TensorData::new(ee, [n, EXTRA]), &dv),
                    Tensor::<I, 1>::from_data(TensorData::new(yy, [n]), &dv),
                )
            };
            let p = inner.forward(t, m, e);
            let d = vec_of(p - y);
            vs += d.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>();
            vn += d.len();
        }
        let rec = json!({"epoch": epoch, "train_mse": sum / count.max(1) as f64, "val_mse": vs / vn.max(1) as f64, "secs": t0.elapsed().as_secs_f64()});
        eprintln!("{rec}");
        log.push(rec);
    }

    // Export to the deployable format.
    let inner = model.valid();
    let wb = |l: &Linear<<Ad as burn::tensor::backend::AutodiffBackend>::InnerBackend>| -> (Vec<f32>, Vec<f32>) {
        (vec_of(l.weight.val()), l.bias.as_ref().map(|b| vec_of(b.val())).unwrap_or_default())
    };
    let (we, b1) = wb(&inner.extra);
    let (w2, b2) = wb(&inner.l2);
    let (w3, b3) = wb(&inner.l3);
    let file = NetFile {
        schema: NET_SCHEMA.into(),
        h1,
        h2,
        emb: vec_of(inner.emb.weight.val()),
        we,
        b1,
        w2,
        b2,
        w3,
        b3: b3[0],
    };
    let mlp = Mlp::new(file.clone())?;
    std::fs::write(out.join("net.json"), serde_json::to_vec(&file)?)?;
    let kr_rows = ordering_accuracy(&kr, &mlp, kr.max_plies_white.div_ceil(2));
    let kq_rows = ordering_accuracy(&kq, &mlp, kq.max_plies_white.div_ceil(2));
    std::fs::write(
        out.join("train-report.json"),
        serde_json::to_vec_pretty(&json!({
            "schema": "v500_train_report_v1", "seed": seed, "variant": a[3], "materials": materials,
            "epochs": epochs, "h1": h1, "h2": h2, "train_positions": train.n, "val_positions": val.n,
            "epochs_log": log, "ordering_test_krvk": kr_rows, "ordering_test_kqvk": kq_rows,
            "parameters": file.emb.len() + file.we.len() + file.b1.len() + file.w2.len() + file.b2.len() + file.w3.len() + 1,
        }))?,
    )?;
    let summarize = |rows: &[serde_json::Value]| -> String {
        rows.iter().map(|r| format!("d{}:{:.2}", r["depth"], r["top1_optimal"].as_f64().unwrap_or(0.0))).collect::<Vec<_>>().join(" ")
    };
    eprintln!("KRvK optimal-move top1 by depth: {}", summarize(&kr_rows));
    eprintln!("KQvK optimal-move top1 by depth: {}", summarize(&kq_rows));
    let _ = Piece::King;
    Ok(())
}
