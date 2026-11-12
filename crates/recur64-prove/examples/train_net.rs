//! V500 P3 (v2): train the ordering network with burn using a LISTWISE RANKING loss on exact tablebase labels
//! (TRAIN split only). v1 (value regression) is recorded in the ledger as a failed first attempt.
//!
//! cargo run --release -p recur64-prove --features cuda --example train_net -- <out_dir> <seed> <variant> [epochs] [h1] [h2] [materials]
//!   variant: `full` or `extrap` (positions deeper than the cap are withheld from training)
//!   materials: `both` (default), `kq` or `kr`
//!
//! For every parent position (white or black to move) the network scores each child; the loss is the cross-entropy
//! between softmax(sign * score) over the children and the uniform distribution over the EXACTLY optimal children
//! (attacker: smallest plies-to-mate; defender: largest). The same scores order moves in the prover.

use std::time::Instant;

use burn::module::{AutodiffModule, Module};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation::{log_softmax, relu};
use burn::tensor::{Int, TensorData};
use cozy_chess::{Board, Color, GameStatus, Piece};
use recur64_prove::net::{EXTRA, MAXP, Mlp, NET_SCHEMA, NetFile, extras, ordering_accuracy, tokens_for, vocab};
use recur64_prove::split::{Part, part_of};
use recur64_prove::tb::{Tablebase, moves_of};
use serde_json::json;

#[cfg(feature = "cuda")]
type Ad = burn::backend::Autodiff<burn::backend::Cuda>;
#[cfg(not(feature = "cuda"))]
type Ad = burn::backend::Autodiff<burn::backend::Flex>;

const W: usize = 48;

#[derive(Module, Debug)]
struct Net<B: Backend> {
    emb: Embedding<B>,
    extra: Linear<B>,
    l2: Linear<B>,
    l3: Linear<B>,
}

impl<B: Backend> Net<B> {
    fn new(h1: usize, h2: usize, vocab: usize, device: &B::Device) -> Self {
        Self {
            emb: EmbeddingConfig::new(vocab, h1).init(device),
            extra: LinearConfig::new(EXTRA, h1).with_bias(true).init(device),
            l2: LinearConfig::new(h1, h2).with_bias(true).init(device),
            l3: LinearConfig::new(h2, 1).with_bias(true).init(device),
        }
    }

    /// `toks [n, MAXP]`, `mask [n, MAXP]`, `ex [n, EXTRA]` -> `[n]`
    fn forward(&self, toks: Tensor<B, 2, Int>, mask: Tensor<B, 2>, ex: Tensor<B, 2>) -> Tensor<B, 1> {
        let e = self.emb.forward(toks) * mask.unsqueeze_dim::<3>(2);
        let s = e.sum_dim(1).squeeze_dim::<2>(1) + self.extra.forward(ex);
        let h = relu(s);
        let z = relu(self.l2.forward(h));
        self.l3.forward(z).squeeze_dim::<1>(1)
    }
}

/// All children of all training parents, flattened.
struct Data {
    toks: Vec<u16>,
    ntok: Vec<u8>,
    ex: Vec<f32>,
    /// per parent: start into the child arrays, number of children, optimal bitmask, attacker (white) flag
    start: Vec<u32>,
    count: Vec<u8>,
    optimal: Vec<u64>,
    attacker: Vec<bool>,
}

type Batch<Bk> = (Tensor<Bk, 2, Int>, Tensor<Bk, 2>, Tensor<Bk, 2>, Tensor<Bk, 2>, Tensor<Bk, 2>, Tensor<Bk, 2>);

impl Data {
    fn new() -> Self {
        Self { toks: Vec::new(), ntok: Vec::new(), ex: Vec::new(), start: Vec::new(), count: Vec::new(), optimal: Vec::new(), attacker: Vec::new() }
    }

    fn parents(&self) -> usize {
        self.start.len()
    }

    fn add_parent(&mut self, tb: &Tablebase, b: &Board, token_set: &str) {
        let mv = moves_of(b);
        if mv.is_empty() || mv.len() > W {
            return;
        }
        let white = b.side_to_move() == Color::White;
        // child value: plies to mate of the child (0 = checkmated), None = no forced mate (draw / dead / unknown)
        let mut vals: Vec<Option<u8>> = Vec::with_capacity(mv.len());
        let start = self.ntok.len() as u32;
        for m in &mv {
            let mut a = b.clone();
            a.play_unchecked(*m);
            let v = if a.status() == GameStatus::Won { Some(0) } else { tb.plies_to_mate(&a) };
            vals.push(v);
            let (t, n) = tokens_for(token_set, &a);
            self.toks.extend_from_slice(&t);
            self.ntok.push(n as u8);
            self.ex.extend_from_slice(&extras(&a));
        }
        let mut mask = 0u64;
        if white {
            // attacker: the smallest known child value
            let Some(best) = vals.iter().flatten().min() else {
                self.rollback(start);
                return;
            };
            for (i, v) in vals.iter().enumerate() {
                if *v == Some(*best) {
                    mask |= 1 << i;
                }
            }
        } else {
            // defender: a child with no forced mate is the best reply; otherwise the largest plies-to-mate
            let any_none = vals.iter().any(Option::is_none);
            let best = vals.iter().flatten().max().copied();
            for (i, v) in vals.iter().enumerate() {
                if (any_none && v.is_none()) || (!any_none && *v == best) {
                    mask |= 1 << i;
                }
            }
        }
        self.start.push(start);
        self.count.push(mv.len() as u8);
        self.optimal.push(mask);
        self.attacker.push(white);
    }

    fn rollback(&mut self, start: u32) {
        let s = start as usize;
        self.toks.truncate(s * MAXP);
        self.ntok.truncate(s);
        self.ex.truncate(s * EXTRA);
    }

    fn batch<Bk: Backend>(&self, parents: &[usize], device: &Bk::Device) -> Batch<Bk> {
        let p = parents.len();
        let mut t = vec![0i32; p * W * MAXP];
        let mut m = vec![0f32; p * W * MAXP];
        let mut e = vec![0f32; p * W * EXTRA];
        let mut cm = vec![0f32; p * W];
        let mut tg = vec![0f32; p * W];
        let mut sg = vec![0f32; p];
        for (r, &pi) in parents.iter().enumerate() {
            let s = self.start[pi] as usize;
            let n = self.count[pi] as usize;
            let opt = self.optimal[pi];
            let k = opt.count_ones() as f32;
            sg[r] = if self.attacker[pi] { -1.0 } else { 1.0 };
            for c in 0..n {
                let ci = s + c;
                for j in 0..MAXP {
                    t[(r * W + c) * MAXP + j] = i32::from(self.toks[ci * MAXP + j]);
                    m[(r * W + c) * MAXP + j] = f32::from(u8::from(j < self.ntok[ci] as usize));
                }
                for j in 0..EXTRA {
                    e[(r * W + c) * EXTRA + j] = self.ex[ci * EXTRA + j];
                }
                cm[r * W + c] = 1.0;
                if opt >> c & 1 == 1 {
                    tg[r * W + c] = 1.0 / k;
                }
            }
        }
        (
            Tensor::from_data(TensorData::new(t, [p * W, MAXP]), device),
            Tensor::from_data(TensorData::new(m, [p * W, MAXP]), device),
            Tensor::from_data(TensorData::new(e, [p * W, EXTRA]), device),
            Tensor::from_data(TensorData::new(cm, [p, W]), device),
            Tensor::from_data(TensorData::new(tg, [p, W]), device),
            Tensor::from_data(TensorData::new(sg, [p, 1]), device),
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

/// Listwise loss and the top-1 optimal rate for one batch.
fn loss_and_acc<Bk: Backend>(net: &Net<Bk>, data: &Data, parents: &[usize], device: &Bk::Device) -> (Tensor<Bk, 1>, f32) {
    let (t, m, e, cm, tg, sg) = data.batch::<Bk>(parents, device);
    let p = parents.len();
    let u = net.forward(t, m, e).reshape([p, W]);
    let z = (u * sg).mask_fill(cm.equal_elem(0.0), -1.0e9);
    let logp = log_softmax(z.clone(), 1);
    let loss = (logp * tg.clone()).sum_dim(1).neg().mean();
    let best = z.argmax(1);
    let hit = tg.gather(1, best).greater_elem(0.0).float().mean();
    (loss, vec_of(hit)[0])
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
    let epochs: usize = a.get(4).map_or(Ok(30), |s| s.parse())?;
    let h1: usize = a.get(5).map_or(Ok(256), |s| s.parse())?;
    let h2: usize = a.get(6).map_or(Ok(64), |s| s.parse())?;
    let materials = a.get(7).map_or("both", String::as_str).to_string();
    let token_set = a.get(9).map_or("v2", String::as_str).to_string();
    std::fs::create_dir_all(&out)?;

    // Materials: (tablebase, extrapolation cap in plies). The cap withholds positions deeper than it from training.
    let mut tbs: Vec<(Tablebase, u8)> = Vec::new();
    if matches!(materials.as_str(), "both" | "kq") {
        tbs.push((Tablebase::build("KQvK", &[Piece::Queen])?, 11));
    }
    if matches!(materials.as_str(), "both" | "kr") {
        tbs.push((Tablebase::build("KRvK", &[Piece::Rook])?, 15));
    }
    if materials == "kbn" {
        tbs.push((Tablebase::build("KBNvK", &[Piece::Bishop, Piece::Knight])?, 41));
    }
    anyhow::ensure!(!tbs.is_empty(), "materials {materials}");
    // Four-piece endings have tens of millions of positions: keep a deterministic symmetry-hash sample of parents.
    let max_parents: usize = a.get(8).map_or(Ok(if materials == "kbn" { 1_500_000 } else { usize::MAX }), |s| s.parse())?;
    let (mut train, mut val) = (Data::new(), Data::new());
    let t0 = Instant::now();
    for (tb, cap) in &tbs {
        let total = tb.positions.iter().filter(|b| b.status() == GameStatus::Ongoing).count();
        let keep_mod = (total / max_parents.max(1)).max(1) as u64;
        for b in tb.positions.iter().filter(|b| b.status() == GameStatus::Ongoing) {
            if keep_mod > 1 && recur64_prove::split::canon_key(b) % keep_mod != 0 {
                continue;
            }
            match part_of(b) {
                Part::Train => {
                    if extrap && tb.plies_to_mate(b).is_some_and(|x| x > *cap) {
                        continue;
                    }
                    train.add_parent(tb, b, &token_set);
                }
                Part::Val => val.add_parent(tb, b, &token_set),
                Part::Test => {}
            }
        }
    }
    eprintln!("parents: train {} val {} ({} train children), built in {:.0}s (variant {}, materials {materials})", train.parents(), val.parents(), train.ntok.len(), t0.elapsed().as_secs_f64(), a[3]);

    let device = Default::default();
    <Ad as burn::tensor::backend::Backend>::seed(&device, seed);
    let mut model = Net::<Ad>::new(h1, h2, vocab(&token_set), &device);
    let mut optim = AdamWConfig::new().with_weight_decay(1e-4).init::<Ad, Net<Ad>>();
    let batch = 256usize;
    let steps_total = (train.parents() / batch).max(1) * epochs;
    let mut rng = Sm(seed ^ 0xA5A5_5A5A);
    let mut order: Vec<usize> = (0..train.parents()).collect();
    let mut step = 0usize;
    let t1 = Instant::now();
    let mut log = Vec::new();
    for epoch in 0..epochs {
        for i in (1..order.len()).rev() {
            order.swap(i, (rng.next() % (i as u64 + 1)) as usize);
        }
        let (mut sum, mut acc, mut count) = (0.0f64, 0.0f64, 0usize);
        for chunk in order.chunks(batch).filter(|c| c.len() == batch) {
            let lr = 2e-3 * (1.0 - step as f64 / steps_total as f64).max(0.02);
            let (loss, hit) = loss_and_acc::<Ad>(&model, &train, chunk, &device);
            sum += f64::from(vec_of(loss.clone())[0]);
            acc += f64::from(hit);
            count += 1;
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(lr, model, grads);
            step += 1;
        }
        let inner = model.valid();
        let idev = Default::default();
        let vidx: Vec<usize> = (0..val.parents()).collect();
        let (mut vl, mut va, mut vn) = (0.0f64, 0.0f64, 0usize);
        for chunk in vidx.chunks(batch).filter(|c| c.len() == batch) {
            let (l, h) = loss_and_acc::<<Ad as burn::tensor::backend::AutodiffBackend>::InnerBackend>(&inner, &val, chunk, &idev);
            vl += f64::from(vec_of(l)[0]);
            va += f64::from(h);
            vn += 1;
        }
        let rec = json!({"epoch": epoch, "train_loss": sum / count.max(1) as f64, "train_top1": acc / count.max(1) as f64,
                         "val_loss": vl / vn.max(1) as f64, "val_top1": va / vn.max(1) as f64, "secs": t1.elapsed().as_secs_f64()});
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
    let file = NetFile { schema: NET_SCHEMA.into(), token_set: token_set.clone(), h1, h2, emb: vec_of(inner.emb.weight.val()), we, b1, w2, b2, w3, b3: b3[0] };
    let mlp = Mlp::new(file.clone())?;
    std::fs::write(out.join("net.json"), serde_json::to_vec(&file)?)?;

    // G0: the deployed Rust forward must match the burn forward on real validation children.
    let parity = {
        type I = <Ad as burn::tensor::backend::AutodiffBackend>::InnerBackend;
        let dv = Default::default();
        let idx: Vec<usize> = (0..val.parents().min(64)).collect();
        let (t, m, e, _, _, _) = val.batch::<I>(&idx, &dv);
        let burn_pred = vec_of(inner.forward(t, m, e));
        let mut worst = 0.0f32;
        for (r, &pi) in idx.iter().enumerate() {
            for c in 0..val.count[pi] as usize {
                let ci = val.start[pi] as usize + c;
                let n = val.ntok[ci] as usize;
                let tk: Vec<u16> = val.toks[ci * MAXP..ci * MAXP + n].to_vec();
                let ex = [val.ex[ci * EXTRA], val.ex[ci * EXTRA + 1], val.ex[ci * EXTRA + 2]];
                worst = worst.max((mlp.eval_features(&tk, &ex) - burn_pred[r * W + c]).abs());
            }
        }
        worst
    };
    // On CUDA the burn matmul may run below f32 precision (observed 9.8e-3 vs the exact-f32 Rust forward), so this number is
    // recorded, not gated. The G0 parity gate is the CPU burn forward of the exported weights (`check_net`).
    eprintln!("GPU burn forward vs Rust forward, max abs diff (recorded, not the G0 gate): {parity:.2e}");

    let mut ordering = serde_json::Map::new();
    for (tb, _) in &tbs {
        let rows = ordering_accuracy(tb, &mlp, tb.max_plies_white.div_ceil(2));
        let line = rows.iter().map(|r| format!("d{}:{:.2}", r["depth"], r["top1_optimal"].as_f64().unwrap_or(0.0))).collect::<Vec<_>>().join(" ");
        eprintln!("{} optimal-move top1 by depth: {line}", tb.name);
        ordering.insert(tb.name.clone(), json!(rows));
    }
    std::fs::write(
        out.join("train-report.json"),
        serde_json::to_vec_pretty(&json!({
            "schema": "v500_train_report_v2", "loss": "listwise_ranking", "seed": seed, "variant": a[3], "materials": materials, "token_set": token_set,
            "epochs": epochs, "h1": h1, "h2": h2, "train_parents": train.parents(), "val_parents": val.parents(),
            "gpu_parity_max_abs_diff": parity, "epochs_log": log, "ordering_test": ordering,
            "parameters": file.emb.len() + file.we.len() + file.b1.len() + file.w2.len() + file.b2.len() + file.w3.len() + 1,
        }))?,
    )?;
    Ok(())
}
