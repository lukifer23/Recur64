//! V500 G0 for a trained net: (1) the deployed Rust forward must match a burn forward of the EXPORTED weights on the CPU
//! backend within 1e-4, on real validation children; (2) exact ordering accuracy by depth on the TEST split.
//!
//! cargo run --release -p recur64-prove --features flex --example check_net -- <net.json> <out.json>

use burn::module::Param;
use burn::nn::{Embedding, EmbeddingConfig, Linear};
use burn::prelude::*;
use burn::tensor::activation::relu;
use burn::tensor::{Int, TensorData};
use cozy_chess::{GameStatus, Piece};
use recur64_prove::net::{EXTRA, MAXP, Mlp, TOKENS, extras, ordering_accuracy, tokens};
use recur64_prove::split::{Part, part_of};
use recur64_prove::tb::{Tablebase, moves_of};
use serde_json::json;

type Cpu = burn::backend::Flex;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 3, "usage: check_net <net.json> <out.json>");
    let mlp = Mlp::load(std::path::Path::new(&a[1]))?;
    let f = mlp.file().clone();
    let dev = Default::default();
    let t2 = |v: &[f32], r: usize, c: usize| Tensor::<Cpu, 2>::from_data(TensorData::new(v.to_vec(), [r, c]), &dev);
    let mut emb: Embedding<Cpu> = EmbeddingConfig::new(TOKENS, f.h1).init(&dev);
    emb.weight = Param::from_tensor(t2(&f.emb, TOKENS, f.h1));
    let lin = |w: &[f32], b: &[f32], i: usize, o: usize| Linear::<Cpu> {
        weight: Param::from_tensor(t2(w, i, o)),
        bias: Some(Param::from_tensor(Tensor::<Cpu, 1>::from_data(TensorData::new(b.to_vec(), [o]), &dev))),
    };
    let extra = lin(&f.we, &f.b1, EXTRA, f.h1);
    let l2 = lin(&f.w2, &f.b2, f.h1, f.h2);
    let l3 = lin(&f.w3, &[f.b3], f.h2, 1);

    let kr = Tablebase::build("KRvK", &[Piece::Rook])?;
    let kq = Tablebase::build("KQvK", &[Piece::Queen])?;
    let mut worst = 0.0f32;
    let mut rows = 0usize;
    for tb in [&kr, &kq] {
        for b in tb.positions.iter().filter(|b| b.status() == GameStatus::Ongoing && part_of(b) == Part::Val).step_by(41).take(600) {
            for m in moves_of(b) {
                let mut c = b.clone();
                c.play_unchecked(m);
                if c.status() != GameStatus::Ongoing {
                    continue;
                }
                let (t, n) = tokens(&c);
                let ex = extras(&c);
                let ti: Vec<i32> = (0..MAXP).map(|i| i32::from(t[i])).collect();
                let mk: Vec<f32> = (0..MAXP).map(|i| f32::from(u8::from(i < n))).collect();
                let toks = Tensor::<Cpu, 2, Int>::from_data(TensorData::new(ti, [1, MAXP]), &dev);
                let mask = Tensor::<Cpu, 2>::from_data(TensorData::new(mk, [1, MAXP]), &dev);
                let exs = Tensor::<Cpu, 2>::from_data(TensorData::new(ex.to_vec(), [1, EXTRA]), &dev);
                let e = emb.forward(toks) * mask.unsqueeze_dim::<3>(2);
                let h = relu(e.sum_dim(1).squeeze_dim::<2>(1) + extra.forward(exs));
                let y = l3.forward(relu(l2.forward(h))).into_data().to_vec::<f32>().expect("f32")[0];
                worst = worst.max((mlp.eval(&c) - y).abs());
                rows += 1;
            }
        }
    }
    eprintln!("CPU burn forward vs Rust forward on {rows} validation children: max abs diff {worst:.2e}");
    let g0 = worst < 1e-4;
    let kr_rows = ordering_accuracy(&kr, &mlp, kr.max_plies_white.div_ceil(2));
    let kq_rows = ordering_accuracy(&kq, &mlp, kq.max_plies_white.div_ceil(2));
    let summarize = |rows: &[serde_json::Value]| -> String {
        rows.iter().map(|r| format!("d{}:{:.2}", r["depth"], r["top1_optimal"].as_f64().unwrap_or(0.0))).collect::<Vec<_>>().join(" ")
    };
    eprintln!("KRvK optimal top1 {}", summarize(&kr_rows));
    eprintln!("KQvK optimal top1 {}", summarize(&kq_rows));
    std::fs::write(
        &a[2],
        serde_json::to_vec_pretty(&json!({"schema": "v500_check_net_v1", "net": a[1], "cpu_parity_rows": rows,
            "cpu_parity_max_abs_diff": worst, "g0_parity_pass": g0, "ordering_test_krvk": kr_rows, "ordering_test_kqvk": kq_rows}))?,
    )?;
    anyhow::ensure!(g0, "G0 parity failed: {worst}");
    Ok(())
}
