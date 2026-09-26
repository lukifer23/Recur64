# H3.5B P1.3 — initialization reproducibility (artifact vs semantic identity)

Binary: `ba3dde346459c9e6c2e0ac1905b9ea5fcc66c37f` (clean), `--features cuda`.
Config: `configs/hp/f15-reference-v2.toml` (seed 1, F15 head v2); CPU arms use the
same file with `device = "cpu"` only. Every freeze is a **separate process**.
Digests were computed with `recur64 model-digest` (CPU load, device-independent).

| checkpoint | device | built by | `model_id` (artifact) | semantic digest |
|---|---|---|---|---|
| `runs/cpu-a` | cpu | H3.1 | `ded3b796…` | `81b02bcd…` |
| `runs/cpu-b` | cpu | H3.1 | `42a7d980…` | `81b02bcd…` |
| `runs/h35b-freeze-cpu1` | cpu | H3.5B | `57be64ec…` | `81b02bcd…` |
| `runs/h35b-freeze-cpu2` | cpu | H3.5B | `13ff5351…` | `81b02bcd…` |
| `runs/h35b-freeze-cpu3` | cpu | H3.5B | `d8bf9ef1…` | `81b02bcd…` |
| `runs/det-a` | cuda | H3.1 | `247e0370…` | `f81938a2…` |
| `runs/det-b` | cuda | H3.1 | `b23597f5…` | `f81938a2…` |
| `runs/hp-h3-ref-f15-v2` (**frozen reference**) | cuda | H3.1 | `d89b408f…` | `f81938a2…` |
| `runs/h35b-freeze-cuda1` | cuda | H3.5B | `3f74e119…` | `f81938a2…` |
| `runs/h35b-freeze-cuda2` | cuda | H3.5B | `5315f6b1…` | `f81938a2…` |

Full digests: `digest-*.json`. Element-wise comparisons: `compare-*.json`.

## MEASURED

- 10 artifacts give 10 distinct `model_id`s but only **2 semantic digests**, one per
  backend.
- Same device, same seed, different processes: **0 of 15,154,632 elements
  differ** (`compare-cpu1-vs-cpu2.json`, `compare-ref-vs-cuda1.json`).
- The frozen reference `d89b408f…` is **semantically identical** to fresh CUDA
  freezes of the same config and seed.
- CPU (Flex) vs CUDA with the same seed: 115 of 136 tensors and 99.93 % of elements
  differ, with max |Δ| 0.183 and mean |Δ| 0.0283 (`compare-cpu1-vs-cuda1.json`).
  The 21 equal tensors are the constant-initialized ones: 18 RMSNorm gammas,
  `alpha_logit`, and the zero-initialized WDL weight and bias.

## INFERRED (not separately measured)

- The 1,699 differing `.mpk` bytes in the original D50 observation are the
  serialized `ParamId`s.
  - The record holds `{id: String, param}` per tensor (Burn 0.21
    `record/primitive.rs`), and the values are identical (measured above).
  - The byte diff forms 136 clusters, one per tensor, all within the base32
    alphabet that `ParamId::serialize` uses.
- CPU and CUDA differ because the two backends use different seeded RNGs:
  Flex uses `StdRng` in `burn-flex`, and CUDA uses `cubek-random`. This is
  expected, not a defect.
