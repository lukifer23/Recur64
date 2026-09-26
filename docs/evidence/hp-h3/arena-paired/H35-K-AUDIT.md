# H3.5 arena K audit (H3.5B, MEASURED)

**Question:** did the H3.5 arena cells (`docs/evidence/hp-h3/arena/v{0,2}`) run at K = 2
as documented?

**Method:** rebuild the scientific identity with `recur64 config-info` (binary `2a1bd10`,
which leaves default identities unchanged; the P4.5 hash pin still passes) and compare
it with the hashes recorded in the evidence JSON.

The configuration is `runs/hp-h3-arena.toml` plus the CLI overrides `--arena-games 32`,
and for V2 also `--sample-plies 30 --noise-epsilon 0.25`.

| variant | K as parsed | scientific hash | recorded | match |
|---|---|---|---|---|
| V2, as written | 1 (key under `[model]`) | `0f1bb811eb2e…` | `0f1bb811eb2e…` | **yes** |
| V0, as written | 1 | `079599f6d429…` | `079599f6d429…` | **yes** |
| V2, K moved to top level | 2 | `e9e17e650ff0…` | `0f1bb811eb2e…` | no |
| V0, K moved to top level | 2 | `ca4bda6c3412…` | `079599f6d429…` | no |

**Result:** both H3.5 arena cells ran at **K = 1**.
- In `runs/hp-h3-arena.toml`, `search_leaves_in_flight = 2` and
  `run_budget_minutes = 60` sit below `[model]`.
- TOML scopes them to the model table, and serde ignored them without
  warning.

**Also found:** the pre-amendment `configs/hp/f15-smoke-v2.toml` (commits `73da820`,
`021af5c`) resolves to `510d9bb442a9…`. That is identical to the abandoned
`runs/hp-h3-smoke-v2` attempt, whose empty `[health_stops]` is outside the
identity.
