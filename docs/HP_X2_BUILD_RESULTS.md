# Chimera V2 — build results (engineering gates)

All MEASURED on this branch; tests are in-tree.

* WorldModel staged: Root / Successor / Replies horizons do different chess work
  (`horizon_tests`, `tool_execution_is_staged_per_budget_...`); sections beyond the horizon
  are exactly zero and their work is not executed. Shared unavoidable cost: Root applies
  every root move and generates opponent legal moves (`attacked_after`).
* Terminal children (mate, stalemate, insufficient, fifty-move) expose zero replies and
  zero summaries; a non-terminal control keeps its replies.
* GameState differential (`world_semantics.rs`): explicit edge fixtures, the 218-move
  position, 320 random fresh positions (>5000 candidates; every reply record compared as a
  multiset), 300 positions for root-fact parity against CandidateFactsV1.
* Native == WASM byte-exact at all three horizons (`x2 bench`, `compute` tests).
* History contract `fresh_no_history_v1` in the V2 identity; world model refuses
  repetition > 1; dataset audit refuses non-fresh positions.
* Capacity audit recomputed at start of train/eval/bench; refuses on overflow. Measured
  max legal 60/54/58 (train/tune/confirm), max replies 8; caps 64/16.
* Config: V1-only fields and `output_blocks != 0` refused by V2.
* Provenance: `COPROC_SOURCE_CONTRACT_SHA256` (tested) + `BUILT_AT_GIT_REV` (build
  environment only). WORLD_MODEL_VERSION = `world_model_v2_staged_v1`.
* Planner bounded through T8 after tiny variable-budget training (test).
* Params 14.41 M (board encoder 11.14 M, planner 1.19 M, successor 0.60 M, reply 0.20 M).
* NOT RUN: visual A/B, process-supervision rescue, self-play, CUDA lifecycle plateau test.
