//! Recur64 evaluation — the internal systems arena.
//!
//! Phase 2 compares a candidate checkpoint against a frozen reference using the
//! exact same rules profile, search budget, and recurrence. This is a systems
//! comparison, not an Elo claim, and no external engine is involved.

pub mod arena;
pub mod openings;

pub use arena::{
    ADJUDICATION_RULE, ArenaAdjudicated, ArenaConfig, ArenaEarlyAdjudication, ArenaGameRecord,
    ArenaPairDiagnostics, ArenaResult, ArenaRngPolicy, ArenaTreePolicy, EarlyAdjudicationRecord,
    EarlyAdjudicationSummary, adjudicate_truncated, adjudicated_summary, arena_game_seed,
    early_adjudication_summary, material_balance_white, pair_diagnostics, play_indexed,
    play_indexed_until, run_arena,
};
pub use openings::{OpeningSuite, generate_openings};
