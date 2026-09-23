//! Recur64 evaluation — the internal systems arena.
//!
//! Phase 2 compares a candidate checkpoint against a frozen reference using the
//! exact same rules profile, search budget, and recurrence. This is a systems
//! comparison, not an Elo claim, and no external engine is involved.

pub mod arena;

pub use arena::{ArenaConfig, ArenaResult, run_arena};
