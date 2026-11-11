//! Recur64 V500: learned proof search for deep forced mates (`docs/V500_RESEARCH_PLAN.md`).
//!
//! `tb` is the exact retrograde tablebase (label source and grader only, never used inside the search under test),
//! `search` the sound AND/OR prover with pluggable move ordering and an independent proof checker.

pub mod tb;
