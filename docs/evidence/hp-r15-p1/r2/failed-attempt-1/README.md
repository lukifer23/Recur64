# R2 failed attempt 1 (2026-09-27 18:46) — refused before any work

The pilot stopped at start-up with "reference checkpoint recurrence differs from
the run config". The config then pinned the R1 reference artifact (recorded
recurrence 1) for a recurrence-2 run. No self-play, training or evaluation ran.

This is kept for the record. The run directory is preserved as
`runs/hp-r15-smoke-r2-failed-refcheck`. The fix is the R15-P1 amendment in
`docs/HP_H3_PREREG.md`: each arm pins a recurrence-matched artifact of the
identical weights.
