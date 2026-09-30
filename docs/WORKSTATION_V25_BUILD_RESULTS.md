# Workstation V2.5 — build results

Status: IN PROGRESS. Nothing below is claimed until a gate has actually run.

- Base main SHA: `fef1ffcf9c38381d4adc671e5e2c5ead9f141e33` (main had not advanced; `git fetch` run).
- Branch: `experiment/workstation-v25`; safety tag `main-pre-workstation-v25-fef1ffc`.
- HP branch `origin/experiment/hp-r15-h3-integration` advanced during fetch
  (`a8aef66` → `79ffd14`). CandidateFactsV1 and the exact mate-in-2 generator
  (`x2_data.rs`) exist there and were inspected read-only; the first exploration pass
  found neither because the tracking ref was stale.

| Gate | State |
|---|---|
| Identity / checkpoint refusal (commit 1) | unit tests pass (DETECTED by tests, TESTED) |
| P0 architecture correctness | NOT RUN |
| P0.5 facts cost | NOT RUN |
| P0.6 CUDA qualification | NOT RUN |
| Proof datasets | NOT RUN |
| P1 / P2 / P3 / P4 | NOT RUN |
