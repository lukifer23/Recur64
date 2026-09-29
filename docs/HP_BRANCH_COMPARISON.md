# HP research branch vs mainline (2026-09-29)

This is a read-only comparison of `experiment/hp-r15` (H1/H2, head v1,
2026-09-23) and `experiment/hp-r15-h3-integration` (H3/R15, head v2, through
`a8aef66`, 2026-09-28) against mainline `d49149a`. Neither branch was
modified.

The integration branch merged mainline at `03e62f7`, before P4.6. It runs on
the HP laptop (RTX 2050) with the 15M-parameter F15/R15 models. Its findings
are **HP-branch evidence**: they come from a different model size and GPU,
and each is re-measured on mainline before it is relied on.

## Numbering

The branch's D50-D56 collide with mainline's D50-D53. Mainline keeps its own
numbers. A ported item gets the next mainline number, and the branch number
is cited.

## Findings and disposition

| HP finding | Evidence on the branch | Mainline disposition |
|---|---|---|
| **HP D54: arena trees mix both networks.** The side router sends each tree node to the network of that node's side to move. | Code path verified. Affects every searched arena, mainline D45 arenas included. | **Ported: mainline D54.** Verified on mainline (same code). `arena_tree_policy = "root_player_v1"`; the default reproduces old identities. New runs use it. |
| **Non-finite model output hidden.** A NaN policy became uniform; NaN WDL became value 0. | Core review. | **Ported** (D54): `evaluate_batch` now refuses non-finite output. |
| **Inference metrics race.** Counters were updated after replying. | Root cause of the branch's recurring test flake. | **Ported** (D54): count first, then reply. |
| **Conversion is the core weakness.** HP D56: a material lead does not predict a win (58.9% agreement). Train1: 109 of 111 repetition draws had the trained side >= +3 material. | Two independent measurements. | **Independent confirmation** of mainline's root cause (P4.6 draw diagnostic, stage 1). Supports the D51 curriculum now in its pilot. |
| **HP D55: fusion + autotune + candidate-width buckets.** +30% self-play and +11% training on the RTX 2050; the fusion drop-order leak was fixed. | Perf ledger #3-#10. | **To re-test on mainline (T6).** On the RTX 2000 Ada, fusion without buckets was 20-40% slower (T1). Buckets give stable kernel shapes, which fusion and autotune need. Autotune cannot guarantee FP32 (T5: TF32 candidates exist), so any adoption must verify the kernels used. |
| **HP D53: material adjudication of truncated arena games; promotion-v3.** | H3.6 arenas truncated 12-16%. | **Not needed now.** Mainline arenas truncate about 0% (D45, 512-ply cap). Revisit if truncation appears. |
| **HP arena RNG pairing (`paired_common_v1`).** Self-vs-self scores exactly 0.500. | H3.5B gate. | **Candidate:** variance reduction for promotion arenas. Pre-register before use. |
| **LR.** R1's best LR is about 7.5e-5 at 15M params; 3e-4 overfits a fixed replay. R4 needs about 4x lower LR than R1. | V1 fast sweeps, 2 seeds. | **Candidate for an F10 LR A/B** (mainline uses 3e-4). Different model size, so it needs its own pre-registered test. |
| **Recurrence** gives no value-learning benefit at 15M (R1 beats R4 at a matched LR, 2 seeds). | V1. | **Recorded** for the eventual R10 study. It is not a mainline result. |
| **Training produces strength:** untrained -> M 0.617 [0.563, 0.671]; M -> Train1 0.633 [0.595, 0.671]. These were mixed-tree arenas before D54, then adjudicated. | 192-game pre-registered checks. | **Encouraging but HP-specific.** Mainline P4.6 reached 0.578-0.594 against the reference, also mixed-tree. |
| **`snapshot_policy = latest`:** the newest healthy candidate is the actor (AlphaZero-style). | R15-T2 loop. | **Candidate** if promotion stays noisy after D54. |
| **T-1: truncated games train the policy** (`policy_only_v1`, +21.6% policy data). | HP T-1. | **Low value on mainline:** truncation is about 0%. |
| **Held-out value worse than uniform** as the value head specialises to its own play (Train1). | HP Train1. | **Consistent with** mainline's value-by-material drift (draw 0.69 -> 0.84). This is what D51 targets. |

## Effect on the running stage 2 pilot

`phase4-f10-cur` started before D54 and uses the historical mixed-tree arena,
as did its P4.6 comparator, so the comparison stays like for like. Its
pre-registered criteria (P1-P3) come from self-play, the value head and the
conversion probe, not from arena scores. Its promotion decisions are
mixed-tree measurements and are labelled as such. Runs after it use
`root_player_v1`.
