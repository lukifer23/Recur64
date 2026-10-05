# V5 Stage B final result: NO_SIGNAL

## Current V5 closure / proposed V6 (2026-10-05)

V5 is **CLOSED ? NO_SIGNAL**, with the frozen seed5301 experiment unchanged.
The owner-authorized closure investigation used existing DEV records and a frozen
216-position TRAIN panel; no retraining, new DEV inference or optimizer steps.
The V2 primary750 versus implemented pooled4500 discrepancy is now explicit.
Producer `3db24a9`, evaluator `7fb7461`, reviewed publication `3efce66` are separate
identities; later documentation commits do not replace them.

Start with [V5 closure](V5_CLOSURE.md), [measured diagnosis](V6_DIAGNOSTICS.md),
[one V6 proposal](V6_EXPERIMENT_PROPOSAL.md) and
[future P0 implementation prompt](V6_IMPLEMENTATION_PROMPT.md).
Recommendation: branch-local adversarial evidence backup with direct TRAIN
branch supervision, competent frozen B0, and same-information nonrecurrent
controls. This is a proposal: **V6 NOT IMPLEMENTED / NOT TRAINED**.
Gradient probes stopped at scratch preflight failure; backward results NOT RUN.
TRAIN/DEV/CONFIRM bytes remain unchanged; CONFIRM sealed and unevaluated.
Next step is owner design review, not a V5 resume or automatic V6 execution.

All lower status accounts and command blocks are preserved HISTORICAL material.

## Decision

**MEASURED classification: NO_SIGNAL. Recommend CLOSE V5 AND DESIGN V6.** The
fixed seed 5301 reader did not change a single DEV action under any evaluated
normal/control/composition condition. No loop benefit, benefit over B0 or useful
payload-specific gain met the frozen gates. This is a completed negative pilot,
not an engineering failure. No next architecture was implemented. Owner review
is the next step; no LR/updates/seed-shopping rescue.

## Engineering and immutable lineage

Starting HEAD `7f8a663d785e7cab3b9e8d49428dc4c6d32efee2`; preregistration `bd1e9460049f54f8eeba6c565cc25408652dbcb4`;
final evaluator source `7fb7461e94fa4819404913d9a51e91d4d154c167`; Stage-B producer
`3db24a926815159d592e93b60a8ae51852abad13`. Contract `v5_payload_shuffle_widening_v1`;
`v5_reader_evaluation_v3` / `v5_reader_pilot_report_v3`. Only `crates/recur64-cli/src/v5.rs`,
`crates/recur64-v5/src/evaluation.rs` and `crates/recur64-v5/src/study.rs` changed. Model, training math, acquisition, data,
loss, optimizer, sampler, CUDA settings and architecture config stayed unchanged:
7,162,896 FP32 parameters, physical microbatch 2, configuration
`849133a5cdf169f187778bace2f858aa4747d2e8defef3bb5ac1bffc839774ee`.

Affected rustfmt, V5 release/CLI boundary tests, CUDA Clippy, serial pinned build
and full release workspace PASS (604 passed, zero failed, two ignored). CLI Clippy
retains explicit allowances for untouched V4 warnings; unrelated historical
formatting was not changed. Fresh CPU/RTX 2050 CUDA qualification PASS: exact D9,
nine Q/R shapes, null=0, 50 resident updates, checkpoint/moment restoration and
exact continuation. Fresh disposable Q8/R4 drill PASS: loss 2.9871070881684623 to
0.018894116083780926, reduction 0.9936747777946704; baseline exact. No drill weights reused.

The evaluation-only bridge accepts ONLY the exact producer 3db24a9 update 0/800
artifacts after normal checkpoint integrity validation. Trainer resume semantics
remain unchanged; the bridge cannot authorize training/update 801.

Donor census before measurement: 72,000 recipients; 71,997 exact-depth mappings,
3 nearest-same-turn mappings (0.004167%), maximum depth distance 2, unresolved 0.
Each KRR cell had one uniform-frontier depth 6 -> 4 mapping; ranked schedule had none.
All donors differ from recipient root and preserve turn/family/root-depth cell.
Candidate pool min/mean/max 7/2274.1535972222223/4676. Payload only was replaced;
recipient graph structure and masks remained exact. Stable donor ordering is
explicit in v3, so only three widened eligibility pools does not imply only
three payload assignments differ from historical v2.

## Training: complete and not repeated

Stage B remains 800/800, seed 5301, recipe `v5_stage_recipe_v3` /
`55533c9a9d9125fd166542667d619b1d40b95119b5129d9a1c63072e296018fb`. One 1140.8644862-second training chunk, zero resumes,
28,800 examples; all losses finite. First 0.6637426661327481, first 50 mean
0.6186572668612643, last 50 mean 0.5513546268940809, final 0.820938692195341.
These sampled TRAIN losses are not held-out performance. Baseline 84 tensors are
exact across Stage A and all 81 Stage B checkpoints; reader 91 optimizer counters 800;
no baseline optimizer states. Each condition 1600 examples; seven cells 3204 each
and KRR M2/M3 3186 each. Frozen acquisition/exposure accounting validated.

Update0 model `2d1c770a43a6455148b774e9ddb552b6ca33efd9fdd5d37593cefe7c8ae0bb00`.
Update800 model `c7f6b10a0b60982199cd352c157a377115c6f00a46c699a6a9b3bb859e8bd273`.
Final optimizer `bb4138e1dc6ff83dcdc065dbd768b403b6e7f0f995d28968b56317c4862fe929`.
Baseline fingerprint unchanged `12b272a941e5b29589195a65c779a60509626d2108975e4793674ad5867d75c9`.

## Complete fixed DEV measurements

Six fresh v3 cells at each update,750 positions per cell,4500 unique per matrix,
174,000 treatment records per matrix. All 12 bounded CUDA invocations exited zero.
The three historical v2 partials were not reused. Exact normal replay PASS. B0
shared per-position actions and eight metrics match the accepted immutable
publication for 18,000 schedule records across both updates. Acquisition graph
manifest hashes and shuffle mappings are identical across update 0/update 800.
No evaluation was used to choose checkpoints or alter training.

| Update/condition | Top1 | Correct mass | Set loss | Uniform CE | Entropy | Centered correction L2 | KL vs B0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 baseline Q0/R0 | 0.792888889 | 0.723600115 | 0.663078197 | 1.47087103 | 0.871348876 | 0 | 0 |
| 0 normal Q8/R1 | 0.792888889 | 0.723599505 | 0.663077391 | 1.47086965 | 0.871355325 | 0.000485244407 | 3.9168088e-10 |
| 0 normal Q8/R4 | 0.792888889 | 0.723605014 | 0.663080457 | 1.47089711 | 0.871313042 | 0.00102295054 | 2.67149146e-09 |
| 800 baseline Q0/R0 | 0.792888889 | 0.723600115 | 0.663078197 | 1.47087103 | 0.871348876 | 0 | 0 |
| 800 normal Q8/R1 | 0.792888889 | 0.723729577 | 0.663130194 | 1.47148567 | 0.870378337 | 0.0201108766 | 5.90975494e-07 |
| 800 normal Q8/R4 | 0.792888889 | 0.724004414 | 0.663239882 | 1.47298591 | 0.868268319 | 0.0723108296 | 5.74131834e-06 |

All action-change rates are zero, including every intervention and composition
group at both updates. All R1->R4 transitions at every cell/schedule/Q have
wrong->right 0, right->wrong 0, wrong->different-wrong 0, unchanged-action 750.
Top1 correctness at Q8 is 3568/4500, with 932 wrong, at R1 and R4 in each schedule.

### Six-cell update-800 Q8/R4 summary

Means average the two schedules; each cell contains 750 unique positions.

| Cell | Correct / 750 | B0 top1 | Reader top1 | Correct mass | Set loss |
|---|---:|---:|---:|---:|---:|
| KQRvK M1 | 750 | 1 | 1 | 0.999953424 | 4.66152664e-05 |
| KQRvK M2 | 545 | 0.726666667 | 0.726666667 | 0.64531386 | 0.821983611 |
| KQRvK M3 | 432 | 0.576 | 0.576 | 0.448652216 | 1.27330478 |
| KRRvK M1 | 750 | 1 | 1 | 0.999956091 | 4.39137903e-05 |
| KRRvK M2 | 563 | 0.750666667 | 0.750666667 | 0.664578612 | 0.829846534 |
| KRRvK M3 | 528 | 0.704 | 0.704 | 0.585572279 | 1.05421384 |

### Update 800 primary-cell complete Q/R matrices

Every row n750. U=uniform_frontier_v1; D=base_ranked_depth_v1. Complete
update 0 matrices and all secondary metrics are in the aggregate archive.

#### KQRvK M3

| Schedule | Q | R | Top1 | Correct mass | Set loss | Uniform CE | Entropy | Centered L2 | KL vs B0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U | 2 | 1 | 0.576 | 0.448315677 | 1.27296245 | 2.57109032 | 1.57873199 | 0.018665755 | 9.38969781e-07 |
| U | 2 | 2 | 0.576 | 0.448496161 | 1.27313526 | 2.57237675 | 1.57704984 | 0.0406845746 | 4.07590032e-06 |
| U | 2 | 4 | 0.576 | 0.448661333 | 1.27331606 | 2.5736213 | 1.57544656 | 0.0655660537 | 9.14375279e-06 |
| U | 4 | 1 | 0.576 | 0.448316816 | 1.27296844 | 2.57110605 | 1.57871954 | 0.018037386 | 9.55542115e-07 |
| U | 4 | 2 | 0.576 | 0.448496677 | 1.27313906 | 2.57238661 | 1.57705054 | 0.0390517297 | 4.08319143e-06 |
| U | 4 | 4 | 0.576 | 0.448658721 | 1.27331388 | 2.57359634 | 1.57549452 | 0.0621682013 | 8.97547776e-06 |
| U | 8 | 1 | 0.576 | 0.448311354 | 1.27296469 | 2.57107292 | 1.57876857 | 0.0171226121 | 9.00701131e-07 |
| U | 8 | 2 | 0.576 | 0.448485542 | 1.27313052 | 2.57231285 | 1.57715591 | 0.0373486541 | 3.83064475e-06 |
| U | 8 | 4 | 0.576 | 0.448644242 | 1.27329334 | 2.57347649 | 1.57565006 | 0.0601412861 | 8.39619393e-06 |
| D | 2 | 1 | 0.576 | 0.448326989 | 1.27297505 | 2.57116462 | 1.57863447 | 0.0199035708 | 1.06015882e-06 |
| D | 2 | 2 | 0.576 | 0.44852106 | 1.27316072 | 2.57254198 | 1.57683533 | 0.0434293292 | 4.62805622e-06 |
| D | 2 | 4 | 0.576 | 0.448703946 | 1.27335331 | 2.57389172 | 1.57509486 | 0.0700644291 | 1.04943325e-05 |
| D | 4 | 1 | 0.576 | 0.448325387 | 1.27297688 | 2.57116039 | 1.57865061 | 0.0189505976 | 1.04326322e-06 |
| D | 4 | 2 | 0.576 | 0.448513576 | 1.27316258 | 2.57251586 | 1.57689513 | 0.0410940734 | 4.48651063e-06 |
| D | 4 | 4 | 0.576 | 0.448687252 | 1.27334918 | 2.57380935 | 1.5752429 | 0.0655546758 | 9.93619767e-06 |
| D | 8 | 1 | 0.576 | 0.448316217 | 1.27297476 | 2.57112022 | 1.5787193 | 0.0177955516 | 9.62125059e-07 |
| D | 8 | 2 | 0.576 | 0.448495916 | 1.27314929 | 2.57240928 | 1.57705059 | 0.038830534 | 4.09883364e-06 |
| D | 8 | 4 | 0.576 | 0.448660189 | 1.27331622 | 2.57361945 | 1.5754957 | 0.062573393 | 8.97485979e-06 |

#### KRRvK M3

| Schedule | Q | R | Top1 | Correct mass | Set loss | Uniform CE | Entropy | Centered L2 | KL vs B0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U | 2 | 1 | 0.704 | 0.585118461 | 1.05408803 | 2.07546408 | 1.27365437 | 0.0161366198 | 9.35440749e-07 |
| U | 2 | 2 | 0.704 | 0.585359093 | 1.05415536 | 2.07649012 | 1.27200712 | 0.0348395586 | 4.08241399e-06 |
| U | 2 | 4 | 0.704 | 0.585588176 | 1.05421754 | 2.07746816 | 1.27043043 | 0.0552435734 | 9.19206111e-06 |
| U | 4 | 1 | 0.704 | 0.585120521 | 1.05409091 | 2.07548091 | 1.27364289 | 0.0157000313 | 9.50782637e-07 |
| U | 4 | 2 | 0.704 | 0.585360254 | 1.05415487 | 2.07650344 | 1.27200841 | 0.0336564704 | 4.08317434e-06 |
| U | 4 | 4 | 0.704 | 0.585582804 | 1.05421799 | 2.07746634 | 1.27048054 | 0.0527155366 | 9.0017197e-06 |
| U | 8 | 1 | 0.704 | 0.585113947 | 1.05408869 | 2.07545438 | 1.2736906 | 0.0149710949 | 8.9352018e-07 |
| U | 8 | 2 | 0.704 | 0.585346615 | 1.05414882 | 2.07644437 | 1.27210863 | 0.0323166946 | 3.83195861e-06 |
| U | 8 | 4 | 0.704 | 0.585562919 | 1.05421219 | 2.07738011 | 1.2706248 | 0.0511935236 | 8.44608552e-06 |
| D | 2 | 1 | 0.704 | 0.5851304 | 1.05409116 | 2.07551143 | 1.27357603 | 0.0170546587 | 1.03504181e-06 |
| D | 2 | 2 | 0.704 | 0.585388078 | 1.05415439 | 2.07659539 | 1.27182891 | 0.0368854504 | 4.55240647e-06 |
| D | 2 | 4 | 0.704 | 0.585635013 | 1.0542212 | 2.07765461 | 1.27013619 | 0.0586172376 | 1.03551573e-05 |
| D | 4 | 1 | 0.704 | 0.585130401 | 1.05408696 | 2.07550942 | 1.27358662 | 0.0163924365 | 1.02347324e-06 |
| D | 4 | 2 | 0.704 | 0.585380303 | 1.05415698 | 2.07658328 | 1.27188409 | 0.0352113975 | 4.41327778e-06 |
| D | 4 | 4 | 0.704 | 0.585615373 | 1.05422247 | 2.07760701 | 1.27027614 | 0.0553086624 | 9.79892115e-06 |
| D | 8 | 1 | 0.704 | 0.585120378 | 1.05408708 | 2.07548002 | 1.27365097 | 0.0155169267 | 9.4346522e-07 |
| D | 8 | 2 | 0.704 | 0.585359241 | 1.05415371 | 2.07651045 | 1.27202434 | 0.03349639 | 4.05060666e-06 |
| D | 8 | 4 | 0.704 | 0.585581638 | 1.05421549 | 2.0774808 | 1.27050172 | 0.0531017227 | 8.91706268e-06 |

## Frozen pilot gates

**Classifier scope: all 4,500 DEV positions, two schedules averaged within each
position, as implemented in the existing frozen classifier.** The750-position
KQR M3 primary cell is reported separately above; these are not primary-cell CIs.
20,000 SplitMix64 resamples; existing seeds and ranks499/19499 unchanged.

| Contrast | Mean | Paired95% CI | Threshold | Pass |
|---|---:|---|---:|---|
| top1_q8_r4_minus_q8_r1 | 0 | [0, 0] | 0.03 | False |
| top1_q8_r4_minus_b0 | 0 | [0, 0] | 0.03 | False |
| set_loss_shuffle_minus_real_q8_r4 | -1.6042064e-07 | [-9.82526732e-07, 6.54967892e-07] | 0.01 | False |
| real_loop_benefit_minus_shuffled_loop_benefit | 0 | [0, 0] | 0.0 | False |

Schedule robustness fails: shuffle-minus-real loss is negative for the ranked
schedule (-5.015344287156474e-7); uniform=1.8069314777545728e-7. Loop and
B0 top1 effects are zero in both schedules. Engineering/integrity/accounting PASS.
Existing classifier returned exactly **NO_SIGNAL**; no manual override.

## Mechanism diagnostics

### Fixed interventions: update 800, allDEV, schedules averaged

| Treatment | R | Top1 | Correct mass | Set loss | Uniform CE | Entropy | Centered L2 | KL vs B0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 1 | 0.792888889 | 0.723729577 | 0.663130194 | 1.47148567 | 0.870378337 | 0.0201108766 | 5.90975494e-07 |
| normal | 4 | 0.792888889 | 0.724004414 | 0.663239882 | 1.47298591 | 0.868268319 | 0.0723108296 | 5.74131834e-06 |
| payload_shuffle | 1 | 0.792888889 | 0.72372962 | 0.663130334 | 1.47148568 | 0.870378049 | 0.0200769142 | 5.91485701e-07 |
| payload_shuffle | 4 | 0.792888889 | 0.724004531 | 0.663239722 | 1.47298557 | 0.86826722 | 0.0722123222 | 5.74662567e-06 |
| all_payload_null | 1 | 0.792888889 | 0.723600115 | 0.663078197 | 1.47087103 | 0.871348876 | 0 | 0 |
| all_payload_null | 4 | 0.792888889 | 0.723600115 | 0.663078197 | 1.47087103 | 0.871348876 | 0 | 0 |
| no_hypothesis_feedback | 1 | 0.792888889 | 0.723729577 | 0.663130194 | 1.47148567 | 0.870378337 | 0.0201108766 | 5.90975494e-07 |
| no_hypothesis_feedback | 4 | 0.792888889 | 0.724003835 | 0.66323966 | 1.47298347 | 0.868272552 | 0.0721981024 | 5.72747626e-06 |
| no_relation_bias | 1 | 0.792888889 | 0.723729424 | 0.663129944 | 1.47148464 | 0.870380162 | 0.0201121766 | 5.88713002e-07 |
| no_relation_bias | 4 | 0.792888889 | 0.724004264 | 0.663239997 | 1.472986 | 0.868268662 | 0.0722923394 | 5.73991543e-06 |

MEASURED: all-null is exactly B0; the trained reader has nonzero correction,
stronger than update 0, but real and shuffled content behave almost identically
on loss and never change selected actions. Feedback/relation interventions cause
small probability differences with no task benefit or action changes. Recurrence
executes, but useful content-sensitive decision correction is not demonstrated.

### Frozen KQR M3 composition

| Schedule | R | Group | Top1 | Correct mass | Set loss |
|---|---:|---|---:|---:|---:|
| U | 1 | composition_neither | 0.576 | 0.448158435 | 1.27279001 |
| U | 1 | composition_a | 0.576 | 0.448257213 | 1.27290473 |
| U | 1 | composition_b | 0.576 | 0.448260332 | 1.27290997 |
| U | 1 | composition_both | 0.576 | 0.448311354 | 1.27296469 |
| U | 4 | composition_neither | 0.576 | 0.448158435 | 1.27279001 |
| U | 4 | composition_a | 0.576 | 0.448464781 | 1.27310469 |
| U | 4 | composition_b | 0.576 | 0.448473754 | 1.27312364 |
| U | 4 | composition_both | 0.576 | 0.448644242 | 1.27329334 |
| D | 1 | composition_neither | 0.576 | 0.448158435 | 1.27279001 |
| D | 1 | composition_a | 0.576 | 0.448260371 | 1.27291775 |
| D | 1 | composition_b | 0.576 | 0.448263809 | 1.27290641 |
| D | 1 | composition_both | 0.576 | 0.448316217 | 1.27297476 |
| D | 4 | composition_neither | 0.576 | 0.448158435 | 1.27279001 |
| D | 4 | composition_a | 0.576 | 0.448478056 | 1.27313618 |
| D | 4 | composition_b | 0.576 | 0.448487857 | 1.27310461 |
| D | 4 | composition_both | 0.576 | 0.448660189 | 1.27331622 |

| Schedule | R | Set-loss interaction | Correct-mass interaction | Action changes neither->A/B/both |
|---|---:|---:|---:|---|
| uniform_frontier_v1 | 1 | -5.99984838e-05 | -4.77564446e-05 | 0/0/0 |
| uniform_frontier_v1 | 4 | -0.000144982943 | -0.000135858719 | 0/0/0 |
| base_ranked_depth_v1 | 1 | -5.9393625e-05 | -4.95288856e-05 | 0/0/0 |
| base_ranked_depth_v1 | 4 | -0.000134563711 | -0.000147289206 | 0/0/0 |

These small nonlinear probability interactions did not improve decisions and
are not evidence of reasoning.

### Loop health and acquisition

KQR M3 Q8/R4 factual stream means (full factual/null per-loop distributions,
all cells/schedules/Q/R, in the complete aggregate archive):

| Schedule | Loop | Evidence RMS | Hypothesis RMS | E update RMS | H update RMS | E entropy | H entropy | E max attention | H max attention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U | 1 | 1.00139335 | 1.00147752 | 0.364960233 | 0.39357733 | 4.8805455 | 4.31210264 | 0.00931540773 | 0.0236256497 |
| U | 2 | 1.00139596 | 1.0017008 | 0.142526057 | 0.189290053 | 4.88140728 | 4.30726594 | 0.0093067921 | 0.0233233024 |
| U | 3 | 1.00136234 | 1.00178324 | 0.123244637 | 0.156956198 | 4.88242095 | 4.31264524 | 0.00926862084 | 0.022713389 |
| U | 4 | 1.00130777 | 1.00177367 | 0.106822776 | 0.128960398 | 4.88318726 | 4.3245419 | 0.00920481936 | 0.022028638 |
| D | 1 | 1.00138484 | 1.00147494 | 0.296503294 | 0.393106773 | 4.88047516 | 4.31483699 | 0.00927270988 | 0.0236669242 |
| D | 2 | 1.00139702 | 1.00169729 | 0.143117363 | 0.188927211 | 4.88107157 | 4.30973063 | 0.00925979934 | 0.0233567197 |
| D | 3 | 1.00136954 | 1.00177994 | 0.123400106 | 0.157007394 | 4.88179257 | 4.31473 | 0.00923068756 | 0.0227267021 |
| D | 4 | 1.00131904 | 1.00177072 | 0.106767878 | 0.129191074 | 4.88241581 | 4.32621479 | 0.00917774854 | 0.0220160069 |

States/updates remain finite; state RMS about 1.001, updates decrease across
loops and attention is diffuse. This shows active, bounded loop computation,
not task-useful integration; it does not isolate a causal failure mechanism.
At KQR M3 Q8: both schedules actual Q8, no exhaustion,9 legal generations;
uniform mean maximumdepth 2.681333/branches 5.212 vs ranked 4.988/2.0.
All acquisition summaries (actualQ/depth/coverage/exhaustion/legal generations
and moves/padding/shared-core applications) are retained in the archive.

## Cross-process trajectory variance

Historical drills had the same selected positions, 48 graphs,seed,initial loss
2.9871070881684623 and model/training mathematics but final losses
0.026624600092569988 versus0.0977521538734436. CROSS-PROCESS TRAINING
TRAJECTORY VARIANCE OBSERVED; this is not proof of a defect or a new gate.
The authorized fresh-source drill also passed (final 0.018894116083780926).
A positive single-seed architectural claim would need independent-seed
replication. Here the frozen negative result stands; no repeat seeks a preferred
outcome and the owner matrix does not authorize seed shopping.

## Interpretation, custody and next review

MEASURED: no decision improvement, no loop benefit, no meaningful shuffle
specificity, exact null cancellation, weak feedback/relation effects, nonzero
bounded computation. INFERRED: this reader learned weak probability correction
without useful content-sensitive integration. The data do not establish that
returned information itself is insufficient; nor do they identify one proven
internal cause. Per owner decision: **CLOSE V5 AND DESIGN V6**, with a substantive
mechanism change. No V5.5/V6 code, retuning, extra updates or new seed in this pass.

Post actual raw-byte custody PASS for TRAIN/DEV/CONFIRM; pairwise exact-FEN and
canonical overlap 0. CONFIRM sealed=true,evaluated=false, no model invocation.
All 614 preserved files byte-identical, both Stage A/B file sets unchanged,
accepted B0 publication unchanged. Unavailable workstation raw cross-disjointness
remains NOT VERIFIED; those artifacts were not used.

R8 NOT RUN because classification is NO_SIGNAL. Q16 NOT RUN because Q8 passed.
No Stage A/B retraining,update 801,LR/loss/query/architecture rescue,multi-seed
replication,learned controller,self-play,CONFIRM,V4_TUNE or HOLDOUT_C evaluation.

## Evidence

- [Final compact result and raw report hashes](evidence/v5/stage-b-v3-summary.json)
- [Frozen pilot report](evidence/v5/stage-b-v3-pilot.json)
- [Ablation/composition report](evidence/v5/stage-b-v3-ablation.json)
- [Complete aggregate metrics and loop/acquisition diagnostics](evidence/v5/stage-b-v3-dev-summary.json.gz)
- [Source scope](evidence/v5/stage-b-v3-source-scope.json), [validation](evidence/v5/stage-b-v3-validation.json)
- [Donor census](evidence/v5/shuffle-donor-census-7fb7461.json)
- [Post custody](evidence/v5/stage-b-v3-post-custody.json), [preservation](evidence/v5/stage-b-v3-preservation.json)

## Preserved historical report (superseded operational status)

The following earlier donor-failure/partial-v2 account remains historical evidence.
The completed v3 result above is the current operational/scientific state.

# V5 Stage B: training complete; DEV evaluation blocked

## Engineering

Starting HEAD: `e40521fe661c87bf8a8e1f832200803bd1d50cae`.
Final scientific source: `3db24a926815159d592e93b60a8ae51852abad13`.
The preregistered `v5_stage_b_predecessor_bridge_v1` changes only exact CLI
predecessor/publication validation and focused tests. Model math, architecture,
data, loss, optimizer, sampler and acquisition are unchanged: 7,162,896 parameters,
FP32, configuration `849133a5cdf169f187778bace2f858aa4747d2e8defef3bb5ac1bffc839774ee`.

Fresh CPU and RTX 2050 CUDA qualification PASS, including unchanged exact D9,
all nine Q/R shapes, zero null error, 50 resident updates and checkpoint/moment
continuation. Graph provenance PASS. Full release workspace: 601 passed, zero
failed, two ignored. Affected formatting and V5 all-target Clippy PASS; unrelated
historical formatting was untouched. Fresh disposable Q8/R4 drill PASS:
2.9871070881684623 to 0.0977521538734436 mean loss, 96.7275% reduction, baseline
exact. Its weights were never reused.

See [validation](evidence/v5/stage-b-validation.json),
[lineage receipt](evidence/v5/stage-b-lineage-receipt.json) and
[source scope](evidence/v5/stage-b-source-scope.json).

## Training

Seed 5301 completed the frozen 800 updates in one bounded CUDA invocation:
1140.8644862 seconds, native exit zero, no resumes. Training execution is VALID.
Recipe remains `v5_stage_recipe_v3`; complete Stage B recipe digest:
`55533c9a9d9125fd166542667d619b1d40b95119b5129d9a1c63072e296018fb`.
Run: `runs/v5/v2/seed-5301/stage-b`.

Update-0 model SHA256:
`2d1c770a43a6455148b774e9ddb552b6ca33efd9fdd5d37593cefe7c8ae0bb00`.
Update-800 model SHA256:
`c7f6b10a0b60982199cd352c157a377115c6f00a46c699a6a9b3bb859e8bd273`.
Final optimizer SHA256:
`bb4138e1dc6ff83dcdc065dbd768b403b6e7f0f995d28968b56317c4862fe929`.

All 800 losses were finite: first 0.6637426661327481, first-50 mean
0.6186572668612643, last-50 mean 0.5513546268940809, final
0.820938692195341; diagnostic minimum 0.08405776543077081 and maximum
1.9292012327932753. Final LR zero. These are sampled TRAIN losses, not held-out
performance. Exactly 28,800 examples, two per condition per update.

| TRAIN cell | Exposure |
|---|---:|
| KQQvK M1 | 3204 |
| KQQvK M2 | 3204 |
| KQQvK M3 | 3204 |
| KQRvK M1 | 3204 |
| KQRvK M2 | 3204 |
| KQRvK M3 | 3204 |
| KRRvK M1 | 3204 |
| KRRvK M2 | 3186 |
| KRRvK M3 | 3186 |

Each of 18 condition samplers exposed 1600 examples, balanced at 177/178 per
cell. All 81 checkpoint generations retain the exact 84 baseline parameter
tensors; all model and final optimizer values are finite. Optimizer state contains
91 reader parameter IDs with counters 800 and zero baseline parameter IDs.
Baseline fingerprint before/after:
`12b272a941e5b29589195a65c779a60509626d2108975e4793674ad5867d75c9`.
The complete baseline parameter digest, computed read-only from checkpoints,
is also identical for Stage A, update 0 and update 800.

See [training integrity](evidence/v5/stage-b-training-summary.json),
[optimizer integrity](evidence/v5/stage-b-optimizer-integrity.json) and
[baseline parameters](evidence/v5/stage-b-baseline-parameters.json).

## DEV measurement

DEV was first used only after update 800 completed and passed integrity checks.
Three update-0 random-reader control cells completed: KQRvK M1/M2/M3, 750 each,
2250 unique positions. Normal replay parity PASS. Their 4500 B0 schedule records
agree exactly with the published Stage A report on all common policy metrics
and action indices. This is a partial cross-check, not a complete all-DEV result.

The fourth process, update-0 KRRvK M1, exited 1 after 111.594 seconds:

```text
Error: cannot derange V5_HP_DEV_V2-KRRvK-m1-.......................................K.R.............R....k... UniformFrontierV1 depth 6 within family/depth cell
```

The existing `shuffled_graphs` boundary in `study.rs:393` requires a donor from
a different root in the same family/root-mate-depth cell at the same acquired-node
depth. This recipient has zero eligible other-root donors at node depth 6.
Root mate depth M1 and acquired-node depth 6 are different quantities. The failure
does not establish a label error. No failed-cell report serialized and no metrics
were recovered from partial state. The original log and native receipt are preserved.

All further measurements stopped. No retry, changed donor pool, changed seed,
omitted position/control or scientific source change was made.

The [partial update-0 summary](evidence/v5/stage-b-update-zero-partial-summary.json)
contains every measured cell/schedule/Q/R/control policy row. For the primary
KQRvK M3 random-reader control, Q8/R1 and Q8/R4 both have top1 432/750 (0.576)
under both schedules, with zero action changes versus B0. Mean set losses:

| Schedule | Q8/R1 | Q8/R4 |
|---|---:|---:|
| uniform_frontier_v1 | 1.2727858525 | 1.2727949426 |
| base_ranked_depth_v1 | 1.2727835980 | 1.2727922221 |

These are untrained-reader controls. No update-800 DEV result exists. Neither
six-cell merge exists. The trained KQRvK M3 and KRRvK M3 Q/R matrices are NOT RUN.
See [failure evidence](evidence/v5/stage-b-evaluation-failure.json).

## Primary pilot gates

All five scientific contrasts/CIs and the final classifier invocation are NOT RUN:
the required complete update-800 evaluation is unavailable. Engineering
qualification passed, but complete evaluation integrity/accounting is unavailable.
No classifier label is assigned, including ENGINEERING_FAILURE or NO_SIGNAL.
No threshold or bootstrap contract changed.

## Mechanism diagnostics

Partial update-0 serialized controls/composition records remain preserved as
random-reader evidence. Final trained payload/null/shuffle, feedback/relation-bias,
composition, loop-state and action-transition analyses are NOT RUN. They cannot
be inferred from TRAIN loss or update-0 controls.

## Conditional R8

NOT RUN: PILOT_CANDIDATE was not established.

## Interpretation and next authorization

Stage B training is complete and valid; reader task benefit remains unmeasured.
The observed blocker is the evaluation shuffle donor eligibility contract.
The owner's final-source freeze and operational stop prohibit repairing/retrying
it within this experiment. A future authorization must prospectively specify
the control-contract treatment of empty donor pools, preserve both completed
checkpoints, and qualify any new evaluator/explicit predecessor bridge before
resuming measurement. No repair is implemented or scientific rescue proposed here.

Post-run actual TRAIN/DEV/CONFIRM custody PASS, all pairwise exact-FEN and canonical
overlaps zero. CONFIRM sealed=true, evaluated=false; no model invoked against it.
All 608 preserved files verified byte-for-byte, including the complete Stage A
file set, published B0 and all Stage B checkpoints. Historical failures remain.
Unavailable workstation-only raw-data cross-disjointness remains NOT VERIFIED.

See [post custody](evidence/v5/stage-b-post-custody.json),
[preservation](evidence/v5/stage-b-preservation.json) and
[compact run summary](evidence/v5/stage-b-seed5301-summary.json).

## Not run

Remaining update-0 KRRvK cells; every update-800 DEV cell; six-cell merges;
final ablation/composition analyses; pilot gates/classification; R8; Stage A
retraining; extra Stage B updates; LR tuning; intermediate DEV evaluation;
checkpoint selection; seeds 5302/5303; CONFIRM evaluation; query controller;
self-play; V4_TUNE; HOLDOUT_C. No architecture or dataset rescue.

V5 STAGE B TRAINING COMPLETE — PILOT CLASSIFICATION NOT RUN; EVALUATION BLOCKED.

V5_HP_CONFIRM_V2 REMAINS SEALED AND UNEVALUATED.
MULTI-SEED REPLICATION NOT RUN.
LEARNED QUERY CONTROLLER NOT TRAINED.
SELF-PLAY NOT RUN.
V4_TUNE_V1 AND HOLDOUT_C REMAIN UNEVALUATED.


## Continuation review and research direction (2026-10-05)

Origin fetched; starting HEAD b46be1b656e1ddb56ff715179e36c0eb45d1e111 is
clean and synchronized, ahead of the owner's7c288ed review snapshot. No recur64,
cargo or rustc process was active. Latest complete checkpoint is update800;
three update0 cell reports already exist, no update800 report exists.
All608 preserved local file hashes were rechecked unchanged. No training,
evaluation, drill or scientific-code modification was performed in this review.

Recommended direction: **HARNESS REPAIR; remain V5 pending valid measurement**.
The recorded empty shuffle-donor pool prevents the required complete measurement.
This is an operational evaluation-control failure, not a classifier output or
architecture result. The pilot classifier was not run; ENGINEERING_FAILURE is
not assigned as its output. The new continuation prompt expressly prohibits
study.rs/scientific changes and requires STOP plus owner review before repair.
Consequently complete matrices, paired CIs, final mechanism diagnostics and the
V5/V5.5/V6 scientific choice remain unavailable. No model-strength conclusion
can be drawn from sampled TRAIN loss or partial random-reader controls.

### Cross-process training trajectory variance observed

Historical d11659e and final-source3db24a9 disposable drills used the same seed,
selected positions, all48 graph manifests, initial mean loss2.9871070881684623,
and model/training math. Final mean losses differed:0.026624600092569988 versus
0.0977521538734436. Both overwhelmingly passed the unchanged engineering gate.
This is CROSS-PROCESS TRAINING TRAJECTORY VARIANCE OBSERVED, not proof of a
harness defect. CUDA settings were unchanged; neither drill nor seed5301 was
rerun to seek a preferred outcome. These measurements show that independent
GPU training trajectories need not be bitwise identical. Any future positive
single-seed Stage B result would require independent-seed replication before
strong architectural claims. No new repeatability gate is adopted.

See [continuation receipt](evidence/v5/stage-b-continuation-review.json).
