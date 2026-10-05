# V5 V2 production integration

DATA-V2-A: 738db983084a998664ce962f87fa3c4a6153f526.
DATA-V2-B: 37ea10e; measured manifests/seal pushed before adoption.

Active data contract: v5_hp_data_v2. Active scientific recipe:
v5_stage_recipe_v3. Historical V1 preregistration remains infeasible; no P25 or
light-family capacity position is an active V5 data identity.

TRAIN has nine equal cells of 3000 (27000 total), KQQ/KQR/KRR x M1/M2/M3.
The retained internal fit field contains every TRAIN index. DEV is an independent
file with six cells of750 (4500 total); primary KQR M3 n750. Each V5Data instance
owns one role. TRAIN/DEV production loaders and scientific entry points enforce
that role. CONFIRM has custody metadata access only; no ordinary V5Data loader
or debug capability can construct a CONFIRM training/evaluation instance.

Compiled scientific constants bind measured producer source, configuration,
record/target/set digests and exact raw-byte SHA256. Committed bindings are
cross-checked against those constants; custody reads actual local bytes.
The producer SHA remains immutable across consumer implementation commits.
Fresh consumer CPU/CUDA qualification must bind the new scientific source.

Stage A/B hyperparameters, optimizer, loss, FP32, query conditions and acquisition
seeds are unchanged. The existing cell_balanced_v1 sampler now sees exactly nine
TRAIN cells. Every update records aggregate and per-condition cell exposure.
Recipe-v1/v2 identities are refused by active recipe-v3 validation. Future Stage B
initial-model/baseline hashes cannot be preregistered before Stage A exists.
The recipe command prints a source-bound frozen contract digest and a complete
Stage A recipe digest without initializing a model or running training.

Evaluation and baseline report schemas are version2, bound to measured V2 target
digests and independent DEV role. Merge requires six exact 750-position cells,
4500 total, primary750. Practical thresholds/bootstrap algorithm are unchanged.
The drill report schema is version2, still stable-ID-hash four TRAIN positions
per KQR/KRR/depth cell, Q8/R4, <=200 updates, LR1e-3, warmup20 and existing loss
criterion. The pre-training drill CLI requires current CPU qualification, current
CUDA qualification and actual local TRAIN/DEV/CONFIRM custody/disjointness/seal.
No Stage A/B or DEV model evaluation is authorized by integration or gate PASS.

Expected raw files:
- runs/v5/data/v2/v5-hp-train-v2.json
- runs/v5/data/v2/v5-hp-dev-v2.json
- runs/v5/data/v2/v5-hp-confirm-v2.json

Engineering commands: v5 custody --split train|dev|confirm, v5 data verify,
v5 recipe. They perform no model evaluation. Full current-source release tests,
Clippy, changed-file formatting, CPU/CUDA qualification and graph provenance are
pending at this implementation checkpoint. Drill remains conditional.
