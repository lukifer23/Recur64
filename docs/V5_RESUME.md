# V5 HP resume and artifact transfer

## Worktree

```text
C:\Users\Caitl\Desktop\Code Projects\Recur64-v5
```

Branch: `experiment/hp-v5-counterfactual-loop`.

## Required missing artifact

Transfer the exact P25_DATA_V1 TRAIN file to a local ignored path, preferably:

```text
runs\v25\p25\data\proof-train.json
```

Required identity:

```text
positions: 44332
content digest: 3b25dc8549dd2fc9d47c30e294c273b3306aecb3eba91b964715326ddf74f2e6
FIT: 39929
FIT sorted-ID digest: a01932d7db863fbd0d160bc04bd3589137449d2c1e33d408f8d5d3f3cb45f8f5
DEV: 4403
DEV sorted-ID digest: f877219bc87d916ad8478a745f1572821c1a051337c3a071f2b37b9a5f83b899
```

Copy as a new file; do not replace another dataset. Run `recur64 v5 custody
verify` before any drill or training. A similar dataset or V3 TUNE is refused.
Regeneration is allowed only from the documented recipe with every exclusion
artifact and must reproduce the content digest bit-for-bit.

## Run safety

- Never start a fresh model in an existing run directory.
- Interrupted attempts are moved to a quarantine name, not deleted.
- Resume requires matching code/config/data/base/model hashes and restores model,
  optimizer, schedule, sampler and acquisition ordinal state.
- Every stage uses <=45-minute deterministic chunks and stops on unexpected
  failure.

Exact commands will be appended only after their CLI boundary tests execute the
documented examples successfully.

