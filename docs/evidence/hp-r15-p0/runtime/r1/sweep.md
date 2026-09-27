# Recur64 runtime sweep (Stage A)

- warmup: 2.88s | games/cell: 32

| requested conc | effective conc | peak in-flight | batch | timeout us | sims | ev/s | games/h | pos/s | trainable pos/s | batch mean/p50/p95 | wait p95 us | vram MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 8 | 8 | 16 | 16 | 1000 | 32 | 508.6 | 275.4 | 16.0 | 12.1 | 13.95/15/16 | 9130 | 289 |
