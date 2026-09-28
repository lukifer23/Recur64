# Recur64 runtime sweep (Stage A)

- warmup: 3.25s | games/cell: 8

| requested conc | effective conc | peak in-flight | batch | timeout us | sims | ev/s | games/h | pos/s | trainable pos/s | batch mean/p50/p95 | wait p95 us | vram MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 8 | 8 | 16 | 16 | 1000 | 32 | 492.8 | 212.8 | 15.5 | 6.6 | 9.82/10/16 | 9704 | 289 |
