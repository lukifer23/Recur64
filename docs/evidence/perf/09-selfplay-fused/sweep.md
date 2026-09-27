# Recur64 runtime sweep (Stage A)

- warmup: 6.14s | games/cell: 8

| requested conc | effective conc | peak in-flight | batch | timeout us | sims | ev/s | games/h | pos/s | trainable pos/s | batch mean/p50/p95 | wait p95 us | vram MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 8 | 8 | 16 | 16 | 1000 | 32 | 569.6 | 202.0 | 17.9 | 9.5 | 11.93/13/16 | 10429 | 289 |
