# Recur64 runtime sweep (Stage A)

- warmup: 3.72s | games/cell: 32

| requested conc | effective conc | peak in-flight | batch | timeout us | sims | ev/s | games/h | pos/s | trainable pos/s | batch mean/p50/p95 | wait p95 us | vram MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 12 | 12 | 24 | 24 | 1000 | 32 | 665.1 | 309.6 | 21.0 | 16.7 | 19.53/22/24 | 10767 | 289 |
