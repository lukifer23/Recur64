# Recur64 runtime sweep (Stage A)

- warmup: 3.40s | games/cell: 16

| requested conc | effective conc | peak in-flight | batch | timeout us | sims | ev/s | games/h | pos/s | trainable pos/s | batch mean/p50/p95 | wait p95 us | vram MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 12 | 12 | 24 | 24 | 1000 | 32 | 595.5 | 273.7 | 18.8 | 15.0 | 16.14/18/24 | 10442 | 289 |
