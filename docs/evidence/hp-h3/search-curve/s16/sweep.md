# Recur64 runtime sweep (Stage A)

- warmup: 3.12s | games/cell: 32

| requested conc | effective conc | peak in-flight | batch | timeout us | sims | ev/s | games/h | pos/s | trainable pos/s | batch mean/p50/p95 | wait p95 us | vram MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 8 | 8 | 16 | 16 | 1000 | 16 | 442.1 | 397.0 | 27.7 | 20.8 | 11.72/14/16 | 10175 | 298 |
