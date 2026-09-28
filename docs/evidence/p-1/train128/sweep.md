# Recur64 runtime sweep (Stage A)

- warmup: 3.60s | games/cell: 8

| requested conc | effective conc | peak in-flight | batch | timeout us | sims | ev/s | games/h | pos/s | trainable pos/s | batch mean/p50/p95 | wait p95 us | vram MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 8 | 8 | 16 | 16 | 1000 | 128 | 566.5 | 83.8 | 4.5 | 4.5 | 11.69/13/16 | 10769 | 289 |
