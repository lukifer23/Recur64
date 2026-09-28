# Recur64 runtime sweep (Stage A)

- warmup: 3.65s | games/cell: 16

| requested conc | effective conc | peak in-flight | batch | timeout us | sims | ev/s | games/h | pos/s | trainable pos/s | batch mean/p50/p95 | wait p95 us | vram MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 8 | 8 | 16 | 16 | 1000 | 32 | 555.7 | 221.9 | 17.4 | 14.4 | 11.72/14/16 | 10312 | 289 |
