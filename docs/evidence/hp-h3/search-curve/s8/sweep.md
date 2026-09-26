# Recur64 runtime sweep (Stage A)

- warmup: 2.42s | games/cell: 32

| requested conc | effective conc | peak in-flight | batch | timeout us | sims | ev/s | games/h | pos/s | trainable pos/s | batch mean/p50/p95 | wait p95 us | vram MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 8 | 8 | 16 | 16 | 1000 | 8 | 422.1 | 737.4 | 52.9 | 35.0 | 11.12/12/16 | 10475 | 298 |
