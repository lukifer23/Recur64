# Recur64 runtime sweep (Stage A)

- warmup: 3.62s | games/cell: 32

| requested conc | effective conc | peak in-flight | batch | timeout us | sims | ev/s | games/h | pos/s | trainable pos/s | batch mean/p50/p95 | wait p95 us | vram MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 16 | 16 | 64 | 64 | 1000 | 64 | 525.7 | 161.0 | 8.3 | 6.6 | 36.10/41/64 | 11076 | 586 |
