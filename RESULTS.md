# Recorded validation and measurements

The complete demo passed at 1, 2, and 4 ranks from a Python 3.13 virtual environment on ARM64 macOS with PyTorch 2.9.1. Linux was not tested.

| Ranks | Max verification error | Training loss, first → last |
|---:|---:|---:|
| 1 | 0 | 0.095015 → 0.080596 |
| 2 | 2.78e-17 | 0.095015 → 0.080596 |
| 4 | 2.78e-17 | 0.095015 → 0.080596 |

## Fixed global batch (256 tokens), learned routing

| Ranks | Parameter bytes/rank | Median ms | p95 ms | Forward / backward activation bytes (cluster) |
|---:|---:|---:|---:|---:|
| 1 | 278,848 | 2.726 | 2.815 | 0 / 0 |
| 2 | 144,704 | 3.543 | 3.638 | 133,018 / 133,018 |
| 4 | 77,632 | 6.953 | 7.409 | 193,894 / 193,894 |

Expert storage scales from 268,288 to 67,072 bytes/rank; shared storage stays 10,560 bytes/rank. Total parameter storage is therefore not exactly divided by four. CPU latency increases with rank count for this small model. This experiment demonstrates sharding and correctness, not a speedup.

Balanced fixture load imbalance is 1.0; the two-expert skew fixture is 4.0. Per-step routing, loss, gradient norm, timing, and reference-error logs are in `results/demo-*.json`. See `results/summary.txt` for both scaling regimes. Run-to-run timing variation is expected. No physical network traffic or GPU measurements are claimed.
