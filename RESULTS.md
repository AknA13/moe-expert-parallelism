# Recorded validation and measurements

All 12 command/rank combinations passed from a fresh virtual environment installed using the pinned requirements. Tested on ARM64 macOS with Python 3.13 and PyTorch 2.9.1. Linux was not tested.

| Ranks | Max verification error | Training loss, first → last |
|---:|---:|---:|
| 1 | 0 | 0.095015 → 0.080596 |
| 2 | 2.78e-17 | 0.095015 → 0.080596 |
| 4 | 2.78e-17 | 0.095015 → 0.080596 |

## Fixed global batch (256 tokens), learned routing

| Ranks | Parameter bytes/rank | Median ms | p95 ms | Forward / backward activation bytes (cluster) |
|---:|---:|---:|---:|---:|
| 1 | 278,848 | 2.505 | 2.686 | 0 / 0 |
| 2 | 144,704 | 3.266 | 3.456 | 133,018 / 133,018 |
| 4 | 77,632 | 5.235 | 5.564 | 193,894 / 193,894 |

Expert storage scales from 268,288 to 67,072 bytes/rank; shared storage stays 10,560 bytes/rank. Total parameter storage is therefore not exactly divided by four. CPU latency increases with rank count for this small model. This experiment demonstrates sharding and correctness, not a speedup.

Balanced fixture load imbalance is 1.0; the two-expert skew fixture is 4.0. See `results/summary.txt` for both scaling regimes and `results/scale-*.json` for full payload accounting. Run-to-run timing variation is expected. No physical network traffic or GPU measurements are claimed.
