# Recorded correctness and scaling results

The final suite ran on ARM64 macOS with Python 3.13 and PyTorch 2.9.1. The one-rank baseline used `MoEModel(parallel=False)`; 2- and 4-rank runs used `MoEModel(parallel=True)`. All used learned top-2 routing, identical data and global expert initialization, and float64.

| Blocks | Total tokens | Epochs | Parallel ranks | Largest recorded reference difference |
|---:|---:|---:|---:|---:|
| 1 | 65 | 30 | 2 | 2.78e-17 |
| 1 | 65 | 30 | 4 | 2.78e-17 |
| 2 | 17 | 2 | 2 | 1.39e-17 |
| 2 | 17 | 2 | 4 | 1.94e-16 |

The first step compares token outputs, input gradients, every owned or replicated parameter gradient, and parameters after SGD. Every epoch compares global loss; final parameters are compared after training. The one-block 65-token loss fell from **0.095015 to 0.080596**. `results/verify/comparison-summary.json` has differences by category. These tests check observed learned routes; they do not force every expert to receive work or an owner to receive zero assignments.

## One-block CPU scaling

Each job ran 25 epochs, with 5 warmup and 20 measured epochs. The slowest rank defines step latency. The 256-token one-rank job is the baseline for both fixed-total and growing-total comparisons.

| Ranks | Total tokens | Parameter bytes/rank | Median ms | p95 ms | Forward / backward activation bytes (cluster) |
|---:|---:|---:|---:|---:|---:|
| 1 | 256 | 278,848 | 2.355 | 2.615 | 0 / 0 |
| 2 | 256 | 144,704 | 3.558 | 3.707 | 133,018 / 133,018 |
| 4 | 256 | 77,632 | 7.249 | 7.902 | 193,894 / 193,894 |
| 2 | 512 | 144,704 | 4.333 | 4.589 | 250,829 / 250,829 |
| 4 | 1,024 | 77,632 | 8.279 | 8.804 | 789,478 / 789,478 |

Expert parameter storage falls from 268,288 to 67,072 bytes/rank at four ranks. Shared storage stays 10,560 bytes/rank. The single-rank baseline avoids distributed exchange entirely; multi-rank runs perform it inside each parallel block. These are parameter tensor bytes, not peak process memory. Logical payload estimates exclude transport overhead. The small CPU model gets slower with more ranks; no GPU speedup is claimed.

Every saved job records its training and model configuration, environment, per-epoch loss, step latency, expert assignments, gradient norms, and logical communication estimates. See `results/summary.txt` for the full measured table and token trace. Linux and GPU executions were not tested.
