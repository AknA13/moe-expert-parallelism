# Recorded correctness and scaling results

The final suite ran on ARM64 macOS with Python 3.13 and PyTorch 2.9.1. The one-rank baseline used `MoEModel(parallel=False)`; 2- and 4-rank runs used `MoEModel(parallel=True)`. All used learned top-2 routing, identical data and global expert initialization, float64, and Adam at learning rate 0.001.

| Blocks | Total tokens | Epochs | Parallel ranks | Largest recorded reference difference |
|---:|---:|---:|---:|---:|
| 1 | 65 | 1 | 2 | 5.55e-17 |
| 1 | 65 | 1 | 4 | 3.47e-17 |
| 1 | 65 | 30 | 2 | 1.56e-16 |
| 1 | 65 | 30 | 4 | 1.39e-16 |
| 2 | 17 | 2 | 2 | 9.71e-17 |
| 2 | 17 | 2 | 4 | 2.62e-15 |

The one-epoch jobs compare token outputs, input gradients, every owned or replicated parameter gradient, and parameters after Adam. All eight experts receive assignments (counts by expert: **12, 5, 19, 9, 5, 35, 26, 19**), have nonzero gradients, and change after the step. The replicated projection and router also have nonzero gradients and change. Their post-step values agree with the one-rank reference on every rank. See `results/correctness/parameter-updates.json` for each group's owner, gradient, and change. The 30-epoch jobs compare global loss at every epoch and final parameters; the one-block loss fell from **0.095015 to 0.014079**. [The loss plot](results/correctness/loss.svg) overlays all three rank counts. The tests do not force a zero-receive owner.

## One-block CPU scaling

Each job ran 25 epochs, with 5 warmup and 20 measured epochs. The slowest rank defines step latency. Payload columns are average logical bytes per measured step across the cluster. The 256-token one-rank job is the baseline for both fixed-total and growing-total comparisons.

| Ranks | Total tokens | Parameter bytes/rank | Median ms | p95 ms | Tokens/s at median | Forward / backward activation bytes | Routing metadata bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 256 | 278,848 | 2.606 | 3.061 | 98,247 | 0 / 0 | 0 |
| 2 | 256 | 144,704 | 3.859 | 4.092 | 66,333 | 132,966 / 132,966 | 2,094 |
| 4 | 256 | 77,632 | 7.142 | 7.930 | 35,842 | 192,077 / 192,077 | 3,097 |
| 2 | 512 | 144,704 | 4.450 | 4.533 | 115,058 | 250,573 / 250,573 | 3,931 |
| 4 | 1,024 | 77,632 | 8.061 | 9.172 | 127,036 | 789,581 / 789,581 | 12,433 |

Expert parameter storage falls from 268,288 to 67,072 bytes/rank at four ranks. Shared storage stays 10,560 bytes/rank. The single-rank baseline avoids distributed exchange entirely; multi-rank runs perform it inside each parallel block. These are parameter tensor bytes; Adam's moment tensors, gradients, activations, and process memory are excluded. Logical payload estimates exclude transport overhead. Twenty measured steps are descriptive rather than a reliable tail estimate. The small CPU model gets slower with more ranks for fixed total work; no GPU speedup is claimed.

Saved scaling jobs record their training and model configuration, environment, per-epoch loss and latency, and expert assignment counts. `scaling.py` saves [the derived summary](results/scaling/summary.json) and [plot](results/scaling/scaling.svg). `correctness.py` uses full tensors temporarily, then saves compact checks, update evidence, a token trace, and a loss plot in `results/correctness`. Linux and GPU executions were not tested.
