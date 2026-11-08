# Expert-parallel MoE, from one model to many ranks

A private CPU/Gloo teaching demo with learned top-2 routing, a configurable expert count, and a configurable number of MoE blocks. One `MoEModel` selects local or expert-parallel blocks with its `parallel` constructor argument.

## Approach

### Model architecture: two ranks

The default model has one MoE block, token dimension 32, eight experts, and top-2 routing. In this two-rank example, each rank gets different input tokens. Each has its own copy of the **same** token-projection and router weights, while the eight expert MLPs are split into disjoint owner groups.

![Two-rank MoE block with bordered projection and router weight groups and neural-network expert glyphs](assets/model-architecture.svg)

The two tall borders group the matching projection and router weights across the ranks. Each small network glyph represents a separate expert MLP. This is a **parameter-placement view**; tokens can route across the rank boundary, as the forward diagram shows. The projection is `Linear(32, 32) + Tanh`, the router is `Linear(32, 8)`, and each expert is `Linear(32, H) + Tanh + Linear(H, 32)`. The router selects two expert IDs per token and softmax-normalizes their two selected logits into combination weights. There is no capacity limit, token dropping, or load-balancing loss. With more blocks, this structure repeats with new weights at every block. The single-rank `StandardMoEBlock` owns all eight experts and serves as the numerical reference.

### Forward pass: dispatch, compute, return

Follow token 2 from rank 0 in the [saved trace](results/correctness/demo.json): its router selects **expert 7** on rank 1 and **expert 0** on rank 0. Rank 0 repeats its projected token row once per selected expert, then packs assignment rows by destination rank. The arrows below show the remote assignment; both ranks run the same sequence for their own token shards.

```mermaid
sequenceDiagram
    participant S as Rank 0: token source
    participant O as Rank 1: expert 7 owner
    Note right of S: Token 2 selects experts 7 and 0
    Note over S,O: Each arrow pair is one all-to-all call
    S->>O: C1 counts: rows rank 1 will receive
    O->>S: C1 counts: rows rank 0 will receive
    S->>O: C2 expert IDs in packed row order
    O->>S: C2 expert IDs for rank 0's owned rows
    S->>O: C3 projected token rows for remote experts
    O->>S: C3 projected rows for rank 0's experts
    Note left of O: Run expert 7 on token 2
    Note right of S: Run expert 0 on token 2
    O->>S: C4 expert outputs back to token sources
    S->>O: C4 outputs for rank 1's source tokens
    Note right of S: Restore order and combine outputs
```

**C1** and **C2** are `dist.all_to_all_single` calls that exchange counts and integer expert IDs. **C3** and **C4** use autograd-aware `all_to_all_single` for activation rows. Rows destined for a rank's own experts remain local within those collectives. The return uses inverse split sizes; the source undoes its packing permutation, restores token/slot order, and combines the two expert outputs with the selected weights. Every rank participates even when it has zero rows for a destination.

### Backward pass: reverse the activation exchanges

Each source rank computes its **summed token loss divided by the global token count** `N`. The two local contributions add up to the global mean loss:

```text
L_r = (1/N) · Σ_{tokens i on rank r} mean_j (prediction_ij - target_ij)²
L_global = L_0 + L_1
```

For token 2, the source first differentiates the weighted combination. The gradient for expert 7's output goes to its owner; the expert backpropagates and sends its input gradient back to the source.

```mermaid
sequenceDiagram
    participant S as Rank 0: token source
    participant O as Rank 1: expert 7 owner
    Note right of S: Backpropagate weighted sum
    S->>O: B1 expert 7 output gradient, backward of C4
    Note left of O: Expert 7 backward, owner gradients
    O->>S: B2 projected-row gradient, backward of C3
    Note right of S: Local and remote gradients meet
    Note over S,O: B3 SUM all-reduce shared gradients
    S->>O: B3 rank 0 shared-gradient contribution
    O->>S: B3 rank 1 shared-gradient contribution
    Note over S,O: Both ranks now run Adam
```

**B1** and **B2** are the backward calls of the two autograd-aware activation exchanges. The arrows at **B3** show the logical contributions to one `dist.all_reduce(SUM)` per replicated parameter, not Gloo's physical algorithm. We **do not divide by rank count again**: each local loss already uses the global `N`. Expert gradients accumulate on their owner from every source rank and are **not all-reduced**. Integer routing IDs and counts have no backward exchange; gradients reach selected router logits through the combination weights. All experts, including those with zero received rows, retain an autograd path.

## Reproduce

Use Python 3.13. The recorded environment was ARM64 macOS with PyTorch 2.9.1; pins are in `requirements.txt`.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python experiments/correctness.py && python experiments/scaling.py
```

`experiments/correctness.py` prints named pass/fail checks and saves a loss plot and [short results report](results/correctness/README.md). `experiments/scaling.py` runs the timing jobs and saves scaling figures and a [short results report](results/scaling/README.md). Both launch experiments only through `run.py`. You can also run one job directly:

```sh
python run.py --ranks 2 --tokens 65 --epochs 30 --lr 0.001 --blocks 1
python run.py --ranks 4 --tokens 1024 --epochs 25 --warmup 5
python run.py --ranks 8 --tokens 16384 --experts 16 --expert-hidden 1024 --epochs 25 --warmup 5
```

`--tokens` is always the **total** batch size; `torch.tensor_split` divides it as evenly as possible. The target is deterministic regression to `0.3*tanh(x)+0.2`. Each epoch reuses that batch. Every run uses learned routing and Adam with its default betas and no weight decay. The default learning rate is 0.001.
`--expert-hidden` sets the hidden width of each expert MLP; its default is 64.
`--experts` sets the global expert count; its default is 8 and it must be divisible by the rank count.

## Read the code

1. `moe/model.py`: `MoEModel` builds `StandardMoEBlock` or `ParallelMoEBlock` instances. `ModelConfig.num_blocks` controls the stack depth. The parallel block uses PyTorch's autograd-aware variable-size all-to-all for activations.
2. `moe/training.py`: the training step, step reporting, shared-gradient synchronization, and one training loop that calls `model(x)` in either mode.
3. `run.py`: one generic worker that initializes the model and data shard. It seeds layers by global block and expert ID so standard and parallel runs start from the same model; the two files in `experiments/` launch it with different inputs.

## Correctness runs

`experiments/correctness.py` launches one-block jobs with **65 total tokens** at 1, 2, and 4 ranks. One-epoch jobs check one-step outputs, input gradients, every owned/shared parameter gradient and post-step value. They also check that the projection, router, and all eight routed experts get nonzero gradients and change after the step. Thirty-epoch jobs compare global loss at every epoch and final parameters. The one-rank job builds `MoEModel(parallel=False)`; the others build `MoEModel(parallel=True)`. Numerical comparisons use float64, `atol=1e-8`, and `rtol=1e-6`.

It also launches a short **two-block** job at 1, 2, and 4 ranks to check that stacked blocks preserve those equivalences. `--capture-state` captures full tensors for external comparison in temporary files; it does not execute a reference model inside parallel workers. Compact checks, parameter-update evidence, a two-token trace, and [the loss plot](results/correctness/loss.svg) are saved in `results/correctness`. Each worker initializes its model and data shard, trains, and records its result.

The 65-token one-step fixture reaches and updates all eight experts. These tests do not force an empty destination. The code supports an empty owner through graph-preserving empty expert calls and matching collectives, but that edge case is not forced by the submitted suite.

## Scaling runs

`experiments/scaling.py` measures five one-block workloads. The first two use eight experts: hidden width 64 with 256–1,024 total tokens, and width 512 with 1,024–8,192 tokens, each across 1, 2, and 4 ranks. The next three use 16 experts with hidden widths 1,024, 2,048, and 4,096, crossing 4,096, 8,192, and 16,384 tokens with 1, 2, 4, and 8 ranks. A final comparison uses two blocks at 16,384 tokens with the width-1,024 model. These grids show how rank and batch-size trends change as expert computation grows, while the separate block comparison tests the cost of adding depth. Each job runs 25 epochs, with 5 warmup epochs and 20 measured epochs. Workers use one CPU thread each; a barrier precedes timing, and an all-reduced maximum makes the slowest rank define step latency. Startup, result serialization, and correctness comparisons are excluded.

Scaling logs contain per-rank parameter bytes and per-epoch loss, latency, expert assignments, and remote assignment counts. `experiments/scaling.py` derives load imbalance, token throughput, and logical payload from those counts. The report leads with the [batch-size crossover](results/scaling/batch-crossover.svg), [rank speedup by expert width](results/scaling/rank-speedup-by-width.svg), and [parameter storage per rank](results/scaling/parameter-storage.svg); the other sweeps and [fixed-local throughput](results/scaling/fixed-local-throughput.svg) are supporting views. For `R` remote assignments, dimension `d`, element size `b`, and `W` ranks, each forward or backward activation phase moves `2Rdb` bytes across the cluster; routing IDs and count exchange add `8R + 8W(W-1)` metadata bytes per block. These are logical tensor payloads, not measured physical network traffic. Parameter storage excludes gradients, optimizer state, activations, and process memory. The ring all-reduce traffic formula, when discussed, is analytical rather than an observation of Gloo's chosen algorithm.

See the [correctness report](results/correctness/README.md) and [scaling report](results/scaling/README.md) for the measured results and their limits. This small single-host CPU experiment does not establish a GPU speedup.
