# Walkthrough and questions to rehearse

The README is the presentation. Open `moe/parallel.py` beside it. Run `python show_results.py` before starting; this displays saved evidence without depending on a live distributed run.

## Demonstration sequence

1. Sketch two ranks, their token shards, and their distinct expert sets. Shared token encoder/router parameters are replicas, not shards.
2. Find token 3 in the two-rank trace: expert 3 is local, expert 4 is remote. Follow its two assignment rows and returned values. Contrast with token 7, which crosses in the other direction.
3. Explain why counts and expert IDs travel separately from differentiable activations. Count exchange determines buffer sizes; expert IDs select local computation.
4. Open `Exchange`: forward communicates values, backward communicates their adjoints using reversed split sizes. Autograd does not infer this from raw distributed calls.
5. Derive loss normalization before discussing gradient synchronization. One expert owner already accumulates all sources; shared replicas accumulate only their source contribution until the SUM.
6. Run `python run.py --ranks 2 --validate --output fresh-results.json`. Show saved `results/verify/2-training.json` and the scaling table. Explain what an assertion failure would invalidate.
7. End with memory savings, hot experts, and the limits of this CPU experiment. Reserve remaining time for discussion.

## Questions with concrete answers

**Why does the router learn with discrete top-k?**
Selected indices are piecewise constant. The selected logits still determine differentiable softmax weights. Routing boundaries are nondifferentiable; this is ordinary sparse gating, not a straight-through estimator. Top-1 with a renormalized single gate would have weight one and no gate gradient; top-2 avoids that issue.

**Why not wrap the entire module in DDP?**
Workers hold different experts. Averaging those unrelated parameter gradients is wrong. Shared parameters need replica synchronization; expert parameters need it only if additional replicas of the same expert exist. Even for shared parameters, DDP averaging must be reconciled with the loss denominator.

**What if an owner receives nothing?**
It still participates in all exchanges and backward. An empty differentiable path preserves collective order. Skipping work locally must not skip a collective globally.

**Does expert parallelism reduce total parameter count?**
No. It distributes expert storage; each worker still holds the shared parameters. Total cluster shared storage grows with rank count. Activations, gradients, and optimizer state are additional costs.

**Why no load-balancing loss?**
The demo isolates communication and gradient correctness. The skewed fixture makes imbalance visible. A balancing objective is a meaningful extension but changes the objective and must be implemented in both reference and distributed versions with consistent global statistics.

**Why can four ranks be slower?**
Tiny experts, process scheduling, metadata exchange, four activation exchanges across forward/backward, and four shared-parameter all-reduces impose overhead. CPU workers share a host. The model does not claim end-to-end speedup merely because expert memory is sharded.

**What would you optimize first on GPUs?**
Profile dispatch, expert execution, return, and shared synchronization separately. Then consider larger workloads, grouped GEMMs, tensor packing, bucketed shared gradients, and overlap. Avoid treating modeled payload bytes as measured link utilization.

**Can the reference conceal a shared routing bug?**
It shares initialization and gate semantics but executes experts without communication. It strongly tests dispatch, return, distributed gradients, and optimizer equivalence. The separate exchange test checks row order and backward gradients independently. It does not establish model quality or validate every conceivable routing implementation.

**How would you change the number of ranks at restart?**
Checkpoint by global expert ID rather than local module order. Redistribute experts and optimizer states to their new owners; restore shared state and the logical data/RNG position. This demo does not implement checkpointing.
