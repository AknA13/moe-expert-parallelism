# Walkthrough and questions to rehearse

The README is the presentation. Open `moe/model.py` beside it and run `python show_results.py` before starting; that displays saved evidence without a live distributed run.

## Demonstration sequence

1. Show that `MoEModel(parallel=False)` builds blocks with all eight experts, while `parallel=True` builds blocks with a rank-specific subset. Both modes use the same router and expert definitions.
2. Find token 2 in the two-rank learned-routing trace: expert 7 is remote, expert 0 is local. Follow both assignment rows through packing, expert work, return, and weighted combination.
3. Open `ParallelMoEBlock.forward`: its two autograd-aware all-to-all calls communicate activation rows and expert outputs. In backward, PyTorch reverses their split sizes to communicate gradients. Each parallel block owns that sequence.
4. Derive the global loss denominator before explaining why each block's replicated projection and router need a SUM, while owned experts do not.
5. Show `results/verify/comparison-summary.json`. Explain the first-step output/gradient comparison, the 30-epoch loss trajectory, final parameter comparison, and the two-block check.
6. Compare saved 256-token runs at 1, 2, and 4 ranks. Then show the 512- and 1,024-token jobs. Interpret storage and payload alongside latency.

## Questions with concrete answers

**Is the one-rank job a real reference?**
Yes. The one-rank job builds `MoEModel(parallel=False)` and runs `StandardMoEBlock.forward`, which executes all experts locally without token exchange. The 2- and 4-rank jobs build `MoEModel(parallel=True)` and run `ParallelMoEBlock.forward`. They share initialization, data, gate semantics, loss, and optimizer, but use distinct expert execution paths.

**How do multiple MoE blocks work?**
The model holds a `ModuleList` of blocks. Each block projects its input, routes the projected tokens, and, in the parallel model, dispatches to its own sharded experts. Shared-gradient synchronization includes every block's projection and router. The short two-block run compares both model types.

**Why does the router learn with discrete top-k?**
Selected indices are piecewise constant. The selected logits still determine differentiable softmax weights. Routing boundaries are nondifferentiable; this is ordinary sparse gating, not a straight-through estimator.

**Why not wrap the entire model in DDP?**
Different ranks hold different experts. Averaging unrelated expert gradients would be wrong. Only replicated parameters need a SUM within this expert-parallel group, and DDP's averaging would need a matching loss normalization.

**What if an owner receives nothing?**
It still enters every exchange and backward call. The zero-valued graph path and empty expert calls preserve the schedule. Learned routing does not force this case in the submitted checks, so describe it as an implementation property rather than a verified fixture.

**Does expert parallelism reduce total model parameters?**
No. It distributes expert storage. Shared parameters remain replicated; activations, gradients, optimizer state, and process overhead are additional costs.

**Why can four ranks be slower?**
The model is small and CPU workers share a host. Metadata exchange, activation communication, small expert operations, and shared-gradient synchronization can exceed compute savings. The results make no GPU speedup claim.

**What would change on GPUs?**
Profile dispatch, expert compute, return, and shared synchronization separately. Validate NCCL behavior, synchronized device timing, grouped GEMMs, and communication overlap with representative workloads.

**What does the reference fail to prove?**
The two execution paths share router and expert definitions, so a bug in those shared definitions might affect both. The suite checks observed learned routes, but does not guarantee an empty owner or update to every expert.
