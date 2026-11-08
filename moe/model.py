"""One MoE model with local and expert-parallel block implementations."""
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_to_all_single as autograd_all_to_all_single
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    dimension: int = 32
    experts: int = 8
    top_k: int = 2
    expert_hidden: int = 64
    num_blocks: int = 1


class MoEBlock(nn.Module):
    """Token projection, router, and experts shared by both execution modes."""

    def __init__(self, config, expert_ids):
        super().__init__()
        self.config = config
        self.token_projection = nn.Sequential(
            nn.Linear(config.dimension, config.dimension), nn.Tanh())
        self.router = nn.Linear(config.dimension, config.experts)
        self.experts = nn.ModuleDict({
            str(expert_id): nn.Sequential(
                nn.Linear(config.dimension, config.expert_hidden),
                nn.Tanh(),
                nn.Linear(config.expert_hidden, config.dimension),
            )
            for expert_id in expert_ids
        })
        self.last_report = None
        self.capture_trace = False

    def route(self, x):
        hidden = self.token_projection(x)
        logits = self.router(hidden)
        ids = logits.topk(self.config.top_k, dim=-1).indices
        weights = logits.gather(1, ids).softmax(-1)
        return hidden, ids, weights

    def record(self, ids, returned, combined, remote_assignments):
        self.last_report = {
            'routing': {
                'assignments': torch.bincount(
                    ids.flatten(), minlength=self.config.experts).tolist(),
                'remote_assignments': remote_assignments,
            },
        }
        if self.capture_trace:
            self.last_report.update({
                'ids': ids.detach(),
                'returned': returned.detach(),
                'combined': combined.detach(),
            })


class StandardMoEBlock(MoEBlock):
    """All experts execute in one process; no distributed calls."""

    def __init__(self, config):
        # The local reference owns every global expert ID.
        super().__init__(config, range(config.experts))

    def forward(self, x):
        # Project tokens and choose experts and weights: (T, D), (T, K), (T, K).
        hidden, chosen_exp, exp_weights = self.route(x)

        # Repeat each hidden row once per selected expert: (T, D) -> (T*K, D).
        hidden = hidden.repeat_interleave(self.config.top_k, dim=0)

        # Run each expert on its assignment rows without changing row order.
        expert_results = torch.zeros_like(hidden)
        for expert_id, expert in self.experts.items():
            chosen_tokens = (chosen_exp.flatten() == int(expert_id)).nonzero().flatten()
            expert_results = expert_results.index_add(
                0, chosen_tokens, expert(hidden[chosen_tokens]))

        # Restore token/slot shape and combine the weighted expert results.
        expert_results = expert_results.reshape(len(x), self.config.top_k, self.config.dimension)
        output = (expert_results * exp_weights[..., None]).sum(1)

        # Save the route and outputs for the trace and measurements.
        self.record(chosen_exp, expert_results, output, 0)
        return output


class ParallelMoEBlock(MoEBlock):
    """Route activations to the ranks that own their selected experts."""

    def __init__(self, config, rank, world):
        # Give each rank a contiguous, equal-sized range of expert IDs.
        if config.experts % world:
            raise ValueError('number of experts must be divisible by world size')
        self.rank = rank
        self.world = world
        experts_per_rank = config.experts // world
        first = rank * experts_per_rank
        super().__init__(config, range(first, first + experts_per_rank))

    def forward(self, x):
        # Project tokens and choose experts and weights: (T, D), (T, K), (T, K).
        hidden, chosen_exp, exp_weights = self.route(x)

        # Group token rows by the ranks owning their chosen experts.
        expert_rank = chosen_exp.flatten() // (self.config.experts // self.world) #(N*K)
        order = expert_rank.argsort(stable=True)

        # All-to-all to learn how many token rows arrive from each rank.
        send_counts = torch.bincount(expert_rank, minlength=self.world) #(world_size)
        receive_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(receive_counts, send_counts)
        send, receive = send_counts.tolist(), receive_counts.tolist()
        #send -> (world_size) how many tokens we dispatch to each rank
        #receive -> (world_size) how many tokens we receive from each rank

        # All-to-all to send the expert id needed each token row
        dispatched_expert_ids = chosen_exp.flatten()[order].contiguous()
        received_expert_ids = dispatched_expert_ids.new_empty(sum(receive))
        dist.all_to_all_single(received_expert_ids, dispatched_expert_ids, receive, send)
        #dispatched_expert_ids -> (world_size) expert ids tensor to send to each rank
        #received_expert_ids -> (world_size) expert ids tensor to received from each rank

        # Repeat hidden rows per assignment, pack them, and dispatch to owners.
        dispatched_hidden = hidden.repeat_interleave(self.config.top_k, dim=0)
        dispatched_hidden = dispatched_hidden[order]
        received_hidden = autograd_all_to_all_single(
            dispatched_hidden.new_empty((sum(receive), *dispatched_hidden.shape[1:])),
            dispatched_hidden.contiguous(),
            output_split_sizes=receive,
            input_split_sizes=send,
        ) #(world_size), contains token activations tensor receive each rank (aligned with received_expert_ids)

        # Compute expert results for each local expert
        local_expert_results = received_hidden * 0
        for expert_id, expert in self.experts.items():
            chosen_tokens = (received_expert_ids == int(expert_id)).nonzero().flatten()
            local_expert_results = local_expert_results.index_add(
                0, chosen_tokens, expert(received_hidden[chosen_tokens]))

        # Return expert results to sources and restore token/slot order.
        expert_results = autograd_all_to_all_single(
            local_expert_results.new_empty((sum(send), *local_expert_results.shape[1:])),
            local_expert_results.contiguous(),
            output_split_sizes=send,
            input_split_sizes=receive,
        )[order.argsort()]
        #(N*K, D) contains each token dispatched's expert-processed result

        # Combine each token's weighted expert outputs on its source rank.
        expert_results = expert_results.reshape(
            len(x), self.config.top_k, self.config.dimension)
        output = (expert_results * exp_weights[..., None]).sum(1)

        # Save the route and communication counts for the trace and measurements.
        self.record(chosen_exp, expert_results, output,
                    sum(send) - send[self.rank])
        return output


class MoEModel(nn.Module):
    """Stack local or expert-parallel blocks with the same model interface."""

    def __init__(self, config=ModelConfig(), *, parallel=False, rank=0, world=1):
        super().__init__()
        self.config = config
        self.parallel = parallel
        self.blocks = nn.ModuleList(
            ParallelMoEBlock(config, rank, world) if parallel
            else StandardMoEBlock(config)
            for _ in range(config.num_blocks)
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def replicated_parameters(self):
        for block in self.blocks:
            yield from block.token_projection.parameters()
            yield from block.router.parameters()

    def reports(self):
        return [block.last_report for block in self.blocks]
