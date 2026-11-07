"""Equivalent local and expert-parallel MoE models."""
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn

from .parallel import Exchange


@dataclass(frozen=True)
class ModelConfig:
    dimension: int = 32
    experts: int = 8
    top_k: int = 2
    expert_hidden: int = 64
    num_blocks: int = 1


class MoEBlock(nn.Module):
    """Router and expert definitions shared by both execution modes."""

    def __init__(self, config, expert_ids):
        super().__init__()
        self.config = config
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

    def route(self, x):
        logits = self.router(x)
        ids = logits.topk(self.config.top_k, dim=-1).indices
        weights = logits.gather(1, ids).softmax(-1)
        return ids, weights

    def record(self, ids, returned, combined, send, receive, rank, world):
        remote = sum(send) - send[rank]
        row_bytes = returned.element_size() * self.config.dimension
        self.last_report = {
            'ids': ids.detach(),
            'returned': returned.detach(),
            'combined': combined.detach(),
            'routing': {
                'assignments': torch.bincount(
                    ids.flatten(), minlength=self.config.experts).tolist(),
                'remote_assignments': remote,
                'activation_forward_bytes': 2 * remote * row_bytes,
                'activation_backward_bytes': 2 * remote * row_bytes,
                'metadata_bytes': 8 * remote + 8 * (world - 1),
                'send': send,
                'receive': receive,
            },
        }


class StandardMoEBlock(MoEBlock):
    """All experts execute in one process; no distributed calls."""

    def __init__(self, config):
        super().__init__(config, range(config.experts))

    def forward(self, x):
        ids, weights = self.route(x)
        rows = x.repeat_interleave(self.config.top_k, dim=0)
        returned = torch.zeros_like(rows)
        for expert_id, expert in self.experts.items():
            positions = (ids.flatten() == int(expert_id)).nonzero().flatten()
            returned = returned.index_add(0, positions, expert(rows[positions]))
        returned = returned.reshape(len(x), self.config.top_k, self.config.dimension)
        combined = (returned * weights[..., None]).sum(1)
        self.record(ids, returned, combined, [len(rows)], [len(rows)], 0, 1)
        return combined


class ParallelMoEBlock(MoEBlock):
    """Route activations to the ranks that own their selected experts."""

    def __init__(self, config, rank, world):
        if config.experts % world:
            raise ValueError('number of experts must be divisible by world size')
        self.rank = rank
        self.world = world
        experts_per_rank = config.experts // world
        first = rank * experts_per_rank
        super().__init__(config, range(first, first + experts_per_rank))

    def forward(self, x):
        ids, weights = self.route(x)
        owners = ids.flatten() // (self.config.experts // self.world)
        order = owners.argsort(stable=True)
        send_counts = torch.bincount(owners, minlength=self.world)
        receive_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(receive_counts, send_counts)
        send, receive = send_counts.tolist(), receive_counts.tolist()

        packed_ids = ids.flatten()[order].contiguous()
        received_ids = packed_ids.new_empty(sum(receive))
        dist.all_to_all_single(received_ids, packed_ids, receive, send)

        packed = x.repeat_interleave(self.config.top_k, dim=0)[order]
        received = Exchange.apply(packed, send, receive)
        computed = received * 0  # Keep the backward exchange for empty owners.
        for expert_id, expert in self.experts.items():
            positions = (received_ids == int(expert_id)).nonzero().flatten()
            computed = computed.index_add(0, positions, expert(received[positions]))

        returned = Exchange.apply(computed, receive, send)[order.argsort()]
        returned = returned.reshape(len(x), self.config.top_k, self.config.dimension)
        combined = (returned * weights[..., None]).sum(1)
        self.record(ids, returned, combined, send, receive, self.rank, self.world)
        return combined


class _MoEModel(nn.Module):
    def __init__(self, config, blocks):
        super().__init__()
        self.config = config
        self.token_encoder = nn.Sequential(
            nn.Linear(config.dimension, config.dimension), nn.Tanh())
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        x = self.token_encoder(x)
        for block in self.blocks:
            x = block(x)
        return x

    def replicated_parameters(self):
        yield from self.token_encoder.parameters()
        for block in self.blocks:
            yield from block.router.parameters()

    def reports(self):
        return [block.last_report for block in self.blocks]


class StandardMoEModel(_MoEModel):
    """Full MoE on one process; the independent correctness baseline."""

    def __init__(self, config=ModelConfig()):
        super().__init__(config, [StandardMoEBlock(config)
                                  for _ in range(config.num_blocks)])


class ParallelMoEModel(_MoEModel):
    """Same MoE with each block's experts sharded across ranks."""

    def __init__(self, rank, world, config=ModelConfig()):
        super().__init__(config, [ParallelMoEBlock(config, rank, world)
                                  for _ in range(config.num_blocks)])
