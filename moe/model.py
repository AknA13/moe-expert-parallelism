"""The neural network and its single-process reference forward pass."""

import torch
from torch import nn

D = 32  # Token dimension
E = 8   # Number of experts
K = 2   # Experts selected per token
TOTAL_EXPERT_WIDTH = 512
EXPERT_HIDDEN = TOTAL_EXPERT_WIDTH // E  # 64 units inside each expert


class Model(nn.Module):
    """A shared token encoder and router, with experts owned by this rank."""

    def __init__(self, rank=0, world=1):
        super().__init__()
        self.token_encoder = nn.Sequential(nn.Linear(D, D), nn.Tanh())
        self.router = nn.Linear(D, E)

        experts_per_rank = E // world
        first_expert = rank * experts_per_rank
        self.experts = nn.ModuleDict({
            str(expert_id): nn.Sequential(
                nn.Linear(D, EXPERT_HIDDEN),
                nn.Tanh(),
                nn.Linear(EXPERT_HIDDEN, D),
            )
            for expert_id in range(first_expert, first_expert + experts_per_rank)
        })

    def route(self, h, expert_ids_override=None):
        """Choose two experts for each encoded token.

        Diagnostic experiments may prescribe IDs; gate weights still come from
        the router's selected logits.
        """
        logits = self.router(h)
        ids = (logits.topk(K, dim=-1).indices
               if expert_ids_override is None else expert_ids_override)
        weights = logits.gather(1, ids).softmax(-1)
        return ids, weights

    def forward(self, x, expert_ids_override=None):
        """Evaluate selected experts locally as the reference computation."""
        h = self.token_encoder(x)
        ids, weights = self.route(h, expert_ids_override)
        output = torch.zeros_like(h)
        for key, expert in self.experts.items():
            token, slot = (ids == int(key)).nonzero(as_tuple=True)
            contribution = expert(h[token]) * weights[token, slot, None]
            output = output.index_add(0, token, contribution)
        return output

    def shared(self):
        return list(self.token_encoder.parameters()) + list(self.router.parameters())
