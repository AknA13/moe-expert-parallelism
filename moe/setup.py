from typing import NamedTuple

import torch
from torch import nn

from .model import D, E, Model


def make_model(rank=0, world=1):
    """Give each shared layer and global expert ID reproducible weights."""
    with torch.random.fork_rng():
        model = Model(rank, world)

        def reset_with_seed(module, seed):
            torch.manual_seed(seed)
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()

        reset_with_seed(model.token_encoder, 10)
        reset_with_seed(model.router, 11)
        for expert_id, expert in model.experts.items():
            reset_with_seed(expert, 100 + int(expert_id))
        return model.double()


def data(n):
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(n, D, generator=generator, dtype=torch.float64)
    return x, 0.3 * x.tanh() + 0.2


def routing(kind, n, world):
    tokens = torch.arange(n)
    if kind == 'learned':
        return None
    if kind == 'balanced':
        return torch.stack((tokens % E, (tokens + 1) % E), 1)
    if kind == 'skewed':
        return torch.tensor([0, 1]).expand(n, 2)
    if kind == 'local':
        owner = torch.empty(n, dtype=torch.long)
        for rank, chunk in enumerate(torch.tensor_split(tokens, world)):
            owner[chunk] = rank
        base = owner * (E // world)
        return torch.stack((base, base + 1), 1)
    raise ValueError(kind)


class RankData(NamedTuple):
    model: Model
    x: torch.Tensor
    target: torch.Tensor
    indices: torch.Tensor
    expert_ids: torch.Tensor | None
    global_tokens: int


def prepare_rank(rank, world, token_count, route_kind):
    """Construct this rank's model and slice its portion of the global batch."""
    model = make_model(rank, world)
    full_x, full_target = data(token_count)
    indices = torch.tensor_split(torch.arange(token_count), world)[rank]
    chosen = routing(route_kind, token_count, world)
    local_ids = None if chosen is None else chosen[indices]
    return RankData(model, full_x[indices].clone(), full_target[indices].clone(),
                    indices, local_ids, token_count)


