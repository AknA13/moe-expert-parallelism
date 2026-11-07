"""Deterministic model initialization and synthetic regression data."""
from typing import NamedTuple

import torch
from torch import nn

from .model import ModelConfig, ParallelMoEModel, StandardMoEModel


class RankData(NamedTuple):
    model: nn.Module
    x: torch.Tensor
    target: torch.Tensor
    indices: torch.Tensor
    global_tokens: int


def make_model(rank, world, config=ModelConfig()):
    """Use global block and expert IDs so weights do not depend on rank count."""
    with torch.random.fork_rng():
        model = (StandardMoEModel(config) if world == 1
                 else ParallelMoEModel(rank, world, config))

        def reset_with_seed(module, seed):
            torch.manual_seed(seed)
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()

        for block_id, block in enumerate(model.blocks):
            offset = block_id * 1000
            reset_with_seed(block.token_projection, 10 + offset)
            reset_with_seed(block.router, 11 + offset)
            for expert_id, expert in block.experts.items():
                reset_with_seed(expert, 100 + offset + int(expert_id))
        return model.double()


def data(total_tokens, dimension):
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(total_tokens, dimension, generator=generator,
                    dtype=torch.float64)
    return x, 0.3 * x.tanh() + 0.2


def prepare_rank(rank, world, total_tokens, model_config):
    model = make_model(rank, world, model_config)
    full_x, full_target = data(total_tokens, model_config.dimension)
    indices = torch.tensor_split(torch.arange(total_tokens), world)[rank]
    return RankData(model, full_x[indices].clone(), full_target[indices].clone(),
                    indices, total_tokens)
