"""Deterministic data, model setup, and one MoE training loop."""
import time
from dataclasses import asdict, dataclass
from typing import NamedTuple

import torch
import torch.distributed as dist
from torch import nn

from .model import ModelConfig, MoEModel


@dataclass(frozen=True)
class TrainConfig:
    total_tokens: int = 65
    epochs: int = 30
    learning_rate: float = 0.15
    warmup: int = 0


class RankData(NamedTuple):
    model: MoEModel
    x: torch.Tensor
    target: torch.Tensor
    indices: torch.Tensor
    global_tokens: int


def make_model(rank, world, config=ModelConfig()):
    """Use global block and expert IDs so weights do not depend on rank count."""
    with torch.random.fork_rng():
        model = MoEModel(config, parallel=world > 1, rank=rank, world=world)

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


def synthetic_data(total_tokens, dimension):
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(total_tokens, dimension, generator=generator,
                    dtype=torch.float64)
    return x, 0.3 * x.tanh() + 0.2


def prepare_rank(rank, world, total_tokens, model_config):
    model = make_model(rank, world, model_config)
    full_x, full_target = synthetic_data(total_tokens, model_config.dimension)
    indices = torch.tensor_split(torch.arange(total_tokens), world)[rank]
    return RankData(model, full_x[indices].clone(), full_target[indices].clone(),
                    indices, total_tokens)


def sync_replicated_gradients(model):
    for parameter in model.parameters():
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
    for parameter in model.replicated_parameters():
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)


def loss(prediction, target, global_tokens):
    return (prediction - target).square().mean(-1).sum() / global_tokens


def parameter_values(model, *, gradients=False):
    return {
        name: (parameter.grad if gradients else parameter).detach().tolist()
        for name, parameter in model.named_parameters()
    }


def train_loop(prepared, config, *, capture_state=False):
    """Train one data shard; the model chooses local or parallel execution."""
    model = prepared.model
    x = prepared.x.requires_grad_(capture_state)
    world = dist.get_world_size()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    logs, trace, first_step = [], None, None

    for epoch in range(config.epochs):
        if world > 1:
            dist.barrier()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        x.grad = None
        prediction = model(x)
        local_loss = loss(prediction, prepared.target, prepared.global_tokens)
        local_loss.backward()
        if world > 1:
            sync_replicated_gradients(model)
        else:
            for parameter in model.parameters():
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
        shared_norm = sum(float(p.grad.square().sum())
                          for p in model.replicated_parameters()) ** 0.5
        expert_norm = sum(float(p.grad.square().sum())
                          for block in model.blocks
                          for expert in block.experts.values()
                          for p in expert.parameters()) ** 0.5
        optimizer.step()
        elapsed = torch.tensor(time.perf_counter() - start, dtype=torch.float64)
        if world > 1:
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)

        global_loss = local_loss.detach().clone()
        if world > 1:
            dist.all_reduce(global_loss)
        reports = model.reports()
        if epoch == 0:
            trace = [
                {'block': block_id,
                 'tokens': [
                     {'token': int(token), 'source': dist.get_rank(),
                      'experts': report['ids'][i].tolist(),
                      'owners': (report['ids'][i] //
                                 (model.config.experts // world)).tolist(),
                      'returned_first_coordinate':
                          report['returned'][i, :, 0].tolist(),
                      'combined_first_coordinate':
                          float(report['combined'][i, 0])}
                     for i, token in enumerate(prepared.indices)]}
                for block_id, report in enumerate(reports)
            ]
            if capture_state:
                first_step = {
                    'output': prediction.detach().tolist(),
                    'input_gradient': x.grad.detach().tolist(),
                    'parameter_gradient': parameter_values(model, gradients=True),
                    'parameter_after_step': parameter_values(model),
                }
        logs.append({
            'epoch': epoch,
            'global_loss': float(global_loss),
            'step_ms': float(elapsed) * 1000,
            'shared_grad_norm': shared_norm,
            'expert_grad_norm': expert_norm,
            'blocks': [report['routing'] for report in reports],
            'warmup': epoch < config.warmup,
        })
    return logs, trace, first_step


def run_training(rank, world, train_config, model_config=ModelConfig(),
                 *, capture_state=False):
    prepared = prepare_rank(rank, world, train_config.total_tokens, model_config)
    model = prepared.model
    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    steps, trace, first_step = train_loop(
        prepared, train_config, capture_state=capture_state)
    changes = {name: float((p.detach() - initial[name]).abs().max())
               for name, p in model.named_parameters()}
    shared_bytes = sum(p.numel() * p.element_size()
                       for p in model.replicated_parameters())
    total_bytes = sum(p.numel() * p.element_size()
                      for p in model.parameters())
    return {
        'train_config': asdict(train_config),
        'model_config': asdict(model_config),
        'parallel': model.parallel,
        'epochs': steps,
        'trace': trace,
        'first_step': first_step,
        'final_parameters': parameter_values(model) if capture_state else None,
        'parameter_changes': changes,
        'parameter_bytes_per_rank': total_bytes,
        'shared_parameter_bytes_per_rank': shared_bytes,
    }
