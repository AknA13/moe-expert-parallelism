"""One training loop for standard and expert-parallel MoE models."""
import time
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist

from .model import ModelConfig
from .parallel import sync_replicated_gradients
from .setup import prepare_rank


@dataclass(frozen=True)
class TrainConfig:
    total_tokens: int = 65
    epochs: int = 30
    learning_rate: float = 0.15
    warmup: int = 0


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
