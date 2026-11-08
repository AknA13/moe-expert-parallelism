"""Training steps, reporting, and the MoE training loop."""
import time
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class TrainConfig:
    total_tokens: int = 65
    epochs: int = 30
    learning_rate: float = 0.001
    warmup: int = 0


@dataclass(frozen=True)
class StepResult:
    prediction: torch.Tensor
    local_loss: torch.Tensor


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


def train_step(model, x, target, optimizer, global_tokens):
    """Update the model once using this rank's data shard."""
    world = dist.get_world_size()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(x)
    local_loss = loss(prediction, target, global_tokens)
    local_loss.backward()
    if world > 1:
        sync_replicated_gradients(model)
    else:
        for parameter in model.parameters():
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    return StepResult(prediction, local_loss)


def record_step(model, x, step, elapsed, epoch, config, token_ids, *, capture_state):
    """Collect metrics and the first-step trace outside the measured step."""
    world = dist.get_world_size()
    global_loss = step.local_loss.detach().clone()
    if world > 1:
        dist.all_reduce(global_loss)
    reports = model.reports()
    trace, first_step = None, None
    if epoch == 0 and capture_state:
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
                 for i, token in enumerate(token_ids)]}
            for block_id, report in enumerate(reports)
        ]
        first_step = {
            'output': step.prediction.detach().tolist(),
            'input_gradient': x.grad.detach().tolist(),
            'parameter_gradient': parameter_values(model, gradients=True),
            'parameter_after_step': parameter_values(model),
        }
    log = {
        'epoch': epoch,
        'global_loss': float(global_loss),
        'step_ms': float(elapsed) * 1000,
        'blocks': [report['routing'] for report in reports],
        'warmup': epoch < config.warmup,
    }
    return log, trace, first_step


def run_training(model, x, target, train_config, *, token_ids, capture_state=False):
    """Train the initialized model on this rank's data shard."""
    world = dist.get_world_size()
    initial = ({name: p.detach().clone() for name, p in model.named_parameters()}
               if capture_state else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    steps, trace, first_step = [], None, None
    for block in model.blocks:
        block.capture_trace = capture_state

    for epoch in range(train_config.epochs):
        if world > 1:
            dist.barrier()
        start = time.perf_counter()
        step = train_step(model, x, target, optimizer, train_config.total_tokens)
        elapsed = torch.tensor(time.perf_counter() - start, dtype=torch.float64)
        if world > 1:
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)

        log, epoch_trace, epoch_first_step = record_step(
            model, x, step, elapsed, epoch, train_config, token_ids,
            capture_state=capture_state)
        steps.append(log)
        if epoch == 0:
            trace, first_step = epoch_trace, epoch_first_step
            for block in model.blocks:
                block.capture_trace = False
            if capture_state:
                # Later epochs do not need input gradients for verification.
                x = x.detach()
    changes = ({name: float((p.detach() - initial[name]).abs().max())
                for name, p in model.named_parameters()}
               if capture_state else None)
    shared_bytes = sum(p.numel() * p.element_size()
                       for p in model.replicated_parameters())
    total_bytes = sum(p.numel() * p.element_size()
                      for p in model.parameters())
    return {
        'train_config': asdict(train_config),
        'model_config': asdict(model.config),
        'optimizer': type(optimizer).__name__,
        'parallel': model.parallel,
        'epochs': steps,
        'trace': trace,
        'first_step': first_step,
        'final_parameters': parameter_values(model) if capture_state else None,
        'parameter_changes': changes,
        'parameter_bytes_per_rank': total_bytes,
        'shared_parameter_bytes_per_rank': shared_bytes,
    }
