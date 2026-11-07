"""One distributed training job, configured by its data and optimizer settings."""
import time
from dataclasses import asdict, dataclass
import torch
import torch.distributed as dist
from .model import E
from .parallel import forward, sync_shared
from .setup import data, make_model, prepare_rank, routing
from .validation import ReferenceValidator


@dataclass(frozen=True)
class TrainConfig:
    tokens: int = 65
    epochs: int = 30
    learning_rate: float = 0.15
    routing: str = 'learned'
    warmup: int = 0
    tokens_per_rank: bool = False
    validate: bool = False


def loss(pred, target, global_n):
    return (pred - target).square().mean(-1).sum() / global_n


def train_loop(prepared, config, *, validator=None):
    """Train a prepared rank and record what happened on every step.

    An optional validator checks the same step outside the timed
    distributed section. All ranks run this loop in the same order.
    """
    rank, world = dist.get_rank(), dist.get_world_size()
    model = prepared.model
    x = prepared.x.requires_grad_(validator is not None)
    global_tokens = prepared.global_tokens
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    logs, token_trace = [], None

    for step in range(config.epochs):
        dist.barrier()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        x.grad = None
        prediction, stats, ids, returned = forward(model, x, prepared.expert_ids)
        local_loss = loss(prediction, prepared.target, global_tokens)
        local_loss.backward()
        sync_shared(model)
        shared_grad_norm = sum(float(p.grad.square().sum()) for p in model.shared()) ** .5
        expert_grad_norm = sum(float(p.grad.square().sum())
                               for expert in model.experts.values()
                               for p in expert.parameters()) ** .5
        optimizer.step()
        elapsed = torch.tensor(time.perf_counter() - start, dtype=torch.float64)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)

        global_loss = local_loss.detach().clone()
        dist.all_reduce(global_loss)
        errors = {}
        if validator is not None:
            errors = validator.check(model, prepared, x, prediction, global_loss)

        if step == 0:
            token_trace = [
                {'token': int(token), 'source': rank,
                 'experts': ids[i].tolist(),
                 'owners': (ids[i] // (E // world)).tolist(),
                 'returned_first_coordinate': returned[i, :, 0].detach().tolist(),
                 'combined_first_coordinate': float(prediction[i, 0].detach())}
                for i, token in enumerate(prepared.indices)
            ]

        logs.append({'step': step, 'global_loss': float(global_loss),
                     'step_ms': 1000 * float(elapsed),
                     'shared_grad_norm': shared_grad_norm,
                     'expert_grad_norm': expert_grad_norm,
                     'routing': stats, 'reference_error': errors,
                     'warmup': step < config.warmup})
    return logs, token_trace


def run_training(rank, world, config):
    """Prepare the data and model, run training, and return observations."""
    token_count = config.tokens * world if config.tokens_per_rank else config.tokens
    prepared = prepare_rank(rank, world, token_count, config.routing)
    model = prepared.model
    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    validator = None
    if config.validate:
        full_x, full_target = data(token_count)
        validator = ReferenceValidator(
            make_model(), full_x, full_target,
            routing(config.routing, token_count, world), config.learning_rate, loss)
    steps, trace = train_loop(prepared, config, validator=validator)
    parameter_changes = {
        name: float((p.detach() - initial[name]).abs().max())
        for name, p in model.named_parameters()
    }
    shared_bytes = sum(p.numel() * p.element_size() for p in model.shared())
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return {'config': asdict(config), 'steps': steps, 'trace': trace,
            'parameter_changes': parameter_changes,
            'parameter_bytes_per_rank': total_bytes,
            'shared_parameter_bytes_per_rank': shared_bytes}

