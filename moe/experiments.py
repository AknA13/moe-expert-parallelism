import time
import statistics
import torch
import torch.distributed as dist
from torch import nn
from typing import NamedTuple
from .model import D, E, Model
from .parallel import forward, sync_shared, Exchange


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
    if kind == 'learned': return None
    if kind == 'balanced': return torch.stack((tokens % E, (tokens + 1) % E), 1)
    if kind == 'skewed': return torch.tensor([0, 1]).expand(n, 2)
    if kind == 'local':
        owner = torch.empty(n, dtype=torch.long)
        for rank, chunk in enumerate(torch.tensor_split(tokens, world)): owner[chunk] = rank
        base = owner * (E // world)
        return torch.stack((base, base + 1), 1)
    raise ValueError(kind)


def loss(pred, target, global_n):
    return (pred - target).square().mean(-1).sum() / global_n


class RankData(NamedTuple):
    model: Model
    x: torch.Tensor
    target: torch.Tensor
    indices: torch.Tensor
    expert_ids: torch.Tensor | None
    full_x: torch.Tensor
    full_target: torch.Tensor


def prepare_rank(rank, world, token_count, route_kind):
    """Construct this rank's model and slice its portion of the global batch."""
    model = make_model(rank, world)
    full_x, full_target = data(token_count)
    indices = torch.tensor_split(torch.arange(token_count), world)[rank]
    chosen = routing(route_kind, token_count, world)
    local_ids = None if chosen is None else chosen[indices]
    return RankData(model, full_x[indices].clone(), full_target[indices],
                    indices, local_ids, full_x, full_target)

def close(a, b):
    torch.testing.assert_close(a, b, atol=1e-8, rtol=1e-6)
    return float((a-b).detach().abs().max()) if a.numel() else 0.

def communication_check():
    rank, world = dist.get_rank(), dist.get_world_size()
    send = [(rank+j) % 3 for j in range(world)]
    if world == 1: send = [2]
    counts = torch.tensor(send)
    received = torch.empty_like(counts)
    dist.all_to_all_single(received, counts)
    receive = received.tolist()
    x = (torch.arange(sum(send), dtype=torch.float64)+rank).requires_grad_()
    y = Exchange.apply(x[:, None], send, receive)
    objective = y.square().sum()/2
    objective.backward()
    close(x.grad, x.detach())  # inverse exchange of y recovers x
    # Global objective finite differences, perturbing one source at a time.
    for source in range(world):
        size = torch.tensor(len(x) if rank == source else 0)
        dist.broadcast(size, source)
        for i in range(int(size)):
            values = []
            for sign in (-1, 1):
                trial = x.detach().clone()
                if rank == source: trial[i] += sign*1e-5
                value = Exchange.apply(trial[:, None], send, receive).square().sum()/2
                dist.all_reduce(value)
                values.append(value)
            if rank == source: close((values[1]-values[0])/2e-5, x.grad[i])
    return 'inverse exchange and global finite differences passed'

def train_loop(prepared, steps, *, reference=None, full_expert_ids=None,
               warmup=0, capture_trace=False):
    """Train a prepared rank and record what happened on every step.

    An optional full-model reference checks the same step outside the timed
    distributed section. All ranks run this loop in the same order.
    """
    rank, world = dist.get_rank(), dist.get_world_size()
    model = prepared.model
    x = prepared.x.requires_grad_(reference is not None)
    full_x = prepared.full_x.requires_grad_(reference is not None)
    optimizer = torch.optim.SGD(model.parameters(), lr=.15)
    ref_optimizer = (torch.optim.SGD(reference.parameters(), lr=.15)
                     if reference is not None else None)
    logs, token_trace = [], None

    for step in range(steps):
        dist.barrier()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        x.grad = None
        prediction, stats, ids, returned = forward(model, x, prepared.expert_ids)
        local_loss = loss(prediction, prepared.target, len(full_x))
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
        if reference is not None:
            ref_optimizer.zero_grad(set_to_none=True)
            full_x.grad = None
            expected = reference(full_x, full_expert_ids)
            errors['output'] = close(prediction, expected[prepared.indices])
            ref_loss = loss(expected, prepared.full_target, len(full_x))
            errors['loss'] = close(global_loss, ref_loss)
            ref_loss.backward()
            errors['input_gradient'] = close(x.grad, full_x.grad[prepared.indices])
            ref_params = dict(reference.named_parameters())
            errors['parameter_gradient'] = 0.
            for name, parameter in model.named_parameters():
                expected_grad = ref_params[name].grad
                if expected_grad is None:
                    expected_grad = torch.zeros_like(ref_params[name])
                errors['parameter_gradient'] = max(
                    errors['parameter_gradient'], close(parameter.grad, expected_grad))
            ref_optimizer.step()
            errors['parameter_update'] = max(
                close(parameter, ref_params[name])
                for name, parameter in model.named_parameters())
            for parameter in model.shared():
                root = parameter.detach().clone()
                dist.broadcast(root, 0)
                close(parameter, root)

        if capture_trace and step == 0:
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
                     'warmup': step < warmup})
    return logs, token_trace


def verify_case(kind, steps=1, n=65, capture_trace=False):
    rank, world = dist.get_rank(), dist.get_world_size()
    prepared = prepare_rank(rank, world, n, kind)
    initial = {name: parameter.detach().clone()
               for name, parameter in prepared.model.named_parameters()}
    logs, token_trace = train_loop(
        prepared, steps, reference=make_model(),
        full_expert_ids=routing(kind, n, world), capture_trace=capture_trace)
    if kind == 'balanced':
        assert all(not torch.equal(initial[name], parameter)
                   for name, parameter in prepared.model.named_parameters())
    losses = [entry['global_loss'] for entry in logs]
    if steps > 1:
        assert losses[-1] < losses[0]
    maximum = max((value for entry in logs
                   for value in entry['reference_error'].values()), default=0.)
    result = {'routing': kind, 'tokens': n, 'steps': steps,
              'max_abs_error': maximum, 'loss': losses, 'step_logs': logs}
    if capture_trace:
        result['token_trace'] = token_trace
    return result


def benchmark(kind, regime, warmup=5, samples=20):
    rank, world = dist.get_rank(), dist.get_world_size()
    n = 256 if regime == 'fixed_global' else 256 * world
    prepared = prepare_rank(rank, world, n, kind)
    logs, _ = train_loop(prepared, warmup + samples, warmup=warmup)
    measured = logs[warmup:]
    all_records = [None] * world
    dist.all_gather_object(all_records, measured)
    durations = [entry['step_ms'] for entry in measured]
    expert_load = [
        sum(entry['routing']['assignments'][expert]
            for records in all_records for entry in records) / samples
        for expert in range(E)
    ]
    shared = sum(p.numel() * p.element_size() for p in prepared.model.shared())
    total = sum(p.numel() * p.element_size()
                for p in prepared.model.parameters())
    metrics = {
        key: sum(entry['routing'][key]
                 for records in all_records for entry in records) / samples
        for key in ('remote_assignments', 'activation_forward_bytes',
                    'activation_backward_bytes', 'metadata_bytes')
    }
    return {'routing': kind, 'regime': regime, 'global_tokens': n,
            'median_ms': statistics.median(durations),
            'p95_ms': sorted(durations)[int(.95 * (samples - 1))],
            'parameter_bytes_per_rank': total, 'shared_parameter_bytes': shared,
            'expert_parameter_bytes_per_rank': total - shared,
            'shared_gradient_logical_input_bytes_per_rank': shared,
            'ring_allreduce_modeled_cluster_bytes': 2 * (world - 1) * shared,
            'expert_assignments_mean': expert_load,
            'max_to_mean_load': max(expert_load) / (sum(expert_load) / E),
            'warmup': warmup, 'samples': samples, 'step_logs': logs, **metrics}

def run_demo(rank, world):
    """Run the complete teaching demo once on this rank."""
    def report(stage):
        if rank == 0:
            print(f'{world} ranks: {stage}', flush=True)

    communication = communication_check()
    cases = [verify_case(kind) for kind in
             ('learned', 'balanced', 'skewed', 'local')]
    report('correctness passed')

    trajectory = verify_case('learned', steps=30)
    report('training passed')

    benchmarks = [benchmark(kind, regime)
                  for regime in ('fixed_global', 'fixed_local')
                  for kind in ('learned', 'balanced', 'skewed')]
    report('scaling measured')

    trace_case = verify_case('balanced', n=8, capture_trace=True)
    tokens = trace_case.pop('token_trace')
    report('token trace recorded')
    return {'communication': communication,
            'cases': cases,
            'trajectory': trajectory,
            'benchmarks': benchmarks,
            'tokens': tokens}
