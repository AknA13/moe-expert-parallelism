import time
import statistics
import torch
import torch.distributed as dist
from torch import nn
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

def verify_case(kind, steps=1, n=65):
    rank, world = dist.get_rank(), dist.get_world_size()
    model, reference = make_model(rank, world), make_model()
    full_x, target = data(n)
    full_x.requires_grad_()
    indices = torch.tensor_split(torch.arange(n), world)[rank]
    x = full_x.detach()[indices].clone().requires_grad_()
    expert_ids = routing(kind, n, world)
    local_expert_ids = None if expert_ids is None else expert_ids[indices]
    opt = torch.optim.SGD(model.parameters(), lr=.15)
    ref_opt = torch.optim.SGD(reference.parameters(), lr=.15)
    history, maximum = [], 0.
    initial = {name: p.detach().clone() for name,p in model.named_parameters()}
    for _ in range(steps):
        opt.zero_grad(set_to_none=True); ref_opt.zero_grad(set_to_none=True)
        x.grad = None; full_x.grad = None
        pred, _, _, _ = forward(model, x, local_expert_ids)
        expected = reference(full_x, expert_ids)
        maximum = max(maximum, close(pred, expected[indices]))
        local_loss = loss(pred, target[indices], n)
        ref_loss = loss(expected, target, n)
        total = local_loss.detach().clone(); dist.all_reduce(total)
        close(total, ref_loss)
        local_loss.backward(); ref_loss.backward(); sync_shared(model)
        close(x.grad, full_x.grad[indices])
        ref_params = dict(reference.named_parameters())
        for name, p in model.named_parameters():
            q = ref_params[name]
            maximum = max(maximum, close(p.grad, q.grad if q.grad is not None else torch.zeros_like(q)))
        opt.step(); ref_opt.step()
        for name, p in model.named_parameters(): maximum = max(maximum, close(p, ref_params[name]))
        for p in model.shared():
            root = p.detach().clone(); dist.broadcast(root, 0); close(p, root)
        history.append(float(total))
    if kind == 'balanced':
        assert all(not torch.equal(initial[name], p) for name,p in model.named_parameters())
    if steps > 1: assert history[-1] < history[0]
    return {'routing': kind, 'tokens': n, 'steps': steps, 'max_abs_error': maximum, 'loss': history}

def benchmark(kind, regime, warmup=5, samples=20):
    rank, world = dist.get_rank(), dist.get_world_size()
    n = 256 if regime == 'fixed_global' else 256*world
    model = make_model(rank, world)
    x, target = data(n)
    indices = torch.tensor_split(torch.arange(n), world)[rank]
    x, target = x[indices], target[indices]
    expert_ids = routing(kind, n, world)
    expert_ids = None if expert_ids is None else expert_ids[indices]
    opt = torch.optim.SGD(model.parameters(), lr=.15)
    durations, records = [], []
    for step in range(warmup+samples):
        dist.barrier(); start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        pred, stats, _, _ = forward(model, x, expert_ids)
        loss(pred, target, n).backward(); sync_shared(model); opt.step()
        elapsed = torch.tensor(time.perf_counter()-start, dtype=torch.float64)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        if step >= warmup: durations.append(float(elapsed)); records.append(stats)
    all_records = [None]*world
    dist.all_gather_object(all_records, records)
    expert_load = [sum(records[s]['assignments'][e] for records in all_records for s in range(samples))/samples for e in range(8)]
    shared = sum(p.numel()*p.element_size() for p in model.shared())
    total = sum(p.numel()*p.element_size() for p in model.parameters())
    metrics = {key: sum(r[key] for records in all_records for r in records)/samples
               for key in ('remote_assignments','activation_forward_bytes','activation_backward_bytes','metadata_bytes')}
    return {'routing':kind, 'regime':regime, 'global_tokens':n,
            'median_ms':1000*statistics.median(durations),
            'p95_ms':1000*sorted(durations)[int(.95*(samples-1))],
            'parameter_bytes_per_rank':total, 'shared_parameter_bytes':shared,
            'expert_parameter_bytes_per_rank':total-shared,
            'shared_gradient_logical_input_bytes_per_rank':shared,
            'ring_allreduce_modeled_cluster_bytes':2*(world-1)*shared,
            'expert_assignments_mean':expert_load, 'max_to_mean_load':max(expert_load)/(sum(expert_load)/8),
            'warmup':warmup,'samples':samples, **metrics}

def trace():
    rank, world = dist.get_rank(), dist.get_world_size()
    x,_ = data(8); indices = torch.tensor_split(torch.arange(8),world)[rank]
    expert_ids = routing('balanced',8,world)
    pred,stats,ids,returned = forward(make_model(rank,world),x[indices],expert_ids[indices])
    return [{'token':int(token),'source':rank,'experts':ids[i].tolist(),
             'owners':(ids[i]//(8//world)).tolist(),
             'returned_first_coordinate':returned[i,:,0].detach().tolist(),
             'combined_first_coordinate':float(pred[i,0].detach())} for i,token in enumerate(indices)]
