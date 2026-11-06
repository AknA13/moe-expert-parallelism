"""The reference intentionally contains no distributed operations."""
import torch
from torch import nn

D, H, E = 32, 64, 8

def seeded(seed, factory):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return factory().double()

class Model(nn.Module):
    def __init__(self, rank=0, world=1):
        super().__init__()
        self.projection = seeded(10, lambda: nn.Linear(D, D))
        self.router = seeded(11, lambda: nn.Linear(D, E))
        self.experts = nn.ModuleDict({str(i): seeded(100+i, lambda: nn.Sequential(
            nn.Linear(D, H), nn.Tanh(), nn.Linear(H, D)))
            for i in range(rank*(E//world), (rank+1)*(E//world))})

    def route(self, x, forced=None):
        h = self.projection(x).tanh()
        logits = self.router(h)
        ids = logits.topk(2, dim=-1).indices if forced is None else forced
        # Equivalent to selected full-softmax probabilities, renormalized.
        weights = logits.gather(1, ids).softmax(-1)
        return h, ids, weights

    def forward(self, x, forced=None):
        h, ids, weights = self.route(x, forced)
        out = torch.zeros_like(h)
        for key, expert in self.experts.items():
            token, slot = (ids == int(key)).nonzero(as_tuple=True)
            out = out.index_add(0, token, expert(h[token]) * weights[token, slot, None])
        return out

    def shared(self):
        return list(self.projection.parameters()) + list(self.router.parameters())

def data(n):
    g = torch.Generator().manual_seed(42)
    x = torch.randn(n, D, generator=g, dtype=torch.float64)
    return x, 0.3 * x.tanh() + 0.2

def routing(kind, n, world):
    t = torch.arange(n)
    if kind == 'learned': return None
    if kind == 'balanced': return torch.stack((t % E, (t+1) % E), 1)
    if kind == 'skewed': return torch.tensor([0, 1]).expand(n, 2)
    if kind == 'local':
        owner = torch.empty(n, dtype=torch.long)
        for rank, chunk in enumerate(torch.tensor_split(t, world)): owner[chunk] = rank
        base = owner * (E//world)
        return torch.stack((base, base+1), 1)
    raise ValueError(kind)

def loss(pred, target, global_n):
    return (pred-target).square().mean(-1).sum() / global_n
