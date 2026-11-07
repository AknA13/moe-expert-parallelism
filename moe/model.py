"""The reference intentionally contains no distributed operations."""
import torch
from torch import nn

D, H, E = 32, 64, 8

class Model(nn.Module):
    """A shared token encoder, a top-2 router, and locally owned experts."""

    def __init__(self, rank=0, world=1):
        super().__init__()
        self.token_encoder = nn.Sequential(nn.Linear(D, D), nn.Tanh())
        self.router = nn.Linear(D, E)

        experts_per_rank = E // world
        first_expert = rank * experts_per_rank
        self.experts = nn.ModuleDict({
            str(expert_id): nn.Sequential(
                nn.Linear(D, H),
                nn.Tanh(),
                nn.Linear(H, D),
            )
            for expert_id in range(first_expert, first_expert + experts_per_rank)
        })

    def route(self, x, expert_ids_override=None):
        """Route normally, or use prescribed expert IDs in diagnostic tests."""
        h = self.token_encoder(x)
        logits = self.router(h)
        if expert_ids_override is None:
            ids = logits.topk(2, dim=-1).indices
        else:
            ids = expert_ids_override
        # Equivalent to selected full-softmax probabilities, renormalized.
        weights = logits.gather(1, ids).softmax(-1)
        return h, ids, weights

    def forward(self, x, expert_ids_override=None):
        h, ids, weights = self.route(x, expert_ids_override)
        out = torch.zeros_like(h)
        for key, expert in self.experts.items():
            token, slot = (ids == int(key)).nonzero(as_tuple=True)
            out = out.index_add(0, token, expert(h[token]) * weights[token, slot, None])
        return out

    def shared(self):
        return list(self.token_encoder.parameters()) + list(self.router.parameters())

def make_model(rank=0, world=1):
    """Create identical shared layers and stable expert weights across ranks.

    The reference and distributed runs need the same initial parameters. The
    seed for an expert depends on its global ID, never its owner's rank.
    fork_rng preserves the caller's random state.
    """
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
