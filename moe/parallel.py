"""Two activation exchanges forward, their transposes backward."""
import torch
import torch.distributed as dist
from .model import E, K

class Exchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, send, receive):
        ctx.send, ctx.receive = send, receive
        out = x.new_empty((sum(receive), *x.shape[1:]))
        dist.all_to_all_single(out, x.contiguous(), receive, send)
        return out

    @staticmethod
    def backward(ctx, grad):
        out = grad.new_empty((sum(ctx.send), *grad.shape[1:]))
        dist.all_to_all_single(out, grad.contiguous(), ctx.send, ctx.receive)
        return out, None, None

def forward(model, x, expert_ids_override=None):
    world, rank = dist.get_world_size(), dist.get_rank()
    h = model.token_encoder(x)
    ids, weights = model.route(h, expert_ids_override)
    owners = ids.flatten() // (E//world)
    order = owners.argsort(stable=True)
    send_tensor = torch.bincount(owners, minlength=world)
    receive_tensor = torch.empty_like(send_tensor)
    dist.all_to_all_single(receive_tensor, send_tensor)
    send, receive = send_tensor.tolist(), receive_tensor.tolist()
    expert_ids = ids.flatten()[order].contiguous()
    received_ids = expert_ids.new_empty(sum(receive))
    dist.all_to_all_single(received_ids, expert_ids, receive, send)
    packed = h.repeat_interleave(K, dim=0)[order]
    received = Exchange.apply(packed, send, receive)
    # Keep the dispatch in the graph even when this owner receives no tokens.
    computed = received * 0
    for key, expert in model.experts.items():
        positions = (received_ids == int(key)).nonzero().flatten()
        computed = computed.index_add(0, positions, expert(received[positions]))
    returned = Exchange.apply(computed, receive, send)
    unsorted = returned[order.argsort()]
    out = (unsorted.reshape(len(x), K, -1) * weights[..., None]).sum(1)
    remote = sum(send)-send[rank]
    stats = {'assignments': torch.bincount(ids.flatten(), minlength=E).tolist(),
             'remote_assignments': remote,
             'activation_forward_bytes': remote*h.shape[1]*h.element_size()*2,
             'activation_backward_bytes': remote*h.shape[1]*h.element_size()*2,
             'metadata_bytes': remote*8 + (world-1)*8,
             'send': send, 'receive': receive}
    return out, stats, ids, unsorted.reshape(len(x), K, -1)

def sync_shared(model):
    for p in model.parameters():
        if p.grad is None: p.grad = torch.zeros_like(p)
    for p in model.shared(): dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
