"""Differentiable variable-size all-to-all for expert assignments."""
import torch
import torch.distributed as dist


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
