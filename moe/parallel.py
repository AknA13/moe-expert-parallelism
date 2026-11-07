"""Autograd-aware variable-size all-to-all and replicated-gradient sync."""
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


def sync_replicated_gradients(model):
    for parameter in model.parameters():
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
    for parameter in model.replicated_parameters():
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
