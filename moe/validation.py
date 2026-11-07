"""Compare a distributed training step with the full-model reference."""
import torch
import torch.distributed as dist


def close(actual, expected):
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-6)
    return float((actual - expected).detach().abs().max()) if actual.numel() else 0.


class ReferenceValidator:
    def __init__(self, reference, full_x, target, expert_ids, learning_rate, loss_fn):
        self.reference = reference
        self.full_x = full_x.requires_grad_()
        self.target = target
        self.expert_ids = expert_ids
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.SGD(reference.parameters(), lr=learning_rate)

    def check(self, model, prepared, x, prediction, global_loss):
        """Check forward, backward, and update after the timed distributed step."""
        self.optimizer.zero_grad(set_to_none=True)
        self.full_x.grad = None
        expected = self.reference(self.full_x, self.expert_ids)
        errors = {'output': close(prediction, expected[prepared.indices])}
        ref_loss = self.loss_fn(expected, self.target, len(self.full_x))
        errors['loss'] = close(global_loss, ref_loss)
        ref_loss.backward()
        errors['input_gradient'] = close(x.grad, self.full_x.grad[prepared.indices])
        ref_params = dict(self.reference.named_parameters())
        errors['parameter_gradient'] = 0.
        for name, parameter in model.named_parameters():
            expected_grad = ref_params[name].grad
            if expected_grad is None:
                expected_grad = torch.zeros_like(ref_params[name])
            errors['parameter_gradient'] = max(
                errors['parameter_gradient'], close(parameter.grad, expected_grad))
        self.optimizer.step()
        errors['parameter_update'] = max(
            close(parameter, ref_params[name])
            for name, parameter in model.named_parameters())
        for parameter in model.shared():
            root = parameter.detach().clone()
            dist.broadcast(root, 0)
            close(parameter, root)
        return errors
