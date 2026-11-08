#!/usr/bin/env python3
"""Train one standard or expert-parallel MoE on synthetic regression data."""
import argparse
import datetime
import json
import platform
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from moe.model import ModelConfig, MoEModel
from moe.training import TrainConfig, run_training


def build_model(model_config, rank, world):
    """Initialize the same global layers and experts at every world size."""
    with torch.random.fork_rng():
        model = MoEModel(model_config, parallel=world > 1, rank=rank, world=world)

        def reset_with_seed(module, seed):
            torch.manual_seed(seed)
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()

        # Seed by global block/expert ID so all rank counts start identically.
        for block_id, block in enumerate(model.blocks):
            offset = block_id * 1000
            reset_with_seed(block.token_projection, 10 + offset)
            reset_with_seed(block.router, 11 + offset)
            for expert_id, expert in block.experts.items():
                reset_with_seed(expert, 100 + offset + int(expert_id))
    return model.double()


def make_training_data(train_config, model_config, rank, world, capture_state):
    """Generate one global batch, then take this rank's token shard."""
    generator = torch.Generator().manual_seed(42)
    full_x = torch.randn(train_config.total_tokens, model_config.dimension,
                         generator=generator, dtype=torch.float64)
    full_target = 0.3 * full_x.tanh() + 0.2
    token_ids = torch.tensor_split(
        torch.arange(train_config.total_tokens), world)[rank]
    x = full_x[token_ids].clone().requires_grad_(capture_state)
    target = full_target[token_ids].clone()
    return x, target, token_ids


def worker(rank, world, init_url, train_config, model_config, capture_state, output):
    """Run one training job on this rank and collect the rank logs."""
    torch.set_num_threads(1)
    dist.init_process_group(
        'gloo', init_method=init_url, rank=rank, world_size=world,
        timeout=datetime.timedelta(seconds=90),
    )
    try:
        model = build_model(model_config, rank, world)
        x, target, token_ids = make_training_data(
            train_config, model_config, rank, world, capture_state)

        result = {'rank': rank, **run_training(
            model, x, target, train_config, token_ids=token_ids,
            capture_state=capture_state)}
        gathered = [None] * world
        dist.all_gather_object(gathered, result)
        if rank == 0:
            Path(output).write_text(json.dumps(gathered, indent=2) + '\n')
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ranks', type=int, choices=(1, 2, 4), default=2)
    parser.add_argument('--tokens', type=int, default=65,
                        help='total tokens across all ranks')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=TrainConfig().learning_rate)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--blocks', type=int, default=1)
    parser.add_argument('--capture-state', action='store_true',
                        help='save first-step tensors and final parameters for verification')
    parser.add_argument('--output', type=Path, default=Path('results/training.json'))
    args = parser.parse_args()
    if args.tokens < args.ranks or args.epochs < 1 or args.lr <= 0:
        parser.error('tokens must cover every rank; epochs and learning rate must be positive')
    if args.warmup < 0 or args.warmup >= args.epochs:
        parser.error('warmup must be between zero and epochs - 1')
    if args.blocks < 1:
        parser.error('blocks must be positive')

    train_config = TrainConfig(args.tokens, args.epochs, args.lr, args.warmup)
    model_config = ModelConfig(num_blocks=args.blocks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    environment = {
        'python': platform.python_version(), 'torch': torch.__version__,
        'platform': platform.platform(), 'machine': platform.machine(),
        'threads_per_worker': 1, 'dtype': 'float64', 'backend': 'gloo',
        'transport': 'variable-split all_to_all_single',
        'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    args.output.with_suffix('.environment.json').write_text(
        json.dumps(environment, indent=2) + '\n')
    with tempfile.TemporaryDirectory() as directory:
        init_url = Path(directory, 'process-group').as_uri()
        mp.spawn(worker, args=(args.ranks, init_url, train_config, model_config,
                               args.capture_state, str(args.output.resolve())),
                 nprocs=args.ranks, join=True)
    print(f'PASS {args.ranks} ranks, {args.epochs} epochs, '
          f'{args.blocks} blocks: {args.output}', flush=True)


if __name__ == '__main__':
    main()
