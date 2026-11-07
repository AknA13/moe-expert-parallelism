"""Independently check split all-to-all row order and its gradient.

Run with: python tests/check_exchange.py
"""
import tempfile
import sys
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from moe.parallel import Exchange


def check():
    rank, world = dist.get_rank(), dist.get_world_size()
    send = [(rank + destination) % 3 for destination in range(world)]
    counts = torch.tensor(send)
    received_counts = torch.empty_like(counts)
    dist.all_to_all_single(received_counts, counts)
    receive = received_counts.tolist()

    x = (100 * rank + torch.arange(sum(send), dtype=torch.float64)).requires_grad_()
    y = Exchange.apply(x[:, None], send, receive).flatten()

    source_chunks = [None] * world
    dist.all_gather_object(source_chunks, list(x.detach().split(send)))
    expected = torch.cat([source_chunks[source][rank]
                          for source in range(world)])
    torch.testing.assert_close(y, expected)

    weights = torch.arange(len(y), dtype=torch.float64) + 10 * rank
    (y * weights).sum().backward()
    destination_weights = [None] * world
    dist.all_gather_object(destination_weights, list(weights.split(receive)))
    expected_grad = torch.cat([destination_weights[destination][rank]
                               for destination in range(world)])
    torch.testing.assert_close(x.grad, expected_grad)


def worker(rank, world, init_url):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=init_url,
                            rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=90))
    try:
        check()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    for world in (1, 2, 4):
        with tempfile.TemporaryDirectory() as folder:
            init_url = Path(folder, 'process-group').as_uri()
            mp.spawn(worker, args=(world, init_url), nprocs=world, join=True)
        print(f'PASS Exchange forward and backward, {world} ranks')
