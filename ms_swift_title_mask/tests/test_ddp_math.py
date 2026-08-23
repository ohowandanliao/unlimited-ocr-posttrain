import importlib.util
import unittest

from ms_swift_title_mask.core import expected_ddp_gradient


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _ddp_worker(rank, init_file, queue):
    import torch
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(1, 1, bias=False))
    weight = model.module.weight
    coefficients = torch.tensor([1.0, 2.0]) if rank == 0 else torch.tensor([3.0, 4.0])
    local_denom = torch.tensor(2.0)
    global_denom = local_denom.clone()
    dist.all_reduce(global_denom)
    loss = (weight.reshape(()) * coefficients).sum() * dist.get_world_size() / global_denom
    loss.backward()
    queue.put(float(weight.grad.item()))
    dist.destroy_process_group()


class DDPReferenceMathTest(unittest.TestCase):

    def test_world_size_compensation_reference(self):
        local_gradient_sums = [6.0, 10.0]
        global_active_weights = 8.0
        self.assertEqual(expected_ddp_gradient(local_gradient_sums, global_active_weights), 2.0)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is supplied by the server ms-swift environment")
    def test_two_process_ddp_gradient_matches_global_active_mean(self):
        import os
        import tempfile

        import torch.multiprocessing as mp

        ctx = mp.get_context("spawn")
        queue = ctx.SimpleQueue()
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            init_file = handle.name
        try:
            mp.spawn(_ddp_worker, args=(init_file, queue), nprocs=2, join=True)
            gradients = [queue.get(), queue.get()]
        finally:
            if os.path.exists(init_file):
                os.unlink(init_file)
        self.assertEqual(gradients, [2.5, 2.5])


if __name__ == "__main__":
    unittest.main()
