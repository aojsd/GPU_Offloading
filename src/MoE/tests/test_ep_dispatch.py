"""Test AllGather + ReduceScatter correctness.

Run: torchrun --nproc_per_node=2 tests/test_ep_dispatch.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch, torch.distributed as dist
from ep_utils import ep_allgather, ep_reducescatter


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    ep_group = dist.new_group([0, 1])

    N, H = 16, 64
    x = torch.randn(N, H, device=device, dtype=torch.bfloat16) * (rank + 1)

    # AllGather
    gathered = ep_allgather(x, ep_group, 2)
    assert gathered.shape == (2 * N, H)
    # Rank 0's data should be in [0:N], rank 1's in [N:2N]
    if rank == 0:
        assert torch.allclose(gathered[:N], x, atol=0)

    # ReduceScatter of gathered → should give back sum of both slices
    reduced = ep_reducescatter(gathered, ep_group, 2)
    assert reduced.shape == (N, H)

    # Test complementary partials (real EP pattern: each rank computes
    # a different subset of experts, zeros for the rest)
    zeros = torch.zeros(N, H, device=device, dtype=torch.bfloat16)
    if rank == 0:
        partial = torch.cat([x, zeros])   # [2N, H]: experts 0-3 only
    else:
        partial = torch.cat([zeros, x])   # [2N, H]: experts 4-7 only
    reduced2 = ep_reducescatter(partial, ep_group, 2)
    # rank 0 gets chunk 0: x_rank0 + 0 = x_rank0
    # rank 1 gets chunk 1: 0 + x_rank1 = x_rank1
    assert torch.allclose(reduced2, x, atol=0)

    # Test pre-allocated output buffers (out= parameter)
    out_gather = torch.empty(2 * N, H, device=device, dtype=torch.bfloat16)
    result_gather = ep_allgather(x, ep_group, 2, out=out_gather)
    assert result_gather.data_ptr() == out_gather.data_ptr(), \
        "ep_allgather out= must reuse the provided buffer"
    assert torch.allclose(result_gather, gathered, atol=0)

    out_reduce = torch.empty(N, H, device=device, dtype=torch.bfloat16)
    result_reduce = ep_reducescatter(gathered, ep_group, 2, out=out_reduce)
    assert result_reduce.data_ptr() == out_reduce.data_ptr(), \
        "ep_reducescatter out= must reuse the provided buffer"
    assert torch.allclose(result_reduce, reduced, atol=0)

    if rank == 0:
        print("EP dispatch/combine test PASSED")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
