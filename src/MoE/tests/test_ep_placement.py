"""Test expert placement maps.

Run: python tests/test_ep_placement.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ep_utils import determine_expert_map, local_expert_ids


def test_linear_even():
    for rank in range(2):
        n, emap = determine_expert_map(2, rank, 8, "linear")
        assert n == 4
        owned = [i for i in range(8) if emap[i] >= 0]
        assert owned == list(range(rank * 4, rank * 4 + 4))
        # Verify local slot values are [0, 1, 2, 3]
        slots = [emap[i].item() for i in owned]
        assert slots == [0, 1, 2, 3], f"rank {rank}: slots={slots}"


def test_round_robin():
    for rank in range(2):
        n, emap = determine_expert_map(2, rank, 8, "round_robin")
        assert n == 4
        owned = [i for i in range(8) if emap[i] >= 0]
        assert owned == list(range(rank, 8, 2))
        # Verify local slot values are [0, 1, 2, 3]
        slots = [emap[i].item() for i in owned]
        assert slots == [0, 1, 2, 3], f"rank {rank}: slots={slots}"


def test_ep1():
    n, emap = determine_expert_map(1, 0, 8)
    assert n == 8 and emap is None


def test_uneven_division():
    """E=7, EP=2: rank 0 gets 4 experts, rank 1 gets 3."""
    n0, emap0 = determine_expert_map(2, 0, 7, "linear")
    n1, emap1 = determine_expert_map(2, 1, 7, "linear")
    assert n0 == 4, f"rank 0: n={n0}"
    assert n1 == 3, f"rank 1: n={n1}"
    owned0 = [i for i in range(7) if emap0[i] >= 0]
    owned1 = [i for i in range(7) if emap1[i] >= 0]
    assert owned0 == [0, 1, 2, 3]
    assert owned1 == [4, 5, 6]
    # No overlap, complete coverage
    assert set(owned0) & set(owned1) == set()
    assert sorted(owned0 + owned1) == list(range(7))
    # Verify slot values
    slots0 = [emap0[i].item() for i in owned0]
    slots1 = [emap1[i].item() for i in owned1]
    assert slots0 == [0, 1, 2, 3]
    assert slots1 == [0, 1, 2]


def test_local_expert_ids_consistent():
    """local_expert_ids must match determine_expert_map."""
    for ep_size in [2, 3, 4]:
        for num_experts in [8, 7, 64]:
            for strategy in ["linear", "round_robin"]:
                for rank in range(ep_size):
                    ids = local_expert_ids(ep_size, rank, num_experts, strategy)
                    n, emap = determine_expert_map(ep_size, rank, num_experts,
                                                    strategy)
                    assert len(ids) == n
                    owned = [i for i in range(num_experts) if emap[i] >= 0]
                    assert ids == owned, (
                        f"ep={ep_size}, rank={rank}, E={num_experts}, "
                        f"strategy={strategy}: {ids} != {owned}")


if __name__ == "__main__":
    test_linear_even()
    test_round_robin()
    test_ep1()
    test_uneven_division()
    test_local_expert_ids_consistent()
    print("All placement tests passed.")
