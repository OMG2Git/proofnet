"""WeightedProportionalPlanner (ARCHITECTURE 6.2): determinism, proportionality, remainders,
memory splitting, bounds. Pure functions, no database."""

import random
from typing import Any

import pytest

from proofnet_api.scheduling.planner import (
    MIN_CHUNK_ROWS,
    WeightedProportionalPlanner,
    allocate_rows,
    split_equal,
)

P = WeightedProportionalPlanner()


def dev(i: str, score: float, mem_gb: float = 8.0, kind: str = "pyodide") -> dict[str, Any]:
    return {
        "_id": i,
        "name": f"phone-{i}",
        "benchmark": {"score_cells_per_sec": score},
        "runtime": {"kind": kind},
        "capabilities": {"memory_gb_reported": mem_gb},
    }


def tiles(chunks: list[Any], n: int) -> bool:
    pos = 0
    for c in sorted(chunks, key=lambda c: c.index):
        if c.row_start != pos or c.row_end <= c.row_start:
            return False
        pos = c.row_end
    return pos == n


def test_documented_example_53334_26666() -> None:
    """Phone A 3.0 M cells/s, Phone B 1.5 M cells/s, N = 80,000 (ARCH 6.2 worked example).

    Exact 53,333.33 / 26,666.67; the leftover row goes to the larger fractional part (B).
    """
    plan = P.plan(80_000, 16, [dev("B", 1.5e6), dev("A", 3.0e6)], max_devices=4)
    rows = {s.device_id: s.rows for s in plan.shares}
    assert rows == {"A": 53_333, "B": 26_667}
    assert [c.preferred_device_id for c in plan.chunks] == ["A", "B"]  # faster device first
    assert [(c.row_start, c.row_end) for c in plan.chunks] == [(0, 53_333), (53_333, 80_000)]
    assert tiles(plan.chunks, 80_000)
    # estimated seconds = rows * d / score
    assert plan.chunks[0].est_seconds == pytest.approx(53_333 * 16 / 3.0e6)
    assert plan.shares[0].weight == pytest.approx(2 / 3)


def test_deterministic_regardless_of_input_order() -> None:
    devices = [dev("c", 2e6), dev("a", 2e6), dev("b", 5e6)]
    base = P.plan(10_001, 8, devices, 4)
    for _ in range(20):
        shuffled = devices[:]
        random.shuffle(shuffled)
        again = P.plan(10_001, 8, shuffled, 4)
        assert again == base
    # ties on score are broken by device id ascending
    assert [s.device_id for s in base.shares] == ["b", "a", "c"]


def test_proportionality_and_remainder_distribution() -> None:
    for n in (1000, 1001, 7777, 99_999, 300_000):
        scores = [3.0, 1.5, 1.0, 0.7]
        rows = allocate_rows(n, scores)
        assert sum(rows) == n
        total = sum(scores)
        for r, s in zip(rows, scores, strict=True):
            assert abs(r - n * s / total) < 1.0  # within one row of the exact share
    # leftover rows go to the largest fractional parts, ties to the earlier device
    assert allocate_rows(10, [1.0, 1.0, 1.0]) == [4, 3, 3]
    assert allocate_rows(2, [1.0, 1.0, 1.0]) == [1, 1, 0]


def test_k_is_bounded_by_devices_max_devices_and_min_chunk_rows() -> None:
    devices = [dev(str(i), 1e6 * (i + 1)) for i in range(6)]
    assert len(P.plan(100_000, 4, devices, max_devices=3).shares) == 3
    assert len(P.plan(100_000, 4, devices, max_devices=8).shares) == 6
    # floor(N / MIN_CHUNK_ROWS) devices at most; always at least one
    assert len(P.plan(2 * MIN_CHUNK_ROWS + 5, 4, devices, 8).shares) == 2
    assert len(P.plan(MIN_CHUNK_ROWS - 1, 4, devices, 8).shares) == 1
    # top-k by score
    assert {s.device_id for s in P.plan(100_000, 4, devices, 2).shares} == {"5", "4"}


def test_single_device_gets_everything() -> None:
    plan = P.plan(50_000, 16, [dev("only", 1e6)], 4)
    assert [(c.row_start, c.row_end) for c in plan.chunks] == [(0, 50_000)]


def test_no_devices_no_plan() -> None:
    assert P.plan(10_000, 4, [], 4).chunks == []


def test_memory_cap_splits_a_share_into_equal_chunks_on_the_same_device() -> None:
    # 0.1 GB reported RAM -> 10 MB budget; 80k x 16 needs ~32.6 MB -> ceil(3.26) = 4 chunks
    small = dev("small", 1e6, mem_gb=0.1)
    plan = P.plan(80_000, 16, [small], 4)
    assert len(plan.chunks) == 4
    assert all(c.preferred_device_id == "small" for c in plan.chunks)
    sizes = [c.n_rows for c in plan.chunks]
    assert max(sizes) - min(sizes) <= 1 and sum(sizes) == 80_000
    assert tiles(plan.chunks, 80_000) and plan.shares[0].n_chunks == 4
    # every chunk fits the budget
    from proofnet_api.scheduling.policies import mem_budget_bytes, mem_estimate_bytes

    assert all(mem_estimate_bytes(c.n_rows, 16) <= mem_budget_bytes(small) for c in plan.chunks)


def test_memory_split_in_a_multi_device_plan_keeps_chunk_order_by_device() -> None:
    plan = P.plan(240_000, 16, [dev("big", 4e6), dev("small", 1e6, mem_gb=0.1)], 4)
    big = [c for c in plan.chunks if c.preferred_device_id == "big"]
    small = [c for c in plan.chunks if c.preferred_device_id == "small"]
    assert len(big) == 1 and len(small) > 1
    assert [c.index for c in plan.chunks] == list(range(len(plan.chunks)))
    assert tiles(plan.chunks, 240_000)
    # the big device's chunk comes first (fastest device first)
    assert plan.chunks[0].preferred_device_id == "big"


def test_extreme_score_ratio_drops_a_device_that_would_get_no_rows() -> None:
    plan = P.plan(2000, 4, [dev("fast", 1e12), dev("slow", 1.0)], 4)
    assert [s.device_id for s in plan.shares] == ["fast"]
    assert tiles(plan.chunks, 2000)


def test_split_equal() -> None:
    assert split_equal(10, 3) == [4, 3, 3]
    assert split_equal(9, 3) == [3, 3, 3]
    assert sum(split_equal(12345, 7)) == 12345


def test_random_plans_always_tile_and_conserve_rows() -> None:
    rng = random.Random(7)
    for _ in range(300):
        n = rng.randint(1, 300_000)
        k = rng.randint(1, 8)
        d = rng.randint(1, 64)
        devices = [
            dev(f"d{i}", rng.uniform(1e5, 1e8), mem_gb=rng.choice([0.5, 1, 2, 4, 8]))
            for i in range(k)
        ]
        plan = P.plan(n, d, devices, rng.randint(1, 8))
        assert plan.chunks, (n, k)
        assert tiles(plan.chunks, n)
        assert sum(c.n_rows for c in plan.chunks) == n
        assert sum(s.rows for s in plan.shares) == n
        assert all(c.n_rows >= 1 for c in plan.chunks)
        assert abs(sum(s.weight for s in plan.shares) - 1.0) < 1e-9
