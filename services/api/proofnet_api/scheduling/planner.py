"""ChunkPlanner interface and the weighted proportional planner (ARCHITECTURE 6.2).

Deterministic heuristic, shown to the evaluator as an explanation:

    sort devices by (benchmark score desc, device id asc)
    k = min(#devices, max_devices, floor(N / MIN_CHUNK_ROWS)), at least 1
    weights w_i = score_i / sum(score of the k devices)
    rows_i = floor(N * w_i); the leftover rows go to the largest fractional parts (ties -> order)
    if a device's share exceeds its memory budget it is split into ceil(estimate / budget)
    equal chunks (extra chunks queue on the same preferred device)
    estimated_seconds = rows * d / score_i
"""

from dataclasses import dataclass
from typing import Any, Protocol

from .policies import mem_budget_bytes, mem_estimate_bytes

MIN_CHUNK_ROWS = 1000


@dataclass(frozen=True)
class PlannedChunk:
    index: int
    row_start: int
    row_end: int
    preferred_device_id: str
    est_seconds: float

    @property
    def n_rows(self) -> int:
        return self.row_end - self.row_start


@dataclass(frozen=True)
class DeviceShare:
    device_id: str
    score: float
    weight: float
    rows: int
    n_chunks: int
    est_seconds: float  # for the whole share


@dataclass(frozen=True)
class Plan:
    chunks: list[PlannedChunk]
    shares: list[DeviceShare]


class ChunkPlanner(Protocol):
    def plan(
        self, n_rows: int, n_features: int, devices: list[dict[str, Any]], max_devices: int
    ) -> Plan: ...


def device_score(device: dict[str, Any]) -> float:
    return float(device["benchmark"]["score_cells_per_sec"])


def split_equal(total: int, parts: int) -> list[int]:
    """`total` rows into `parts` pieces as equal as possible (earlier pieces get the extra rows)."""
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def allocate_rows(n_rows: int, scores: list[float]) -> list[int]:
    """floor(N * w_i), then leftover rows by largest fractional part (ties -> earlier device)."""
    total = sum(scores)
    exact = [n_rows * s / total for s in scores]
    rows = [int(e) for e in exact]
    leftover = n_rows - sum(rows)
    order = sorted(range(len(scores)), key=lambda i: (-(exact[i] - rows[i]), i))
    for i in order[:leftover]:
        rows[i] += 1
    return rows


class WeightedProportionalPlanner:
    def plan(
        self, n_rows: int, n_features: int, devices: list[dict[str, Any]], max_devices: int
    ) -> Plan:
        if not devices or n_rows <= 0:
            return Plan([], [])
        ordered = sorted(devices, key=lambda d: (-device_score(d), d["_id"]))
        k = max(1, min(len(ordered), max_devices, n_rows // MIN_CHUNK_ROWS))
        chosen = ordered[:k]
        rows = allocate_rows(n_rows, [device_score(d) for d in chosen])
        # A device that would get no rows is dropped and the plan recomputed (extreme score ratios).
        while len(chosen) > 1 and min(rows) == 0:
            chosen = [d for d, r in zip(chosen, rows, strict=True) if r > 0]
            rows = allocate_rows(n_rows, [device_score(d) for d in chosen])

        total_score = sum(device_score(d) for d in chosen)
        chunks: list[PlannedChunk] = []
        shares: list[DeviceShare] = []
        cursor = 0
        for dev, share_rows in zip(chosen, rows, strict=True):
            score = device_score(dev)
            pieces = max(
                1,
                -(-mem_estimate_bytes(share_rows, n_features) // mem_budget_bytes(dev)),
            )  # ceil
            pieces = min(pieces, share_rows)  # never an empty chunk
            for part in split_equal(share_rows, pieces):
                chunks.append(
                    PlannedChunk(
                        index=len(chunks),
                        row_start=cursor,
                        row_end=cursor + part,
                        preferred_device_id=dev["_id"],
                        est_seconds=part * n_features / score,
                    )
                )
                cursor += part
            shares.append(
                DeviceShare(
                    device_id=dev["_id"],
                    score=score,
                    weight=score / total_score,
                    rows=share_rows,
                    n_chunks=pieces,
                    est_seconds=share_rows * n_features / score,
                )
            )
        return Plan(chunks, shares)
