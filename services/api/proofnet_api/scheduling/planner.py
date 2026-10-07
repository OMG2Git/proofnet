"""ChunkPlanner interface and the P4 single-chunk planner (ARCHITECTURE 6.2).

P5 adds the weighted proportional planner behind the same interface.
"""

from dataclasses import dataclass
from typing import Any, Protocol


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


class ChunkPlanner(Protocol):
    def plan(
        self, n_rows: int, n_features: int, devices: list[dict[str, Any]]
    ) -> list[PlannedChunk]: ...


def device_score(device: dict[str, Any]) -> float:
    return float(device["benchmark"]["score_cells_per_sec"])


class SingleChunkPlanner:
    """One eligible device -> one chunk covering all training rows.

    Deterministic: the best device by (score desc, device_id asc).
    """

    def plan(
        self, n_rows: int, n_features: int, devices: list[dict[str, Any]]
    ) -> list[PlannedChunk]:
        if not devices:
            return []
        best = sorted(devices, key=lambda d: (-device_score(d), d["_id"]))[0]
        est = n_rows * n_features / device_score(best)
        return [PlannedChunk(0, 0, n_rows, best["_id"], est)]
