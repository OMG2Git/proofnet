"""bench_v1: real kernel workload used as the scheduling weight. NumPy + stdlib only."""

import statistics
import time
from typing import Any

import numpy as np

from . import gaussian_nb

BENCH_VERSION = "bench_v1"
ROWS, FEATURES, CLASSES, REPEATS = 50_000, 16, 3, 3


def run() -> dict[str, Any]:
    """Run the actual GNB map kernel on a fixed-seed matrix; score = cells / median seconds."""
    rng = np.random.default_rng(12345)
    x = rng.normal(size=(ROWS, FEATURES))
    y = rng.integers(0, CLASSES, size=ROWS)
    times: list[float] = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        gaussian_nb.map(x, y, CLASSES)
        times.append(time.perf_counter() - t0)
    median = statistics.median(times)
    return {
        "bench_version": BENCH_VERSION,
        "score_cells_per_sec": ROWS * FEATURES / median,
        "median_seconds": median,
        "runs_seconds": times,
    }
