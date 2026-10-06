import json

import numpy as np
import pandas as pd


def partition(n: int, k: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """k contiguous, possibly very uneven, non-empty ranges tiling [0, n)."""
    if k == 1:
        return [(0, n)]
    cuts = sorted(rng.choice(np.arange(1, n), size=k - 1, replace=False).tolist())
    edges = [0, *cuts, n]
    return list(zip(edges[:-1], edges[1:], strict=True))


def roundtrip(payload: dict) -> dict:  # type: ignore[type-arg]
    """Simulate the worker->backend JSON hop."""
    return json.loads(json.dumps(payload, allow_nan=False))  # type: ignore[no-any-return]


def clf_frame(n: int, d: int, classes: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, classes, size=n)
    x = rng.normal(size=(n, d)) * rng.uniform(0.5, 3, size=d) + y[:, None] * 0.7
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(d)])
    df["label"] = y
    return df


def reg_frame(n: int, d: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)) * rng.uniform(0.5, 3, size=d) + rng.normal(size=d)
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(d)])
    df["target"] = x @ rng.normal(size=d) + 2.0 + rng.normal(scale=0.3, size=n)
    return df
