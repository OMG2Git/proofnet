"""Linear/Ridge regression sufficient statistics: worker map + merge. NumPy + stdlib only.

State/payload: {"n": int, "mean_x": [d], "mean_y": float, "Sxx": [d][d], "Sxy": [d], "Syy": float}
"""

from typing import Any

import numpy as np

from .moments import comoments, merge_comoments
from .serialize import from_list, to_list

KERNEL = "linear_ridge_train"
VERSION = "1"


def map(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:  # noqa: A001
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    return _pack(comoments(x, y))


def merge(partials: list[dict[str, Any]]) -> dict[str, Any]:
    if not partials:
        raise ValueError("nothing to merge")
    acc = _unpack(partials[0])
    for nxt in partials[1:]:
        acc = merge_comoments(acc, _unpack(nxt))
    return _pack(acc)


def _pack(s: tuple[int, np.ndarray, float, np.ndarray, np.ndarray, float]) -> dict[str, Any]:
    n, mx, my, sxx, sxy, syy = s
    return {
        "n": n,
        "mean_x": to_list(mx),
        "mean_y": float(my),
        "Sxx": to_list(sxx),
        "Sxy": to_list(sxy),
        "Syy": float(syy),
    }


def _unpack(p: dict[str, Any]) -> tuple[int, np.ndarray, float, np.ndarray, np.ndarray, float]:
    return (
        int(p["n"]),
        from_list(p["mean_x"]),
        float(p["mean_y"]),
        from_list(p["Sxx"]),
        from_list(p["Sxy"]),
        float(p["Syy"]),
    )
