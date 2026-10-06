"""Gaussian Naive Bayes sufficient statistics: worker map + merge. NumPy + stdlib only.

State/payload (JSON-serializable):
  {"n": int, "mean": [d], "M2": [d],                 # overall, for variance smoothing
   "classes": {"n": [C], "mean": [C][d], "M2": [C][d]}}
"""

from typing import Any

import numpy as np

from .moments import merge_moments, moments
from .serialize import from_list, to_list

KERNEL = "gaussian_nb_train"
VERSION = "1"


def map(x: np.ndarray, y: np.ndarray, n_classes: int) -> dict[str, Any]:  # noqa: A001
    """Partial result for one partition. y holds integer labels in [0, n_classes)."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.asarray(y).astype(np.int64)
    d = x.shape[1]
    n, mean, m2 = moments(x)
    cn: list[int] = []
    cmean = np.zeros((n_classes, d))
    cm2 = np.zeros((n_classes, d))
    for c in range(n_classes):
        k, mc, m2c = moments(x[y == c])
        cn.append(k)
        cmean[c] = mc
        cm2[c] = m2c
    return {
        "n": n,
        "mean": to_list(mean),
        "M2": to_list(m2),
        "classes": {"n": cn, "mean": to_list(cmean), "M2": to_list(cm2)},
    }


def merge(partials: list[dict[str, Any]]) -> dict[str, Any]:
    """Pairwise left-to-right merge in the given (chunk-index) order."""
    if not partials:
        raise ValueError("nothing to merge")
    acc = partials[0]
    for nxt in partials[1:]:
        acc = _merge2(acc, nxt)
    return acc


def _merge2(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    n, mean, m2 = merge_moments(
        (a["n"], from_list(a["mean"]), from_list(a["M2"])),
        (b["n"], from_list(b["mean"]), from_list(b["M2"])),
    )
    ca, cb = a["classes"], b["classes"]
    n_classes = len(ca["n"])
    d = len(a["mean"])
    cn: list[int] = []
    cmean = np.zeros((n_classes, d))
    cm2 = np.zeros((n_classes, d))
    for c in range(n_classes):
        k, mc, m2c = merge_moments(
            (ca["n"][c], from_list(ca["mean"][c]), from_list(ca["M2"][c])),
            (cb["n"][c], from_list(cb["mean"][c]), from_list(cb["M2"][c])),
        )
        cn.append(k)
        cmean[c] = mc
        cm2[c] = m2c
    return {
        "n": n,
        "mean": to_list(mean),
        "M2": to_list(m2),
        "classes": {"n": cn, "mean": to_list(cmean), "M2": to_list(cm2)},
    }
