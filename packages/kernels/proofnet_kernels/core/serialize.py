"""Canonical JSON, digests and array encoding. NumPy + stdlib only."""

import hashlib
import json
import math
from typing import Any

import numpy as np


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, NaN/Infinity rejected."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def payload_sha256(payload: Any) -> str:
    return sha256_hex(canonical_json(payload).encode("utf-8"))


def to_list(a: np.ndarray) -> Any:
    """float64 array -> nested lists of Python floats (exact JSON round-trip)."""
    return np.asarray(a, dtype=np.float64).tolist()


def from_list(v: Any) -> np.ndarray:
    return np.asarray(v, dtype=np.float64)


def all_finite(v: Any) -> bool:
    """True if every number in a nested list/scalar structure is finite."""
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return math.isfinite(v)
    if isinstance(v, list):
        return all(all_finite(x) for x in v)
    return False
