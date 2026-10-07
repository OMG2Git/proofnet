"""Tiny in-process TTL cache for live snapshots (ARCHITECTURE 3.5: ~1 s, to respect Atlas limits)."""

import time
from typing import Any

from fastapi import FastAPI


def get(app: FastAPI, key: str) -> Any | None:
    store: dict[str, tuple[float, Any]] = app.state.snapshot_cache
    hit = store.get(key)
    if hit is None or hit[0] < time.monotonic():
        return None
    return hit[1]


def put(app: FastAPI, key: str, value: Any, ttl_seconds: float) -> None:
    if ttl_seconds <= 0:
        return
    store: dict[str, tuple[float, Any]] = app.state.snapshot_cache
    if len(store) > 256:  # bounded: drop expired entries
        now = time.monotonic()
        for k in [k for k, v in store.items() if v[0] < now]:
            del store[k]
    store[key] = (time.monotonic() + ttl_seconds, value)
