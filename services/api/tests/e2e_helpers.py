"""Shared helpers for end-to-end API tests (CLI workers driven through the live API)."""

import time
from collections.abc import Callable
from typing import Any, cast

import httpx
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from pymongo import MongoClient

from cli_worker.worker import Worker, login_or_signup
from proofnet_api.config import Settings

V1 = "/api/v1"
Mongo = MongoClient[dict[str, Any]]
Env = tuple[TestClient, Mongo, Settings]


class Prefixed:
    def __init__(self, c: TestClient) -> None:
        self._c = c

    def get(self, url: str, **kw: Any) -> httpx.Response:
        return cast(httpx.Response, self._c.get(V1 + url, **kw))

    def post(self, url: str, **kw: Any) -> httpx.Response:
        return cast(httpx.Response, self._c.post(V1 + url, **kw))


def clf_csv(n: int = 3000, d: int = 4, classes: int = 3, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, classes, size=n)
    x = rng.normal(size=(n, d)) + y[:, None] * 0.8
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(d)])
    df["label"] = y
    return bytes(df.to_csv(index=False).encode())


def reg_csv(n: int = 3000, d: int = 4, seed: int = 1) -> bytes:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    y = x @ rng.normal(size=d) + 1.5 + rng.normal(scale=0.2, size=n)
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(d)])
    df["target"] = y
    return bytes(df.to_csv(index=False).encode())


def make_worker(client: TestClient, email: str, name: str) -> Worker:
    http = cast(httpx.Client, Prefixed(client))
    token = login_or_signup(http, email, "password123")
    w = Worker(http, token, name)
    w.register()
    w.start_session(w.load_runtime())
    return w


def isolate(env: tuple[TestClient, Mongo, Settings], w: Worker) -> None:
    """Leftover devices from earlier scenarios must not take this scenario's work."""
    _, mongo, s = env
    mongo[s.mongodb_db]["devices"].update_many(
        {"_id": {"$ne": w.device_id}}, {"$set": {"status": "disabled"}}
    )


def upload(client: TestClient, h: dict[str, str], data: bytes) -> str:
    r = client.post(f"{V1}/datasets", headers=h, files={"file": ("d.csv", data, "text/csv")})
    assert r.status_code == 201, r.text
    return str(r.json()["id"])


def manifest(ds: str, task_type: str, target: str, d: int, **params: Any) -> dict[str, Any]:
    return {
        "name": "e2e",
        "task_type": task_type,
        "dataset_id": ds,
        "params": {
            "target_column": target,
            "feature_columns": [f"f{i}" for i in range(d)],
            **params,
        },
        "execution": {"min_devices": 1, "max_devices": 1},
    }


def drive(worker: Worker, done: Callable[[], bool], timeout: float = 45) -> None:
    """Heartbeat like the real worker loop until `done()`; fail loudly on timeout."""
    end = time.time() + timeout
    while time.time() < end:
        worker.heartbeat_once()
        if done():
            return
        time.sleep(0.25)
    raise AssertionError("timed out waiting for the task")


def status(client: TestClient, h: dict[str, str], task_id: str) -> dict[str, Any]:
    r = client.get(f"{V1}/tasks/{task_id}/status", headers=h)
    assert r.status_code == 200, r.text
    return cast(dict[str, Any], r.json())


def drive_many(workers: list[Worker], done: Callable[[], bool], timeout: float = 90) -> None:
    """Heartbeat every worker in turn (like several real worker loops) until `done()`."""
    end = time.time() + timeout
    while time.time() < end:
        for w in workers:
            w.heartbeat_once()
        if done():
            return
        time.sleep(0.1)
    raise AssertionError("timed out waiting for the task")


def make_workers(client: TestClient, prefix: str, n: int) -> list[Worker]:
    return [make_worker(client, f"{prefix}{i}@example.com", f"{prefix}-{i}") for i in range(n)]


def only_these(env: Env, workers: list[Worker]) -> None:
    """Disable every other device so leftovers from earlier scenarios cannot take work."""
    _, mongo, s = env
    mine = [w.device_id for w in workers]
    mongo[s.mongodb_db]["devices"].update_many(
        {"_id": {"$nin": mine}}, {"$set": {"status": "disabled"}}
    )


def set_scores(env: Env, workers: list[Worker], scores: list[float]) -> None:
    """Fix benchmark scores so the planner's inputs are known (the compute stays real)."""
    _, mongo, s = env
    for w, sc in zip(workers, scores, strict=True):
        mongo[s.mongodb_db]["devices"].update_one(
            {"_id": w.device_id}, {"$set": {"benchmark.score_cells_per_sec": sc}}
        )


def assert_one_active_assignment_per_device(env: Env) -> None:
    """The MVP invariant: a device never holds two assignments at once."""
    _, mongo, s = env
    by_dev: dict[str, list[dict[str, Any]]] = {}
    for a in mongo[s.mongodb_db]["assignments"].find({}).sort("assigned_at", 1):
        by_dev.setdefault(a["device_id"], []).append(a)
    for dev, asgs in by_dev.items():
        for prev, nxt in zip(asgs, asgs[1:], strict=False):
            assert prev["finished_at"] is not None, (dev, prev["_id"])
            assert nxt["assigned_at"] >= prev["finished_at"], f"{dev} overlapped assignments"


import hashlib  # noqa: E402
import io  # noqa: E402


def result_body(w: Worker, a: dict[str, Any], client: TestClient) -> dict[str, Any]:
    from proofnet_kernels.core import gaussian_nb
    from proofnet_kernels.core.serialize import payload_sha256

    raw = client.get(V1 + a["input_url"].removeprefix("/api/v1"), headers=w._dev()).content
    assert hashlib.sha256(raw).hexdigest() == a["input_sha256"]
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        payload = gaussian_nb.map(z["X"], z["y"], a["params"]["n_classes"])
    return {
        "kernel": a["kernel"],
        "kernel_version": a["kernel_version"],
        "input_sha256": a["input_sha256"],
        "n_rows": a["n_rows"],
        "payload": payload,
        "payload_sha256": payload_sha256(payload),
        "timings": {"download_ms": 1.0, "compute_ms": 2.0, "total_ms": 3.0},
        "runtime": {"kind": "cpython", "python": "3.12", "numpy": "2.4.6", "bundle": "1"},
    }


def raw_dispatch(client: TestClient, w: Worker, timeout: float = 30) -> dict[str, Any]:
    """Heartbeat as `w` (without executing) until the run directive arrives; returns the assignment."""
    end = time.time() + timeout
    while time.time() < end:
        hb = client.post(
            f"{V1}/worker/heartbeat",
            headers=w._dev(),
            json={"session_id": w.session_id, "state": "idle"},
        ).json()
        runs = [d for d in hb["directives"] if d["type"] == "run"]
        if runs:
            return cast(dict[str, Any], runs[0]["assignment"])
        time.sleep(0.2)
    raise AssertionError("no run directive arrived")


def heartbeat_holding(
    client: TestClient, w: Worker, assignment_id: str | None, state: str = "busy"
) -> dict[str, Any]:
    hb = client.post(
        f"{V1}/worker/heartbeat",
        headers=w._dev(),
        json={"session_id": w.session_id, "state": state, "current_assignment_id": assignment_id},
    )
    assert hb.status_code == 200, hb.text
    return cast(dict[str, Any], hb.json())
