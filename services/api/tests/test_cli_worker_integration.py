"""Level 2 (CLI side): simulated devices register, benchmark, appear idle and go offline."""

from datetime import timedelta
from typing import Any, cast

import httpx
from api_helpers import signup
from fastapi.testclient import TestClient
from pymongo import MongoClient

from cli_worker.worker import StateFile, Worker, WorkerError, login_or_signup
from proofnet_api.config import Settings
from proofnet_api.db import utcnow

V1 = "/api/v1"
Mongo = MongoClient[dict[str, Any]]


class _Prefixed:
    """Mounts the shared ASGI TestClient under /api/v1 like httpx.Client(base_url=...)."""

    def __init__(self, client: TestClient) -> None:
        self._c = client

    def get(self, url: str, **kw: Any) -> httpx.Response:
        return cast(httpx.Response, self._c.get(V1 + url, **kw))

    def post(self, url: str, **kw: Any) -> httpx.Response:
        return cast(httpx.Response, self._c.post(V1 + url, **kw))


def _http(client: TestClient) -> httpx.Client:
    return cast(httpx.Client, _Prefixed(client))


def _make_worker(
    client: TestClient, email: str, name: str, state: StateFile | None = None
) -> Worker:
    http = _http(client)
    token = login_or_signup(http, email, "password123")
    return Worker(http, token, name, state=state)


def test_cli_worker_registers_benchmarks_and_heartbeats(
    client: TestClient, mongo: Mongo, settings: Settings
) -> None:
    w = _make_worker(client, "cliw1@example.com", "cli-1")
    w.register()
    manifest = w.load_runtime()
    w.start_session(manifest)
    body = w.heartbeat_once()
    assert body["directives"] == []
    doc = mongo[settings.mongodb_db]["devices"].find_one({"_id": w.device_id})
    assert doc is not None
    assert doc["status"] == "idle" and doc["runtime"]["kind"] == "cpython"
    assert doc["benchmark"]["bench_version"] == "bench_v1"
    assert doc["benchmark"]["score_cells_per_sec"] == w.score and w.score and w.score > 0


def test_three_cli_workers_appear_in_network_summary(client: TestClient) -> None:
    headers = signup(client, "cli3@example.com")
    http = _http(client)
    token = login_or_signup(http, "cli3@example.com", "password123")
    workers = [Worker(http, token, f"sim-{i}") for i in range(3)]
    for w in workers:
        w.register()
        w.start_session(w.load_runtime())
    summary = client.get(f"{V1}/network/summary", headers=headers).json()
    mine = {d["id"]: d for d in summary["devices"] if d["id"] in {w.device_id for w in workers}}
    assert len(mine) == 3
    assert all(d["status"] == "idle" and d["runtime_kind"] == "cpython" for d in mine.values())
    assert all(d["score_cells_per_sec"] > 0 for d in mine.values())
    assert summary["counts"]["idle"] >= 3


def test_identity_persists_across_restarts(client: TestClient, tmp_path: Any) -> None:
    state_path = tmp_path / "devices.json"
    a = _make_worker(client, "persist@example.com", "p-1", StateFile(state_path))
    a.register()
    first = a.device_id
    b = _make_worker(client, "persist@example.com", "p-1", StateFile(state_path))
    b.register()  # a new process reading the same state file
    assert b.device_id == first and b.device_token == a.device_token
    b.start_session(b.load_runtime())  # new session, same device
    assert b.session_id != a.session_id


def test_offline_then_reopen_returns_idle_with_same_device(
    client: TestClient, mongo: Mongo, settings: Settings
) -> None:
    import asyncio

    from proofnet_api import reconciler
    from proofnet_api.db import connect

    w = _make_worker(client, "offline@example.com", "off-1")
    w.register()
    w.start_session(w.load_runtime())
    devices = mongo[settings.mongodb_db]["devices"]
    devices.update_one(
        {"_id": w.device_id}, {"$set": {"last_seen_at": utcnow() - timedelta(seconds=25)}}
    )

    async def sweep() -> None:
        db = await connect(settings)
        try:
            await reconciler.mark_offline_devices(db, settings)
        finally:
            await db.close()

    asyncio.run(sweep())
    assert devices.find_one({"_id": w.device_id})["status"] == "offline"  # type: ignore[index]
    # "reopen the page": same device, new session
    w.session_id = None
    w.start_session(w.load_runtime())
    d = devices.find_one({"_id": w.device_id})
    assert d is not None and d["status"] == "idle" and d["_id"] == w.device_id


def test_tampered_bundle_is_refused(client: TestClient, monkeypatch: Any) -> None:
    w = _make_worker(client, "tamper@example.com", "t-1")
    w.register()
    real_get = w.http.get

    def evil_get(url: str, **kw: Any) -> httpx.Response:
        r = real_get(url, **kw)
        if "/runtime/kernels/" in str(url):
            return httpx.Response(200, content=r.content + b"x", request=r.request)
        return r

    monkeypatch.setattr(w.http, "get", evil_get)
    try:
        w.load_runtime()
    except WorkerError as e:
        assert "SHA-256 mismatch" in str(e)
    else:
        raise AssertionError("tampered bundle was accepted")
