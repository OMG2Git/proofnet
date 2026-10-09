"""Global event feed for the admin network dashboard (GET /network/events)."""

import secrets
from collections.abc import Iterator
from typing import Any

import pytest
from api_helpers import signup
from e2e_helpers import V1, Env, clf_csv, drive, isolate, make_worker, manifest, status, upload
from fastapi.testclient import TestClient
from pymongo import MongoClient

from proofnet_api.config import Settings
from proofnet_api.main import create_app

ADMIN = f"netadmin{secrets.token_hex(3)}@example.com"


@pytest.fixture(scope="module")
def admin_env() -> Iterator[Env]:
    base = Settings()
    s = Settings(
        mongodb_uri=base.mongodb_uri,
        mongodb_db=f"proofnet_test_netev_{secrets.token_hex(4)}",
        jwt_secret="test-secret-test-secret-test-secret-123",
        status_cache_seconds=0,
        reconciler_interval_seconds=1,
        admin_emails=ADMIN,
    )
    mongo: MongoClient[dict[str, Any]] = MongoClient(s.mongodb_uri, tz_aware=True)
    try:
        with TestClient(create_app(s)) as c:
            yield c, mongo, s
    finally:
        mongo.drop_database(s.mongodb_db)
        mongo.close()


@pytest.fixture(scope="module")
def boss(admin_env: Env) -> dict[str, str]:
    return signup(admin_env[0], ADMIN)


def test_events_feed_is_admin_only(admin_env: Env, boss: dict[str, str]) -> None:
    client, _, _ = admin_env
    other = signup(client)
    assert client.get(f"{V1}/network/events").status_code == 401
    assert client.get(f"{V1}/network/events", headers=other).status_code == 403
    assert client.get(f"{V1}/network/events", headers=boss).status_code == 200
    assert client.get(f"{V1}/network/events?limit=0", headers=boss).status_code == 422


def test_events_feed_reports_a_real_task_lifecycle(admin_env: Env, boss: dict[str, str]) -> None:
    client, _, _ = admin_env
    ds = upload(client, boss, clf_csv())
    w = make_worker(client, "netev1@example.com", "netev-dev")
    isolate(admin_env, w)
    r = client.post(f"{V1}/tasks", headers=boss, json=manifest(ds, "gaussian_nb_train", "label", 4))
    assert r.status_code == 201, r.text
    tid = r.json()["id"]
    drive(w, lambda: status(client, boss, tid)["task"]["status"] in ("completed", "failed"))

    feed = client.get(f"{V1}/network/events?limit=200", headers=boss).json()
    mine = [e for e in feed if e["task_id"] == tid]
    types = [e["type"] for e in mine]
    for expected in ("task_created", "chunk_assigned", "assignment_started", "result_accepted"):
        assert expected in types, types
    assert "task_completed" in types
    assert all("T" in e["ts"] for e in feed)  # ISO timestamps
    assert [e["ts"] for e in feed] == sorted(e["ts"] for e in feed)  # oldest first
    assigned = next(e for e in mine if e["type"] == "chunk_assigned")
    assert assigned["device_id"] == w.device_id
    assert "netev-dev" in assigned["message"]  # human line names the real device
    # the per-task endpoint and the global feed describe the same events
    per_task = client.get(f"{V1}/tasks/{tid}/events", headers=boss).json()
    assert {e["id"] for e in per_task} <= {e["id"] for e in feed}

    # `since` returns only newer events, so polling is cheap and idempotent
    last = feed[-1]["ts"]
    assert client.get(f"{V1}/network/events", params={"since": last}, headers=boss).json() == []
    first_gap = client.get(
        f"{V1}/network/events", params={"since": feed[0]["ts"]}, headers=boss
    ).json()
    assert feed[0]["id"] not in {e["id"] for e in first_gap}
    # limit keeps the newest events
    newest = client.get(f"{V1}/network/events?limit=2", headers=boss).json()
    assert [e["id"] for e in newest] == [e["id"] for e in feed[-2:]]
