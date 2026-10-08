"""Level 7: failures are survivable and visible (PHASE_PLAN P6 scenario table).

CLI workers do real work and inject faults; the API runs against real MongoDB with short
timeouts so each scenario finishes in seconds."""

import asyncio
import secrets
import time
from datetime import timedelta
from typing import Any, cast

import httpx
import pytest
from api_helpers import signup
from e2e_helpers import (
    V1,
    Env,
    Prefixed,
    clf_csv,
    drive_many,
    heartbeat_holding,
    make_workers,
    manifest,
    only_these,
    raw_dispatch,
    result_body,
    set_scores,
    status,
    upload,
)
from fastapi.testclient import TestClient
from pymongo import MongoClient

from cli_worker.worker import Faults, Worker
from proofnet_api.config import Settings
from proofnet_api.db import connect, utcnow
from proofnet_api.main import create_app
from proofnet_api.scheduling.scheduler import deadline_seconds

DONE = ("completed", "failed", "cancelled")


def new_task(
    env: Env, h: dict[str, str], n: int = 4000, max_devices: int = 1, seed: int = 1
) -> str:
    client, _, _ = env
    m = manifest(upload(client, h, clf_csv(n=n, seed=seed)), "gaussian_nb_train", "label", 4)
    m["execution"] = {"min_devices": 1, "max_devices": max_devices}
    r = client.post(f"{V1}/tasks", headers=h, json=m)
    assert r.status_code == 201, r.text
    return str(r.json()["id"])


def wait_status(
    env: Env, workers: list[Worker], h: dict[str, str], tid: str, states: tuple[str, ...] = DONE
) -> dict[str, Any]:
    client, _, _ = env
    drive_many(workers, lambda: status(client, h, tid)["task"]["status"] in states, timeout=60)
    return status(client, h, tid)


def messages(env: Env, h: dict[str, str], tid: str) -> list[str]:
    client, _, _ = env
    ev = client.get(f"{V1}/tasks/{tid}/events", headers=h).json()
    return [e["message"] for e in ev]


# ---------- worker dies / corrupt / always fails ----------
def test_worker_dies_mid_chunk_chunk_is_reassigned_and_task_completes(fast_env: Env) -> None:
    client, mongo, s = fast_env
    h = signup(client)
    a, b = make_workers(client, "die", 2)
    only_these(fast_env, [a, b])
    set_scores(fast_env, [a, b], [3e6, 1e6])  # the doomed device is the fastest -> gets the chunk
    a.faults = Faults(die_after_start=True)
    tid = new_task(fast_env, h)
    st = wait_status(fast_env, [a, b], h, tid)
    assert st["task"]["status"] == "completed", st["task"]["error"]
    asgs = st["assignments"]
    assert [x["status"] for x in asgs] == ["expired", "succeeded"]
    assert asgs[0]["device_name"] == a.name and asgs[1]["device_name"] == b.name
    assert asgs[0]["error"]["code"] == "DEVICE_OFFLINE"
    assert st["chunks"][0]["attempt_count"] == 2 and st["chunks"][0]["status"] == "completed"
    assert st["task"]["result"]["reference_check"]["passed"] is True
    dev = mongo[s.mongodb_db]["devices"].find_one({"_id": a.device_id})
    assert dev is not None and dev["stats"]["expired"] == 1 and dev["status"] == "offline"
    msgs = messages(fast_env, h, tid)
    assert any(f"{a.name} went offline - chunk 0 will be reassigned" in m for m in msgs)
    assert any("returned to the queue (attempt 1/3)" in m for m in msgs)


def test_corrupt_payload_is_rejected_retried_elsewhere_and_counted(fast_env: Env) -> None:
    client, mongo, s = fast_env
    h = signup(client)
    a, b = make_workers(client, "cor", 2)
    only_these(fast_env, [a, b])
    set_scores(fast_env, [a, b], [3e6, 1e6])
    a.faults = Faults(corrupt_result=True)
    tid = new_task(fast_env, h, seed=2)
    st = wait_status(fast_env, [a, b], h, tid)
    assert st["task"]["status"] == "completed", st["task"]["error"]
    assert [x["status"] for x in st["assignments"]] == ["rejected", "succeeded"]
    dev = mongo[s.mongodb_db]["devices"].find_one({"_id": a.device_id})
    assert dev is not None and dev["stats"]["invalid_results"] == 1
    prs = list(mongo[s.mongodb_db]["partial_results"].find({"task_id": tid}))
    assert sorted(p["acceptance"] for p in prs) == ["rejected_structural", "verified"]
    assert any("was rejected" in m for m in messages(fast_env, h, tid))
    assert st["task"]["result"]["reference_check"]["passed"] is True  # the bad data never merged


def test_worker_that_always_fails_exhausts_three_attempts_and_fails_the_task(
    fast_env: Env,
) -> None:
    client, mongo, s = fast_env
    h = signup(client)
    (w,) = make_workers(client, "fail", 1)
    only_these(fast_env, [w])
    w.faults = Faults(fail_rate=1.0, seed=1)
    tid = new_task(fast_env, h, seed=3)
    st = wait_status(fast_env, [w], h, tid)
    assert st["task"]["status"] == "failed"
    assert "failed after 3 attempts" in st["task"]["error"]
    assert [a["status"] for a in st["assignments"]] == ["failed"] * 3
    assert st["chunks"][0]["status"] == "failed" and st["chunks"][0]["attempt_count"] == 3
    assert st["artifacts"] == [] and st["task"]["result"] is None  # no partial aggregation
    dev = mongo[s.mongodb_db]["devices"].find_one({"_id": w.device_id})
    assert dev is not None and dev["stats"]["failed"] == 3 and dev["status"] == "idle"
    assert any("Task failed" in m for m in messages(fast_env, h, tid))


# ---------- late result, deadline, session restart ----------
def test_late_result_after_reassignment_is_409_logged_and_never_merged(fast_env: Env) -> None:
    client, mongo, s = fast_env
    h = signup(client)
    a, b = make_workers(client, "late", 2)
    only_these(fast_env, [a, b])
    set_scores(fast_env, [a, b], [3e6, 1e6])
    tid = new_task(fast_env, h, seed=4)
    asg = raw_dispatch(client, a)  # A gets the chunk and starts it...
    base = f"{V1}/worker/assignments/{asg['assignment_id']}"
    assert client.post(f"{base}/start", headers=a._dev()).status_code == 200
    body = result_body(a, asg, client)  # ...computes the right answer...
    mongo[s.mongodb_db]["assignments"].update_one(  # ...but its lease runs out first
        {"_id": asg["assignment_id"]}, {"$set": {"deadline_at": utcnow() - timedelta(seconds=1)}}
    )
    end = time.time() + 20
    while time.time() < end:
        heartbeat_holding(client, a, asg["assignment_id"])
        cur = status(client, h, tid)["assignments"][0]["status"]
        if cur == "expired":
            break
        time.sleep(0.3)
    assert status(client, h, tid)["assignments"][0]["error"]["code"] == "DEADLINE"
    # A still holds the closed assignment: its next heartbeat carries a cancel directive
    hb = heartbeat_holding(client, a, asg["assignment_id"])
    assert {"type": "cancel", "assignment_id": asg["assignment_id"]} in hb["directives"]
    late = client.post(f"{base}/result", headers=a._dev(), json=body)
    assert late.status_code == 409 and late.json()["error"]["code"] == "LATE_RESULT"
    ev = mongo[s.mongodb_db]["events"].find_one(
        {"assignment_id": asg["assignment_id"], "type": "late_result"}
    )
    assert ev is not None and ev["data"]["payload_sha256"] == body["payload_sha256"]
    st = wait_status(fast_env, [b], h, tid)  # B (not A) completes the task
    assert st["task"]["status"] == "completed", st["task"]["error"]
    prs = list(mongo[s.mongodb_db]["partial_results"].find({"task_id": tid}))
    assert len(prs) == 1 and prs[0]["device_id"] == b.device_id  # the late payload never stored
    assert any("Late result" in m for m in messages(fast_env, h, tid))


def test_deadline_expiry_then_same_device_may_retry_when_nobody_else_can(fast_env: Env) -> None:
    client, mongo, s = fast_env
    h = signup(client)
    (w,) = make_workers(client, "dl", 1)
    only_these(fast_env, [w])
    tid = new_task(fast_env, h, seed=5)
    asg = raw_dispatch(client, w)  # assigned, never started...
    mongo[s.mongodb_db]["assignments"].update_one(
        {"_id": asg["assignment_id"]}, {"$set": {"deadline_at": utcnow() - timedelta(seconds=1)}}
    )
    end = time.time() + 20  # heartbeat without executing until the reconciler expires it
    while time.time() < end and status(client, h, tid)["assignments"][0]["status"] != "expired":
        heartbeat_holding(client, w, None, state="idle")
        time.sleep(0.3)
    st = wait_status(fast_env, [w], h, tid)  # then (after the relax delay) the device retries
    assert st["task"]["status"] == "completed", st["task"]["error"]
    assert [a["status"] for a in st["assignments"]] == ["expired", "succeeded"]
    assert st["assignments"][0]["error"]["code"] == "DEADLINE"
    assert st["chunks"][0]["attempt_count"] == 2


def test_session_restart_mid_chunk_expires_the_lost_computation(fast_env: Env) -> None:
    client, mongo, s = fast_env
    h = signup(client)
    (w,) = make_workers(client, "sess", 1)
    only_these(fast_env, [w])
    tid = new_task(fast_env, h, seed=6)
    asg = raw_dispatch(client, w)
    assert (
        client.post(
            f"{V1}/worker/assignments/{asg['assignment_id']}/start", headers=w._dev()
        ).status_code
        == 200
    )
    w.session_id = None
    w.start_session(w.load_runtime())  # the page was reloaded: Pyodide state is gone
    first = status(client, h, tid)["assignments"][0]
    assert first["status"] == "expired" and first["error"]["code"] == "SESSION_RESTARTED"
    st = wait_status(fast_env, [w], h, tid)  # retried (same device, after the relax delay)
    assert st["task"]["status"] == "completed", st["task"]["error"]
    assert any("restarted its worker" in m for m in messages(fast_env, h, tid))


def test_unstarted_assignment_is_redispatched_after_a_session_restart(fast_env: Env) -> None:
    """The stale-page case seen on real phones: assigned, ignored, then the page reloads."""
    client, _, _ = fast_env
    h = signup(client)
    (w,) = make_workers(client, "stale", 1)
    only_these(fast_env, [w])
    tid = new_task(fast_env, h, seed=7)
    first = raw_dispatch(client, w)
    w.session_id = None
    w.start_session(w.load_runtime())
    again = raw_dispatch(client, w)
    assert again["assignment_id"] == first["assignment_id"]  # same assignment, no wasted attempt
    st = wait_status(fast_env, [w], h, tid)
    assert st["task"]["status"] == "completed" and st["chunks"][0]["attempt_count"] == 1


# ---------- cancel ----------
def test_user_cancel_mid_run_cancels_assignments_and_tells_the_worker(fast_env: Env) -> None:
    client, mongo, s = fast_env
    h = signup(client)
    (w,) = make_workers(client, "cancel", 1)
    only_these(fast_env, [w])
    tid = new_task(fast_env, h, seed=8)
    asg = raw_dispatch(client, w)
    base = f"{V1}/worker/assignments/{asg['assignment_id']}"
    client.post(f"{base}/start", headers=w._dev())
    body = result_body(w, asg, client)
    assert client.post(f"{V1}/tasks/{tid}/cancel", headers=signup(client)).status_code == 404
    r = client.post(f"{V1}/tasks/{tid}/cancel", headers=h)
    assert r.status_code == 200 and r.json()["status"] == "cancelled"
    st = status(client, h, tid)
    assert [a["status"] for a in st["assignments"]] == ["cancelled"]
    assert all(c["status"] == "cancelled" for c in st["chunks"])
    dev = mongo[s.mongodb_db]["devices"].find_one({"_id": w.device_id})
    assert dev is not None and dev["status"] == "idle" and dev["current_assignment_id"] is None
    # the worker is told to stop on its next heartbeat; its late result is refused
    hb = heartbeat_holding(client, w, asg["assignment_id"])
    assert {"type": "cancel", "assignment_id": asg["assignment_id"]} in hb["directives"]
    assert client.post(f"{base}/result", headers=w._dev(), json=body).status_code == 409
    # cancel is idempotent; the device is immediately usable again
    assert client.post(f"{V1}/tasks/{tid}/cancel", headers=h).status_code == 200
    again = new_task(fast_env, h, seed=9)
    assert wait_status(fast_env, [w], h, again)["task"]["status"] == "completed"
    # a finished task cannot be cancelled
    done = client.post(f"{V1}/tasks/{again}/cancel", headers=h)
    assert done.status_code == 409 and done.json()["error"]["code"] == "CONFLICT"
    assert "Task cancelled by the owner" in messages(fast_env, h, tid)


def test_cancel_a_queued_task(fast_env: Env) -> None:
    client, mongo, s = fast_env
    h = signup(client)
    mongo[s.mongodb_db]["devices"].update_many({}, {"$set": {"status": "disabled"}})
    tid = new_task(fast_env, h, seed=10)
    assert client.post(f"{V1}/tasks/{tid}/cancel", headers=h).json()["status"] == "cancelled"
    st = status(client, h, tid)
    assert st["chunks"] == [] and st["task"]["status"] == "cancelled"


# ---------- timeouts ----------
def test_no_devices_task_stays_queued_then_fails_at_the_queue_timeout(fast_env: Env) -> None:
    client, mongo, s = fast_env
    h = signup(client)
    mongo[s.mongodb_db]["devices"].update_many({}, {"$set": {"status": "disabled"}})
    tid = new_task(fast_env, h, seed=11)
    time.sleep(1.5)
    st = status(client, h, tid)
    assert st["task"]["status"] == "queued" and st["waiting_reasons"]
    end = time.time() + 30
    while time.time() < end and status(client, h, tid)["task"]["status"] == "queued":
        time.sleep(0.5)
    st = status(client, h, tid)
    assert st["task"]["status"] == "failed" and "queue timeout" in st["task"]["error"]
    assert any("Task failed" in m for m in messages(fast_env, h, tid))


def test_task_timeout_fails_the_task_and_stops_a_slow_worker(timeout_env: Env) -> None:
    client, mongo, s = timeout_env
    h = signup(client)
    (w,) = make_workers(client, "slow", 1)
    only_these(timeout_env, [w])
    tid = new_task(timeout_env, h, seed=12)
    asg = raw_dispatch(client, w)
    client.post(f"{V1}/worker/assignments/{asg['assignment_id']}/start", headers=w._dev())
    end = time.time() + 25
    cancel_seen = False
    while time.time() < end:  # a worker that keeps heartbeating but never finishes
        hb = heartbeat_holding(client, w, asg["assignment_id"])
        cancel_seen = cancel_seen or any(d["type"] == "cancel" for d in hb["directives"])
        if status(client, h, tid)["task"]["status"] == "failed":
            break
        time.sleep(0.5)
    st = status(client, h, tid)
    assert st["task"]["status"] == "failed" and "task timeout" in st["task"]["error"]
    assert [a["status"] for a in st["assignments"]] == ["cancelled"]
    hb = heartbeat_holding(client, w, asg["assignment_id"])
    assert cancel_seen or any(d["type"] == "cancel" for d in hb["directives"])


# ---------- restart / aggregation ----------
def test_backend_restart_mid_task_state_survives_and_task_completes() -> None:
    base = Settings()
    s = Settings(
        mongodb_uri=base.mongodb_uri,
        mongodb_db=f"proofnet_test_restart_{secrets.token_hex(4)}",
        jwt_secret="test-secret-test-secret-test-secret-123",
        status_cache_seconds=0,
        reconciler_interval_seconds=1,
    )
    mongo: MongoClient[dict[str, Any]] = MongoClient(s.mongodb_uri, tz_aware=True)
    try:
        with TestClient(create_app(s)) as c1:  # --- first backend process ---
            h = signup(c1)
            from e2e_helpers import make_worker

            w = make_worker(c1, "restart@example.com", "restart-1")
            tid = c1.post(
                f"{V1}/tasks",
                headers=h,
                json=manifest(
                    upload(c1, h, clf_csv(n=4000, seed=13)), "gaussian_nb_train", "label", 4
                ),
            ).json()["id"]
            asg = raw_dispatch(c1, w)
            assert (
                c1.post(
                    f"{V1}/worker/assignments/{asg['assignment_id']}/start", headers=w._dev()
                ).status_code
                == 200
            )
            body = result_body(w, asg, c1)
        # --- the backend process is gone; a new one starts on the same database ---
        with TestClient(create_app(s)) as c2:
            w.http = cast(httpx.Client, Prefixed(c2))
            st = status(c2, h, tid)
            assert st["task"]["status"] == "running" and st["assignments"][0]["status"] == "running"
            r = c2.post(
                f"{V1}/worker/assignments/{asg['assignment_id']}/result",
                headers=w._dev(),
                json=body,
            )
            assert r.status_code == 200, r.text
            end = time.time() + 40
            while time.time() < end and status(c2, h, tid)["task"]["status"] != "completed":
                time.sleep(0.5)
            st = status(c2, h, tid)
            assert st["task"]["status"] == "completed", st["task"]["error"]
            assert st["task"]["result"]["reference_check"]["passed"] is True
    finally:
        mongo.drop_database(s.mongodb_db)
        mongo.close()


def test_stuck_aggregation_is_resumed_by_the_reconciler(
    fast_env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the request path never runs aggregation (e.g. the process died right after the last
    chunk), the reconciler finds the task in `aggregating` and finishes it."""
    client, _, _ = fast_env

    async def never(*_: Any, **__: Any) -> bool:
        return False

    monkeypatch.setattr("proofnet_api.worker_gateway.assignments.aggregate_task", never)
    h = signup(client)
    (w,) = make_workers(client, "agg", 1)
    only_these(fast_env, [w])
    tid = new_task(fast_env, h, seed=14)
    st = wait_status(fast_env, [w], h, tid, states=("aggregating", "completed", "failed"))
    assert st["task"]["status"] in ("aggregating", "completed")
    st = wait_status(fast_env, [w], h, tid)
    assert st["task"]["status"] == "completed" and st["task"]["result"]["reference_check"]["passed"]
    hist = [x["status"] for x in st["task"]["status_history"]]
    assert hist == ["queued", "running", "aggregating", "completed"]  # aggregated exactly once


def test_stale_aggregation_claim_is_released() -> None:
    """Uses its own database: a running reconciler would release the claim before we check."""
    from proofnet_api.aggregation.service import CLAIM_STALE_SECONDS, release_stale_claims

    base = Settings()
    s = Settings(
        mongodb_uri=base.mongodb_uri,
        mongodb_db=f"proofnet_test_claim_{secrets.token_hex(4)}",
        jwt_secret="test-secret-test-secret-test-secret-123",
    )
    mongo: MongoClient[dict[str, Any]] = MongoClient(s.mongodb_uri, tz_aware=True)
    col = mongo[s.mongodb_db]["tasks"]
    now = utcnow()
    col.insert_many(
        [
            {"_id": "fresh", "status": "aggregating", "aggregation_claimed_at": now},
            {
                "_id": "stale",
                "status": "aggregating",
                "aggregation_claimed_at": now - timedelta(seconds=CLAIM_STALE_SECONDS + 30),
            },
            {
                "_id": "done",
                "status": "completed",
                "aggregation_claimed_at": now - timedelta(days=1),
            },
        ]
    )

    async def go() -> int:
        db = await connect(s)
        try:
            return await release_stale_claims(db)
        finally:
            await db.close()

    try:
        assert asyncio.run(go()) == 1
        assert col.find_one({"_id": "stale"})["aggregation_claimed_at"] is None  # type: ignore[index]
        assert col.find_one({"_id": "fresh"})["aggregation_claimed_at"] is not None  # type: ignore[index]
        assert col.find_one({"_id": "done"})["aggregation_claimed_at"] is not None  # type: ignore[index]
    finally:
        mongo.drop_database(s.mongodb_db)
        mongo.close()


# ---------- deadline sizing ----------
def test_deadline_includes_a_transfer_allowance() -> None:
    assert deadline_seconds(0.03) == 60  # tiny compute: the 60 s floor
    # a 10.9 MB chunk at >= 50 KB/s needs ~218 s (a real phone took 110 s on weak 5G)
    assert deadline_seconds(0.03, 10_880_000, 50_000) == pytest.approx(0.12 + 217.6, abs=1)
    assert deadline_seconds(1000.0, 10**9, 50_000) == 600  # the 600 s ceiling
