import asyncio
import hashlib
import io
import pickle
import time
import zipfile
from datetime import timedelta
from typing import Any

import gridfs
import numpy as np
from api_helpers import SESSION_BODY, register_device, signup
from fastapi.testclient import TestClient
from pymongo import MongoClient

from proofnet_api import reconciler
from proofnet_api.config import Settings
from proofnet_api.db import connect, utcnow
from proofnet_api.main import create_app

V1 = "/api/v1"
Mongo = MongoClient[dict[str, Any]]


# ---------- health & auth ----------
def test_health_reports_db_ok(client: TestClient) -> None:
    r = client.get(f"{V1}/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "db": "ok"}


def test_signup_login_me(client: TestClient) -> None:
    h = signup(client, "alice@example.com")
    me = client.get(f"{V1}/auth/me", headers=h).json()
    assert me["email"] == "alice@example.com" and "password_hash" not in me
    ok = client.post(
        f"{V1}/auth/login", json={"email": "ALICE@example.com", "password": "password123"}
    )
    assert ok.status_code == 200
    bad = client.post(f"{V1}/auth/login", json={"email": "alice@example.com", "password": "no"})
    assert bad.status_code == 401 and bad.json()["error"]["code"] == "INVALID_CREDENTIALS"
    dup = client.post(
        f"{V1}/auth/signup",
        json={"email": "alice@example.com", "password": "password123", "display_name": "x"},
    )
    assert dup.status_code == 409


def test_auth_required(client: TestClient) -> None:
    assert client.get(f"{V1}/auth/me").status_code == 401
    junk = {"Authorization": "Bearer junk"}
    assert client.get(f"{V1}/devices/mine", headers=junk).status_code == 401


# ---------- devices & worker gateway ----------
def test_device_registration_and_token_hashed(
    client: TestClient, user_headers: dict[str, str], mongo: Mongo, settings: Settings
) -> None:
    dev_id, dev_h = register_device(client, user_headers)
    doc = mongo[settings.mongodb_db]["devices"].find_one({"_id": dev_id})
    token = dev_h["Authorization"].removeprefix("Bearer ")
    assert doc is not None and doc["status"] == "initializing"
    assert doc["token_hash"] != token and token not in str(doc)
    mine = client.get(f"{V1}/devices/mine", headers=user_headers).json()
    assert [d["id"] for d in mine] == [dev_id] and "device_token" not in mine[0]
    other = signup(client)
    assert client.get(f"{V1}/devices/mine", headers=other).json() == []
    r = client.patch(f"{V1}/devices/{dev_id}", headers=other, json={"name": "x"})
    assert r.status_code == 404


def test_runtime_manifest_and_bundle(client: TestClient, user_headers: dict[str, str]) -> None:
    _, dev_h = register_device(client, user_headers)
    assert client.get(f"{V1}/runtime/manifest").status_code == 401
    m = client.get(f"{V1}/runtime/manifest", headers=dev_h).json()
    assert m["pyodide_version"] == "314.0.7" and m["numpy_version"] == "2.4.6"
    b = client.get(f"{V1}/runtime/kernels/1", headers=dev_h)
    assert hashlib.sha256(b.content).hexdigest() == m["kernel_bundle_sha256"]
    names = zipfile.ZipFile(io.BytesIO(b.content)).namelist()
    assert "proofnet_kernels/core/gaussian_nb.py" in names
    assert not any("server" in n for n in names)  # server code never ships to workers
    assert client.get(f"{V1}/runtime/kernels/99", headers=dev_h).status_code == 404


def test_session_heartbeat_and_offline(
    client: TestClient, user_headers: dict[str, str], mongo: Mongo, settings: Settings
) -> None:
    dev_id, dev_h = register_device(client, user_headers)
    sid = client.post(f"{V1}/worker/session", headers=dev_h, json=SESSION_BODY).json()["session_id"]
    devices = mongo[settings.mongodb_db]["devices"]

    def get() -> dict[str, Any]:
        d = devices.find_one({"_id": dev_id})
        assert d is not None
        return d

    assert get()["status"] == "idle"
    hb = client.post(
        f"{V1}/worker/heartbeat", headers=dev_h, json={"session_id": sid, "state": "idle"}
    )
    assert hb.status_code == 200 and hb.json()["directives"] == []
    assert hb.json()["next_heartbeat_ms"] == 2000
    seen = get()["last_seen_at"]

    bad = client.post(
        f"{V1}/worker/heartbeat", headers=dev_h, json={"session_id": "nope", "state": "idle"}
    )
    assert bad.json()["directives"] == [{"type": "reregister"}]

    async def offline_pass() -> int:
        db = await connect(settings)
        try:
            return await reconciler.mark_offline_devices(db, settings)
        finally:
            await db.close()

    # 15 s after the last heartbeat: still online; 25 s: offline (20 s threshold; margin for connect time)
    devices.update_one(
        {"_id": dev_id}, {"$set": {"last_seen_at": utcnow() - timedelta(seconds=15)}}
    )
    asyncio.run(offline_pass())
    assert get()["status"] == "idle"
    devices.update_one(
        {"_id": dev_id}, {"$set": {"last_seen_at": utcnow() - timedelta(seconds=25)}}
    )
    assert asyncio.run(offline_pass()) >= 1
    assert get()["status"] == "offline"
    assert asyncio.run(offline_pass()) == 0  # idempotent: nothing left to change
    events = mongo[settings.mongodb_db]["events"]
    assert events.count_documents({"device_id": dev_id, "type": "device_offline"}) == 1

    # the next heartbeat of the same session brings it back to idle
    client.post(f"{V1}/worker/heartbeat", headers=dev_h, json={"session_id": sid, "state": "idle"})
    d = get()
    assert d["status"] == "idle" and d["last_seen_at"] > seen


def test_background_reconciler_marks_offline(mongo: Mongo) -> None:
    """The real loop (not just a direct call) runs inside the app and flips a stale device."""
    s = Settings(
        mongodb_db=f"proofnet_test_loop_{int(time.time())}",
        jwt_secret="test-secret-test-secret-test-secret-123",
        status_cache_seconds=0,
        offline_after_seconds=1,
        reconciler_interval_seconds=1,
    )
    try:
        with TestClient(create_app(s)) as c:
            h = signup(c)
            _, dev_h = register_device(c, h)
            sid = c.post(f"{V1}/worker/session", headers=dev_h, json=SESSION_BODY).json()[
                "session_id"
            ]
            c.post(
                f"{V1}/worker/heartbeat", headers=dev_h, json={"session_id": sid, "state": "idle"}
            )
            deadline = time.time() + 20
            status = "idle"
            while time.time() < deadline and status != "offline":
                time.sleep(0.5)
                status = c.get(f"{V1}/devices/mine", headers=h).json()[0]["status"]
            assert status == "offline"
    finally:
        mongo.drop_database(s.mongodb_db)


def test_disable_and_enable_device(client: TestClient, user_headers: dict[str, str]) -> None:
    dev_id, dev_h = register_device(client, user_headers)
    client.post(f"{V1}/worker/session", headers=dev_h, json=SESSION_BODY)
    r = client.patch(f"{V1}/devices/{dev_id}", headers=user_headers, json={"disabled": True})
    assert r.json()["status"] == "disabled"
    assert client.post(f"{V1}/worker/session", headers=dev_h, json=SESSION_BODY).status_code == 403
    r = client.patch(
        f"{V1}/devices/{dev_id}", headers=user_headers, json={"disabled": False, "name": "renamed"}
    )
    assert r.json()["status"] == "initializing" and r.json()["name"] == "renamed"


# ---------- datasets & tasks ----------
def _csv(n: int = 400, classes: int = 3) -> bytes:
    rng = np.random.default_rng(0)
    lines = ["f0,f1,f2,label"]
    for i in range(n):
        y = i % classes
        a, b, c = rng.normal(size=3) + y
        lines.append(f"{a:.5f},{b:.5f},{c:.5f},{y}")
    return ("\n".join(lines) + "\n").encode()


def _upload(client: TestClient, h: dict[str, str], data: bytes) -> Any:
    return client.post(f"{V1}/datasets", headers=h, files={"file": ("d.csv", data, "text/csv")})


def _manifest(ds_id: str, **over: object) -> dict[str, object]:
    m: dict[str, object] = {
        "name": "t1",
        "task_type": "gaussian_nb_train",
        "dataset_id": ds_id,
        "params": {"target_column": "label", "feature_columns": ["f0", "f1", "f2"]},
        "execution": {"min_devices": 1, "max_devices": 2},
    }
    m.update(over)
    return m


def test_dataset_upload_profile_and_limits(
    client: TestClient, user_headers: dict[str, str]
) -> None:
    r = _upload(client, user_headers, _csv())
    assert r.status_code == 201
    ds = r.json()
    assert ds["profile"]["n_rows"] == 400 and len(ds["sha256"]) == 64
    assert {c["name"] for c in ds["profile"]["columns"]} == {"f0", "f1", "f2", "label"}
    assert client.get(f"{V1}/datasets/{ds['id']}", headers=user_headers).status_code == 200
    assert client.get(f"{V1}/datasets/{ds['id']}", headers=signup(client)).status_code == 404
    assert _upload(client, user_headers, b"").status_code == 422
    assert _upload(client, user_headers, b"\xff\xfe\x00garbage").status_code == 422


def test_upload_size_limit(
    client: TestClient, user_headers: dict[str, str], settings: Settings
) -> None:
    big = b"a,b\n" + b"1,2\n" * (settings.max_upload_mb * 1024 * 1024 // 4 + 10)
    r = _upload(client, user_headers, big)
    assert r.status_code == 413 and r.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


def test_task_types_catalog(client: TestClient, user_headers: dict[str, str]) -> None:
    types = client.get(f"{V1}/task-types", headers=user_headers).json()
    assert {t["task_type"] for t in types} == {"gaussian_nb_train", "linear_ridge_train"}
    assert all(t["params_schema"]["properties"] for t in types)


def test_validate_reports_errors(client: TestClient, user_headers: dict[str, str]) -> None:
    ds = _upload(client, user_headers, _csv()).json()
    ok = client.post(f"{V1}/tasks/validate", headers=user_headers, json=_manifest(ds["id"])).json()
    assert ok["ok"] and ok["errors"] == []
    pv = ok["plan_preview"]  # plan preview from the current device pool (P5)
    assert pv["estimated"] is True and pv["n_train"] == 320 and "shares" in pv
    bad = _manifest(ds["id"], params={"target_column": "nope", "feature_columns": ["f0"]})
    r = client.post(f"{V1}/tasks/validate", headers=user_headers, json=bad).json()
    assert not r["ok"] and any("not found" in e for e in r["errors"])
    wrong = _manifest(
        ds["id"], params={"target_column": "label", "feature_columns": ["f0"], "alpha": 1.0}
    )
    assert client.post(f"{V1}/tasks/validate", headers=user_headers, json=wrong).status_code == 422


def test_create_task_prepares_npz_with_matching_sha(
    client: TestClient, user_headers: dict[str, str], mongo: Mongo, settings: Settings
) -> None:
    ds = _upload(client, user_headers, _csv(500)).json()
    r = client.post(f"{V1}/tasks", headers=user_headers, json=_manifest(ds["id"]))
    assert r.status_code == 201, r.text
    t = r.json()
    assert t["status"] == "queued" and t["verification_policy"] == {"mode": "adaptive"}
    assert t["prepared"]["n_train"] + t["prepared"]["n_test"] == 500
    assert t["prepared"]["class_labels"] == [0, 1, 2]
    # the prepared .npz really exists in GridFS and its SHA-256 matches
    mdb = mongo[settings.mongodb_db]
    task_doc = mdb["tasks"].find_one({"_id": t["id"]})
    assert task_doc is not None
    buf = io.BytesIO()
    gridfs.GridFSBucket(mdb).download_to_stream(task_doc["prepared"]["file_id"], buf)
    assert hashlib.sha256(buf.getvalue()).hexdigest() == t["prepared"]["sha256"]
    z = np.load(io.BytesIO(buf.getvalue()), allow_pickle=False)
    assert z["X_train"].shape == (t["prepared"]["n_train"], 3)
    # listing / detail / ownership
    assert client.get(f"{V1}/tasks", headers=user_headers).json()[0]["id"] == t["id"]
    assert client.get(f"{V1}/tasks/{t['id']}", headers=user_headers).status_code == 200
    assert client.get(f"{V1}/tasks/{t['id']}", headers=signup(client)).status_code == 404
    assert mdb["events"].count_documents({"task_id": t["id"], "type": "task_created"}) == 1


def test_create_task_rejects_invalid(client: TestClient, user_headers: dict[str, str]) -> None:
    ds = _upload(client, user_headers, _csv(50)).json()  # too few rows
    r = client.post(f"{V1}/tasks", headers=user_headers, json=_manifest(ds["id"]))
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_FAILED"
    assert client.get(f"{V1}/tasks", headers=user_headers).json() == []  # never becomes a task
    missing = client.post(f"{V1}/tasks", headers=user_headers, json=_manifest("ds_missing"))
    assert missing.status_code == 404


def test_ridge_task_creation(client: TestClient, user_headers: dict[str, str]) -> None:
    rng = np.random.default_rng(1)
    rows = ["a,b,y"] + [
        f"{x:.4f},{z:.4f},{2 * x - z + 1:.4f}" for x, z in rng.normal(size=(300, 2))
    ]
    ds = _upload(client, user_headers, ("\n".join(rows) + "\n").encode()).json()
    m = _manifest(
        ds["id"],
        task_type="linear_ridge_train",
        params={"target_column": "y", "feature_columns": ["a", "b"]},
    )
    r = client.post(f"{V1}/tasks", headers=user_headers, json=m)
    assert r.status_code == 201 and r.json()["prepared"]["class_labels"] == []


def test_pickle_never_accepted_as_dataset(client: TestClient, user_headers: dict[str, str]) -> None:
    assert _upload(client, user_headers, pickle.dumps({"a": 1})).status_code == 422


# ---------- admin demo reset ----------
def test_admin_demo_reset_clears_work_data_but_keeps_users_and_devices() -> None:
    """Own database: the reset wipes everything, so it must not share one with other tests."""
    import secrets

    admin_email = f"admin{secrets.token_hex(3)}@example.com"
    base = Settings()
    s = Settings(
        mongodb_uri=base.mongodb_uri,
        mongodb_db=f"proofnet_test_admin_{secrets.token_hex(4)}",
        jwt_secret="test-secret-test-secret-test-secret-123",
        status_cache_seconds=0,
        reconciler_interval_seconds=3600,
        admin_emails=admin_email,
    )
    mongo: Mongo = MongoClient(s.mongodb_uri, tz_aware=True)
    try:
        with TestClient(create_app(s)) as c:
            boss = signup(c, admin_email)
            other = signup(c)
            assert c.get(f"{V1}/admin/me", headers=boss).json() == {"admin": True}
            assert c.get(f"{V1}/admin/me", headers=other).json() == {"admin": False}
            dev_id, _ = register_device(c, boss)
            ds = _upload(c, boss, _csv()).json()
            assert c.post(f"{V1}/tasks", headers=boss, json=_manifest(ds["id"])).status_code == 201
            mdb = mongo[s.mongodb_db]
            assert (
                mdb["datasets"].count_documents({}) == 1 and mdb["tasks"].count_documents({}) == 1
            )
            assert c.post(f"{V1}/admin/demo/reset", headers=other).status_code == 403
            assert c.post(f"{V1}/admin/demo/reset").status_code == 401
            r = c.post(f"{V1}/admin/demo/reset", headers=boss)
            assert r.status_code == 200 and r.json()["reset"] is True
            assert r.json()["deleted"]["tasks"] == 1 and r.json()["deleted"]["files"] >= 2
            for coll in (
                "tasks",
                "datasets",
                "chunks",
                "assignments",
                "artifacts",
                "partial_results",
            ):
                assert mdb[coll].count_documents({}) == 0, coll
            assert mdb["fs.files"].count_documents({}) == 0
            # accounts and devices survive: phones stay registered, the user can still log in
            assert mdb["devices"].find_one({"_id": dev_id}) is not None
            assert c.get(f"{V1}/auth/me", headers=boss).status_code == 200
    finally:
        mongo.drop_database(s.mongodb_db)
        mongo.close()
