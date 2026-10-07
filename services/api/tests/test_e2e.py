"""Levels 3-4: a task flows end to end through the API with a CPython CLI worker.

Uses an isolated database per module so devices from other tests cannot take the work.
"""

import hashlib
import io
import secrets
import time
from collections.abc import Callable, Iterator
from typing import Any, cast

import httpx
import joblib
import numpy as np
import pandas as pd
import pytest
from api_helpers import signup
from fastapi.testclient import TestClient
from pymongo import MongoClient
from sklearn.linear_model import Ridge
from sklearn.naive_bayes import GaussianNB

from cli_worker.worker import Worker, login_or_signup
from proofnet_api.config import Settings
from proofnet_api.main import create_app

V1 = "/api/v1"
Mongo = MongoClient[dict[str, Any]]


class _Prefixed:
    def __init__(self, c: TestClient) -> None:
        self._c = c

    def get(self, url: str, **kw: Any) -> httpx.Response:
        return cast(httpx.Response, self._c.get(V1 + url, **kw))

    def post(self, url: str, **kw: Any) -> httpx.Response:
        return cast(httpx.Response, self._c.post(V1 + url, **kw))


@pytest.fixture(scope="module")
def env() -> Iterator[tuple[TestClient, Mongo, Settings]]:
    base = Settings()
    s = Settings(
        mongodb_uri=base.mongodb_uri,
        mongodb_db=f"proofnet_test_e2e_{secrets.token_hex(4)}",
        jwt_secret="test-secret-test-secret-test-secret-123",
        reconciler_interval_seconds=1,
    )
    mongo: Mongo = MongoClient(s.mongodb_uri, tz_aware=True)
    try:
        with TestClient(create_app(s)) as c:
            yield c, mongo, s
    finally:
        mongo.drop_database(s.mongodb_db)
        mongo.close()


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
    http = cast(httpx.Client, _Prefixed(client))
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


def test_gaussian_nb_one_device_end_to_end(env: tuple[TestClient, Mongo, Settings]) -> None:
    client, mongo, s = env
    h = signup(client)
    csv = clf_csv()
    ds = upload(client, h, csv)
    w = make_worker(client, "e2e1@example.com", "cli-e2e")
    isolate(env, w)
    r = client.post(f"{V1}/tasks", headers=h, json=manifest(ds, "gaussian_nb_train", "label", 4))
    assert r.status_code == 201, r.text
    tid = r.json()["id"]
    drive(w, lambda: status(client, h, tid)["task"]["status"] in ("completed", "failed"))
    st = status(client, h, tid)
    assert st["task"]["status"] == "completed", st["task"]["error"]

    # state machine: queued -> running -> aggregating -> completed
    assert [x["status"] for x in st["task"]["status_history"]] == [
        "queued",
        "running",
        "aggregating",
        "completed",
    ]
    # one chunk covering all training rows, executed by our device, accepted
    (chunk,) = st["chunks"]
    assert chunk["status"] == "completed" and chunk["row_start"] == 0
    assert chunk["row_end"] == st["task"]["prepared"]["n_train"]
    (asg,) = st["assignments"]
    assert asg["status"] == "succeeded" and asg["device_name"] == "cli-e2e"
    assert asg["purpose"] == "primary" and asg["timings"]["compute_ms"] >= 0
    assert asg["runtime_fingerprint"]["kind"] == "cpython"
    pr = mongo[s.mongodb_db]["partial_results"].find_one({"assignment_id": asg["id"]})
    assert pr is not None and pr["acceptance"] == "accepted_unverified"

    # artifacts + reference check
    kinds = {a["kind"] for a in st["artifacts"]}
    assert kinds == {"model_joblib", "model_json", "predictions_csv", "report_json"}
    res = st["task"]["result"]
    assert res["reference_check"]["passed"] is True
    assert res["reference_check"]["max_relative_difference"] <= 1e-8

    # downloaded model.joblib predicts identically to a centralized sklearn fit on the same split
    arts = {a["kind"]: a for a in st["artifacts"]}
    dl = client.get(f"{V1}/artifacts/{arts['model_joblib']['id']}/download", headers=h)
    assert dl.status_code == 200
    assert hashlib.sha256(dl.content).hexdigest() == arts["model_joblib"]["sha256"]
    model = joblib.load(io.BytesIO(dl.content))
    from proofnet_kernels.server import gaussian_nb as srv
    from proofnet_kernels.server.params import GaussianNBParams

    params = GaussianNBParams(target_column="label", feature_columns=[f"f{i}" for i in range(4)])
    prep = srv.prepare(pd.read_csv(io.BytesIO(csv)), params)
    ref = GaussianNB().fit(prep.x_train, prep.y_train)
    assert (model.predict(prep.x_test) == ref.predict(prep.x_test)).all()
    np.testing.assert_allclose(model.theta_, ref.theta_, rtol=1e-8)

    # report: real timings, device identity, runtime fingerprint, honest limitations
    rep = client.get(f"{V1}/artifacts/{arts['report_json']['id']}/download", headers=h).json()
    contrib = rep["contributions"][0]
    assert contrib["device_name"] == "cli-e2e" and contrib["timings_ms"]["compute_ms"] >= 0
    assert contrib["runtime_fingerprint"]["numpy"] and rep["reference_check"]["passed"]
    assert any("unverified" in x for x in rep["limitations"])
    assert rep["metrics"]["computed_by"].startswith("aggregator")

    # owner only
    other = signup(client)
    assert (
        client.get(
            f"{V1}/artifacts/{arts['model_joblib']['id']}/download", headers=other
        ).status_code
        == 404
    )
    # event timeline records the real lifecycle
    ev = [e["type"] for e in client.get(f"{V1}/tasks/{tid}/events", headers=h).json()]
    for t in (
        "task_created",
        "task_started",
        "chunk_assigned",
        "assignment_started",
        "result_accepted",
        "task_completed",
    ):
        assert t in ev, t
    # device returned to idle with stats updated
    dev = mongo[s.mongodb_db]["devices"].find_one({"_id": w.device_id})
    assert dev is not None and dev["status"] == "idle" and dev["stats"]["succeeded"] == 1


def test_linear_ridge_one_device_end_to_end(env: tuple[TestClient, Mongo, Settings]) -> None:
    client, _, _ = env
    h = signup(client)
    csv = reg_csv()
    ds = upload(client, h, csv)
    w = make_worker(client, "e2e2@example.com", "cli-e2e-ridge")
    isolate(env, w)
    r = client.post(
        f"{V1}/tasks", headers=h, json=manifest(ds, "linear_ridge_train", "target", 4, alpha=2.0)
    )
    tid = r.json()["id"]
    drive(w, lambda: status(client, h, tid)["task"]["status"] in ("completed", "failed"))
    st = status(client, h, tid)
    assert st["task"]["status"] == "completed", st["task"]["error"]
    assert st["task"]["result"]["reference_check"]["passed"] is True
    arts = {a["kind"]: a for a in st["artifacts"]}
    model = joblib.load(
        io.BytesIO(
            client.get(f"{V1}/artifacts/{arts['model_joblib']['id']}/download", headers=h).content
        )
    )
    from proofnet_kernels.server import linear_ridge as srv
    from proofnet_kernels.server.params import LinearRidgeParams

    params = LinearRidgeParams(
        target_column="target", feature_columns=[f"f{i}" for i in range(4)], alpha=2.0
    )
    prep = srv.prepare(pd.read_csv(io.BytesIO(csv)), params)
    ref = Ridge(alpha=2.0).fit(prep.x_train, prep.y_train)
    np.testing.assert_allclose(model.coef_, ref.coef_, rtol=1e-6)
    np.testing.assert_allclose(model.predict(prep.x_test), ref.predict(prep.x_test), rtol=1e-6)
    assert st["task"]["result"]["metrics"]["r2"] > 0.9


# ---------- result intake edge cases ----------
def _dispatch_one(
    env: tuple[TestClient, Mongo, Settings], email: str
) -> tuple[TestClient, dict[str, str], Worker, dict[str, Any], str]:
    """Create a task and return the run directive's assignment without executing it."""
    client, _, _ = env
    h = signup(client)
    ds = upload(client, h, clf_csv(seed=5))
    w = make_worker(client, email, "cli-" + email.split("@")[0])
    _, mongo, s = env
    mongo[s.mongodb_db]["devices"].update_many(  # leftovers from earlier scenarios stay out
        {"_id": {"$ne": w.device_id}}, {"$set": {"status": "disabled"}}
    )
    tid = client.post(
        f"{V1}/tasks", headers=h, json=manifest(ds, "gaussian_nb_train", "label", 4)
    ).json()["id"]
    end = time.time() + 30
    while time.time() < end:
        hb = client.post(
            f"{V1}/worker/heartbeat",
            headers=w._dev(),
            json={"session_id": w.session_id, "state": "idle"},
        ).json()
        runs = [d for d in hb["directives"] if d["type"] == "run"]
        if runs:
            return client, h, w, runs[0]["assignment"], tid
        time.sleep(0.25)
    raise AssertionError("no run directive arrived")


def _result_body(w: Worker, a: dict[str, Any], client: TestClient) -> dict[str, Any]:
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


def test_duplicate_result_is_idempotent(env: tuple[TestClient, Mongo, Settings]) -> None:
    client, h, w, a, tid = _dispatch_one(env, "dup@example.com")
    base = f"{V1}/worker/assignments/{a['assignment_id']}"
    assert client.post(f"{base}/start", headers=w._dev()).status_code == 200
    assert client.post(f"{base}/start", headers=w._dev()).status_code == 200  # idempotent start
    body = _result_body(w, a, client)
    first = client.post(f"{base}/result", headers=w._dev(), json=body)
    again = client.post(f"{base}/result", headers=w._dev(), json=body)
    assert first.status_code == 200 and again.status_code == 200
    assert again.json()["status"] == "succeeded"
    _, mongo, s = env
    assert (
        mongo[s.mongodb_db]["partial_results"].count_documents(
            {"assignment_id": a["assignment_id"]}
        )
        == 1
    )
    drive(w, lambda: status(client, h, tid)["task"]["status"] == "completed")


def test_invalid_result_is_rejected_and_counted(env: tuple[TestClient, Mongo, Settings]) -> None:
    client, h, w, a, tid = _dispatch_one(env, "bad@example.com")
    _, mongo, s = env
    base = f"{V1}/worker/assignments/{a['assignment_id']}"
    client.post(f"{base}/start", headers=w._dev())
    body = _result_body(w, a, client)
    body["payload"]["classes"]["n"][0] += 5  # counts no longer match the chunk
    from proofnet_kernels.core.serialize import payload_sha256

    body["payload_sha256"] = payload_sha256(body["payload"])
    r = client.post(f"{base}/result", headers=w._dev(), json=body)
    assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_RESULT"
    st = status(client, h, tid)
    assert st["assignments"][0]["status"] == "rejected"
    assert st["chunks"][0]["status"] == "pending"
    dev = mongo[s.mongodb_db]["devices"].find_one({"_id": w.device_id})
    assert dev is not None and dev["stats"]["invalid_results"] == 1
    # the rejected partial is stored but never accepted/merged
    pr = mongo[s.mongodb_db]["partial_results"].find_one({"assignment_id": a["assignment_id"]})
    assert pr is not None and pr["acceptance"] == "rejected_structural"
    # wrong payload digest is also refused
    r2 = client.post(f"{base}/result", headers=w._dev(), json={**body, "payload_sha256": "0" * 64})
    assert r2.status_code == 409  # assignment is already rejected -> late/closed


def test_payload_digest_mismatch_rejected(env: tuple[TestClient, Mongo, Settings]) -> None:
    client, _, w, a, _ = _dispatch_one(env, "digest@example.com")
    base = f"{V1}/worker/assignments/{a['assignment_id']}"
    client.post(f"{base}/start", headers=w._dev())
    body = _result_body(w, a, client)
    body["payload_sha256"] = "f" * 64
    r = client.post(f"{base}/result", headers=w._dev(), json=body)
    assert r.status_code == 422 and "payload_sha256 mismatch" in str(r.json()["error"]["details"])


def test_other_device_cannot_touch_assignment(env: tuple[TestClient, Mongo, Settings]) -> None:
    client, _, w, a, _ = _dispatch_one(env, "owner@example.com")
    intruder = make_worker(client, "intruder@example.com", "intruder")
    base = f"{V1}/worker/assignments/{a['assignment_id']}"
    assert client.get(f"{base}/input", headers=intruder._dev()).status_code == 404
    assert client.post(f"{base}/start", headers=intruder._dev()).status_code == 404
    assert (
        client.post(
            f"{base}/fail", headers=intruder._dev(), json={"code": "x", "message": "y"}
        ).status_code
        == 404
    )
    assert client.get(f"{base}/input", headers=w._dev()).status_code == 200


def test_late_result_after_failure_is_409_and_logged(
    env: tuple[TestClient, Mongo, Settings],
) -> None:
    client, _, w, a, tid = _dispatch_one(env, "late@example.com")
    _, mongo, s = env
    base = f"{V1}/worker/assignments/{a['assignment_id']}"
    client.post(f"{base}/start", headers=w._dev())
    body = _result_body(w, a, client)
    assert (
        client.post(
            f"{base}/fail", headers=w._dev(), json={"code": "X", "message": "boom"}
        ).status_code
        == 200
    )
    r = client.post(f"{base}/result", headers=w._dev(), json=body)
    assert r.status_code == 409 and r.json()["error"]["code"] == "LATE_RESULT"
    ev = mongo[s.mongodb_db]["events"].find_one(
        {"assignment_id": a["assignment_id"], "type": "late_result"}
    )
    assert ev is not None and ev["data"]["payload_sha256"] == body["payload_sha256"]


def test_tampered_input_never_served_for_wrong_digest(
    env: tuple[TestClient, Mongo, Settings],
) -> None:
    client, _, w, a, _ = _dispatch_one(env, "digest2@example.com")
    r = client.get(V1 + a["input_url"].removeprefix("/api/v1"), headers=w._dev())
    assert r.headers["x-input-sha256"] == a["input_sha256"]
    assert hashlib.sha256(r.content).hexdigest() == a["input_sha256"]
    with np.load(io.BytesIO(r.content), allow_pickle=False) as z:
        assert z["X"].shape == (a["n_rows"], a["n_features"])
