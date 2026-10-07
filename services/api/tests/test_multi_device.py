"""Levels 5-6: several devices execute different chunks of one task and the merged result equals
centralized computation. CLI workers do real work; benchmark scores are fixed in the database so
the planner's inputs are known (the compute itself is never faked)."""

import io
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from api_helpers import signup
from e2e_helpers import (
    V1,
    Env,
    assert_one_active_assignment_per_device,
    clf_csv,
    drive_many,
    make_workers,
    manifest,
    only_these,
    reg_csv,
    set_scores,
    status,
    upload,
)
from sklearn.linear_model import Ridge
from sklearn.naive_bayes import GaussianNB

from proofnet_api.scheduling.planner import allocate_rows


def run_task(env: Env, workers: list[Any], m: dict[str, Any], h: dict[str, str]) -> dict[str, Any]:
    client, _, _ = env
    tid = client.post(f"{V1}/tasks", headers=h, json=m).json()["id"]
    drive_many(workers, lambda: status(client, h, tid)["task"]["status"] in ("completed", "failed"))
    return status(client, h, tid)


@pytest.mark.parametrize(
    ("kind", "scores"),
    [
        ("gnb", [3.0e6, 1.5e6]),
        ("ridge", [5.0e6, 3.0e6, 2.0e6]),
        ("gnb", [4.0e6, 3.0e6, 2.0e6, 1.0e6]),
    ],
)
def test_split_by_measured_capability_and_merge_equals_centralized(
    env: Env, kind: str, scores: list[float]
) -> None:
    client, _, _ = env
    n_dev = len(scores)
    h = signup(client)
    workers = make_workers(client, f"md{kind}{n_dev}", n_dev)
    only_these(env, workers)
    set_scores(env, workers, scores)
    if kind == "gnb":
        csv = clf_csv(n=12_000, d=4, seed=n_dev)
        m = manifest(upload(client, h, csv), "gaussian_nb_train", "label", 4)
    else:
        csv = reg_csv(n=12_000, d=4, seed=n_dev)
        m = manifest(upload(client, h, csv), "linear_ridge_train", "target", 4, alpha=1.5)
    m["execution"] = {"min_devices": n_dev, "max_devices": n_dev}

    st = run_task(env, workers, m, h)
    assert st["task"]["status"] == "completed", st["task"]["error"]

    # Level 5: every device computed a different chunk, row ranges tile [0, N)
    n_train = st["task"]["prepared"]["n_train"]
    chunks = st["chunks"]
    assert len(chunks) == n_dev and all(c["status"] == "completed" for c in chunks)
    pos = 0
    for c in chunks:
        assert c["row_start"] == pos
        pos = c["row_end"]
    assert pos == n_train
    done_by = {
        a["chunk_id"]: a["device_name"] for a in st["assignments"] if a["status"] == "succeeded"
    }
    assert len(set(done_by.values())) == n_dev

    # the faster device received proportionally more rows, and the plan says why
    expected = allocate_rows(n_train, scores)
    plan = st["task"]["plan"]
    assert [s["rows"] for s in plan["shares"]] == expected
    assert [c["n_rows"] for c in chunks] == expected  # one chunk per device, fastest first
    assert [s["device_name"] for s in plan["shares"]] == [w.name for w in workers]
    assert "proportion" in plan["explanation"]
    rows_by_name = {s["device_name"]: s["rows"] for s in plan["shares"]}
    assert rows_by_name[workers[0].name] > rows_by_name[workers[-1].name]
    assert abs(sum(s["weight"] for s in plan["shares"]) - 1.0) < 1e-9

    # Level 6: merged result equals centralized computation
    res = st["task"]["result"]
    assert res["reference_check"]["passed"] is True
    arts = {a["kind"]: a for a in st["artifacts"]}
    blob = client.get(f"{V1}/artifacts/{arts['model_joblib']['id']}/download", headers=h).content
    model = joblib.load(io.BytesIO(blob))
    if kind == "gnb":
        from proofnet_kernels.server import gaussian_nb as srv
        from proofnet_kernels.server.params import GaussianNBParams

        prep = srv.prepare(
            pd.read_csv(io.BytesIO(csv)),
            GaussianNBParams(target_column="label", feature_columns=[f"f{i}" for i in range(4)]),
        )
        ref = GaussianNB().fit(prep.x_train, prep.y_train)
        np.testing.assert_allclose(model.theta_, ref.theta_, rtol=1e-8)
        np.testing.assert_allclose(model.var_, ref.var_, rtol=1e-8)
        assert (model.predict(prep.x_test) == ref.predict(prep.x_test)).all()
    else:
        from proofnet_kernels.server import linear_ridge as srv_r
        from proofnet_kernels.server.params import LinearRidgeParams

        prep_r = srv_r.prepare(
            pd.read_csv(io.BytesIO(csv)),
            LinearRidgeParams(
                target_column="target", feature_columns=[f"f{i}" for i in range(4)], alpha=1.5
            ),
        )
        ref_r = Ridge(alpha=1.5).fit(prep_r.x_train, prep_r.y_train)
        np.testing.assert_allclose(model.coef_, ref_r.coef_, rtol=1e-6)
        np.testing.assert_allclose(
            model.predict(prep_r.x_test), ref_r.predict(prep_r.x_test), rtol=1e-6
        )
    assert_one_active_assignment_per_device(env)

    # the report records every device's contribution (plan vs actual inputs)
    rep = client.get(f"{V1}/artifacts/{arts['report_json']['id']}/download", headers=h).json()
    assert len(rep["contributions"]) == n_dev
    assert {c["device_name"] for c in rep["contributions"]} == {w.name for w in workers}
    assert all(c["timings_ms"]["compute_ms"] >= 0 for c in rep["contributions"])
    # events: one assignment/finish pair per chunk, in human-readable form
    ev = client.get(f"{V1}/tasks/{st['task']['id']}/events", headers=h).json()
    finished = [e["message"] for e in ev if e["type"] == "result_accepted"]
    assert len(finished) == n_dev and all(" finished chunk " in x for x in finished)


def test_min_devices_waits_for_enough_eligible_devices(env: Env) -> None:
    client, _, _ = env
    h = signup(client)
    a, b = make_workers(client, "wait", 2)
    only_these(env, [a, b])
    set_scores(env, [a, b], [2e6, 1e6])
    env[1][env[2].mongodb_db]["devices"].update_one(
        {"_id": b.device_id}, {"$set": {"status": "offline"}}
    )
    m = manifest(upload(client, h, clf_csv(n=6000, seed=3)), "gaussian_nb_train", "label", 4)
    m["execution"] = {"min_devices": 2, "max_devices": 2}
    tid = client.post(f"{V1}/tasks", headers=h, json=m).json()["id"]
    for _ in range(8):  # one eligible device only: the task must stay queued, with a reason
        a.heartbeat_once()
        time.sleep(0.3)
    st = status(client, h, tid)
    assert st["task"]["status"] == "queued" and st["chunks"] == []
    assert any("start policy needs 2" in r and "1 available" in r for r in st["waiting_reasons"])
    assert any(f"{b.name}: status is offline" in r for r in st["waiting_reasons"])
    # the second device comes online -> the task starts and uses both
    b.session_id = None
    b.start_session(b.load_runtime())
    drive_many([a, b], lambda: status(client, h, tid)["task"]["status"] in ("completed", "failed"))
    st = status(client, h, tid)
    assert st["task"]["status"] == "completed" and len(st["chunks"]) == 2


def test_busy_and_offline_devices_are_excluded_from_the_plan(env: Env) -> None:
    client, mongo, s = env
    h = signup(client)
    ws = make_workers(client, "excl", 3)
    only_these(env, ws)
    set_scores(env, ws, [3e6, 2e6, 1e6])
    devs = mongo[s.mongodb_db]["devices"]
    devs.update_one({"_id": ws[0].device_id}, {"$set": {"status": "offline"}})  # fastest is offline
    devs.update_one({"_id": ws[1].device_id}, {"$set": {"status": "busy"}})  # second is busy
    m = manifest(upload(client, h, clf_csv(n=6000, seed=4)), "gaussian_nb_train", "label", 4)
    m["execution"] = {"min_devices": 1, "max_devices": 4}
    tid = client.post(f"{V1}/tasks", headers=h, json=m).json()["id"]
    drive_many([ws[2]], lambda: status(client, h, tid)["task"]["status"] in ("completed", "failed"))
    st = status(client, h, tid)
    assert st["task"]["status"] == "completed"
    assert [x["device_name"] for x in st["task"]["plan"]["shares"]] == [ws[2].name]
    assert len(st["chunks"]) == 1


def test_memory_cap_splits_work_into_sequential_chunks_on_one_device(env: Env) -> None:
    client, mongo, s = env
    h = signup(client)
    (w,) = make_workers(client, "mem", 1)
    only_these(env, [w])
    # Pretend a small browser: ~0.2 MB budget -> the share needs several chunks
    mongo[s.mongodb_db]["devices"].update_one(
        {"_id": w.device_id},
        {"$set": {"runtime.kind": "pyodide", "capabilities.memory_gb_reported": 0.002}},
    )
    m = manifest(upload(client, h, clf_csv(n=10_000, seed=6)), "gaussian_nb_train", "label", 4)
    st = run_task(env, [w], m, h)
    assert st["task"]["status"] == "completed", st["task"]["error"]
    assert len(st["chunks"]) > 2, "memory cap should have produced several chunks"
    assert all(c["preferred_device_id"] == w.device_id for c in st["chunks"])
    assert st["task"]["result"]["reference_check"]["passed"] is True
    assert_one_active_assignment_per_device(env)  # chunks ran one after another


def test_plan_preview_in_validate_matches_the_real_plan(env: Env) -> None:
    client, _, _ = env
    h = signup(client)
    ws = make_workers(client, "prev", 2)
    only_these(env, ws)
    set_scores(env, ws, [3e6, 1.5e6])
    ds = upload(client, h, clf_csv(n=9000, seed=8))
    m = manifest(ds, "gaussian_nb_train", "label", 4)
    m["execution"] = {"min_devices": 2, "max_devices": 2}
    pv = client.post(f"{V1}/tasks/validate", headers=h, json=m).json()["plan_preview"]
    assert pv["estimated"] and pv["ready_to_start"] and pv["eligible_devices"] == 2
    assert [s["device_name"] for s in pv["shares"]] == [ws[0].name, ws[1].name]
    assert [s["rows"] for s in pv["shares"]] == allocate_rows(pv["n_train"], [3e6, 1.5e6])
    # not enough devices -> says so
    m["execution"] = {"min_devices": 3, "max_devices": 3}
    pv3 = client.post(f"{V1}/tasks/validate", headers=h, json=m).json()["plan_preview"]
    assert pv3["ready_to_start"] is False and "needs 3 eligible" in pv3["message"]
    # the real task's plan uses the same split
    m["execution"] = {"min_devices": 2, "max_devices": 2}
    st = run_task(env, ws, m, h)
    assert [s["rows"] for s in st["task"]["plan"]["shares"]] == [
        s["rows"] for s in pv["shares"]
    ] or st["task"]["prepared"]["n_train"] != pv["n_train"]


def test_status_snapshot_is_cached_about_one_second(env: Env) -> None:
    from fastapi.testclient import TestClient

    from proofnet_api.main import create_app

    _, _, s = env
    cached = s.model_copy(update={"status_cache_seconds": 5.0})
    with TestClient(create_app(cached)) as c:
        h = signup(c)
        ds = upload(c, h, clf_csv(n=1000, seed=2))
        tid = c.post(
            f"{V1}/tasks", headers=h, json=manifest(ds, "gaussian_nb_train", "label", 4)
        ).json()["id"]
        first = c.get(f"{V1}/tasks/{tid}/status", headers=h).json()["server_time"]
        again = c.get(f"{V1}/tasks/{tid}/status", headers=h).json()["server_time"]
        assert first == again  # served from the snapshot cache
        # ownership is still enforced on a cache hit
        assert c.get(f"{V1}/tasks/{tid}/status", headers=signup(c)).status_code == 404


def test_stress_eight_workers_ten_consecutive_tasks(env: Env) -> None:
    client, _, _ = env
    h = signup(client)
    ws = make_workers(client, "stress", 8)
    only_these(env, ws)
    csv = clf_csv(n=8000, d=4, seed=11)
    ds = upload(client, h, csv)
    t0 = time.time()
    participants: set[str] = set()
    for i in range(10):
        m = manifest(ds, "gaussian_nb_train", "label", 4)
        m["name"] = f"stress-{i}"
        m["execution"] = {"min_devices": 2, "max_devices": 8}
        st = run_task(env, ws, m, h)
        assert st["task"]["status"] == "completed", (i, st["task"]["error"])
        assert st["task"]["result"]["reference_check"]["passed"] is True
        assert len(st["chunks"]) >= 2
        participants |= {a["device_name"] for a in st["assignments"]}
    assert len(participants) >= 4  # work really was spread across devices
    assert_one_active_assignment_per_device(env)
    print(
        f"stress: 10 tasks over 8 workers in {time.time() - t0:.1f}s, {len(participants)} devices used"
    )
