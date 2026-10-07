"""Image CNN training end to end through the API: iterative rounds across devices, per-round
gradient aggregation, centralized verification, failure handling inside training. CLI workers do the
real computation (same core/cnn.py kernel the phones run)."""

import io
import time
import zipfile
from typing import Any

import numpy as np
import pytest
from api_helpers import signup
from e2e_helpers import (
    V1,
    Env,
    assert_one_active_assignment_per_device,
    drive_many,
    make_workers,
    only_these,
    set_scores,
    status,
)
from PIL import Image

from cli_worker.worker import Faults
from proofnet_kernels.core import cnn as core
from proofnet_kernels.server import cnn as srv

CLASSES = ("hbar", "vbar", "square")


def synth_zip(per_class: int = 60, side: int = 28, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name in CLASSES:
            for i in range(per_class):
                a = rng.integers(0, 40, size=(side, side)).astype(np.uint8)
                o = int(rng.integers(3, 8))
                if name == "hbar":
                    a[o : o + 4, 4 : side - 4] = 230
                elif name == "vbar":
                    a[4 : side - 4, o : o + 4] = 230
                else:
                    a[o : o + 12, o : o + 12] = 230
                b = io.BytesIO()
                Image.fromarray(a).save(b, format="PNG")
                z.writestr(f"train/{name}/{i:03d}.png", b.getvalue())
    return buf.getvalue()


def upload_images(client: Any, h: dict[str, str], data: bytes | None = None) -> dict[str, Any]:
    r = client.post(
        f"{V1}/image-datasets",
        headers=h,
        files={"file": ("shapes.zip", data or synth_zip(), "application/zip")},
    )
    assert r.status_code == 201, r.text
    return dict(r.json())


def image_manifest(ds_id: str, **over: Any) -> dict[str, Any]:
    params = {
        "steps": 14, "global_batch_size": 32, "learning_rate": 0.05, "conv1_filters": 4,
        "conv2_filters": 4, "dense_units": 16, "verify_rounds": 3,
    }  # fmt: skip
    params.update(over.pop("params", {}))
    m: dict[str, Any] = {
        "name": "cnn-test",
        "task_type": "cnn_image_train",
        "dataset_id": ds_id,
        "params": params,
        "execution": {"min_devices": 1, "max_devices": 4},
    }
    m.update(over)
    return m


def run_training(
    env: Env, workers: list[Any], h: dict[str, str], m: dict[str, Any]
) -> dict[str, Any]:
    client, _, _ = env
    r = client.post(f"{V1}/image-tasks", headers=h, json=m)
    assert r.status_code == 201, r.text
    tid = r.json()["id"]
    drive_many(
        workers,
        lambda: status(client, h, tid)["task"]["status"] in ("completed", "failed", "cancelled"),
        timeout=180,
    )
    return status(client, h, tid)


def central_replay(task: dict[str, Any], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The same training run on one machine (whole batches), for the final-weights comparison."""
    p = task["params"]
    arch = {
        "input": task["prepared"]["image_shape"], "conv1": p["conv1_filters"],
        "conv2": p["conv2_filters"], "dense": p["dense_units"],
        "classes": len(task["prepared"]["class_labels"]),
    }  # fmt: skip
    w = core.init_weights(arch, p["init_seed"])
    v = np.zeros_like(w)
    for step in range(p["steps"]):
        idx = srv.batch_indices(len(x), p["global_batch_size"], step, p["split_seed"] + 7)
        _, _, g = core.loss_and_grad_sum(w, x[idx], y[idx], arch)
        w, v = srv.sgd_step(
            w, v, g.astype(np.float64), p["global_batch_size"], p["learning_rate"], p["momentum"]
        )
    return w


# ---------- ingestion API ----------
def test_image_dataset_upload_profile_and_errors(env: Env) -> None:
    client, _, _ = env
    h = signup(client)
    ds = upload_images(client, h)
    prof = ds["profile"]
    assert prof["n_images"] == 180 and prof["shape"] == [28, 28, 1]
    assert prof["classes"] == sorted(CLASSES) and prof["class_counts"] == [60, 60, 60]
    assert len(prof["samples"]) == 24 and prof["samples"][0]["png_b64"]
    assert client.get(f"{V1}/image-datasets/{ds['id']}", headers=h).status_code == 200
    assert [d["id"] for d in client.get(f"{V1}/image-datasets", headers=h).json()] == [ds["id"]]
    assert client.get(f"{V1}/image-datasets/{ds['id']}", headers=signup(client)).status_code == 404
    bad = client.post(
        f"{V1}/image-datasets",
        headers=h,
        files={"file": ("x.zip", b"not a zip", "application/zip")},
    )
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "INVALID_IMAGES"
    big = b"\0" * (26 * 1024 * 1024)
    over = client.post(
        f"{V1}/image-datasets", headers=h, files={"file": ("b.zip", big, "application/zip")}
    )
    assert over.status_code == 413


def test_image_task_validation(env: Env) -> None:
    client, _, _ = env
    h = signup(client)
    ds = upload_images(client, h)
    ok = client.post(f"{V1}/image-tasks/validate", headers=h, json=image_manifest(ds["id"])).json()
    assert ok["ok"] and ok["summary"]["n_params"] > 0
    assert ok["plan_preview"]["kind"] == "iterative" and ok["plan_preview"]["steps"] == 14
    too_big = image_manifest(ds["id"], params={"global_batch_size": 512})
    r = client.post(f"{V1}/image-tasks/validate", headers=h, json=too_big).json()
    assert not r["ok"] and any("global_batch_size" in e for e in r["errors"])
    assert client.post(f"{V1}/image-tasks", headers=h, json=too_big).status_code == 422
    assert (
        client.post(f"{V1}/image-tasks", headers=h, json=image_manifest("img_missing")).status_code
        == 404
    )
    # the CSV endpoint does not accept image manifests (separate pipelines)
    csv_try = client.post(f"{V1}/tasks", headers=h, json=image_manifest(ds["id"]))
    assert csv_try.status_code == 422


# ---------- the main property ----------
def test_cnn_training_across_two_devices_equals_centralized(env: Env) -> None:
    client, mongo, s = env
    h = signup(client)
    a, b = make_workers(client, "cnn", 2)
    only_these(env, [a, b])
    set_scores(env, [a, b], [3.0e6, 1.0e6])  # A is 3x faster -> gets ~3/4 of every batch
    ds = upload_images(client, h)
    m = image_manifest(ds["id"], execution={"min_devices": 2, "max_devices": 2})
    st = run_training(env, [a, b], h, m)
    assert st["task"]["status"] == "completed", st["task"]["error"]
    tid = st["task"]["id"]

    # --- real iterative execution: 14 rounds x 2 chunks, both devices every round ---
    tr = client.get(f"{V1}/tasks/{tid}/training", headers=h).json()
    assert tr["steps"] == 14 and tr["round"] == 14 and tr["state"] == "done"
    assert [r["round"] for r in tr["rounds"]] == list(range(14))
    for r in tr["rounds"]:
        assert r["n"] == 32 and len(r["devices"]) == 2
        rows = {d["name"]: d["rows"] for d in r["devices"]}
        assert rows[a.name] + rows[b.name] == 32 and rows[a.name] > rows[b.name]  # faster gets more
        assert rows[a.name] == 24 and rows[b.name] == 8  # 3:1 of 32 (weights = score / sum)
    # the live status shows the last round only (bounded payload); the database holds all 28 chunks
    assert [c["status"] for c in st["chunks"]] == ["completed", "completed"]
    assert (
        mongo[s.mongodb_db]["chunks"].count_documents({"task_id": tid, "status": "completed"}) == 28
    )
    assert {x["device_name"] for x in st["assignments"]} == {a.name, b.name}
    assert_one_active_assignment_per_device(env)

    # --- distributed == centralized: sampled-round gradient checks + full replay ---
    ref = st["task"]["result"]["reference_check"]
    assert ref["passed"] is True and ref["verified_rounds"] == [0, 6, 13]
    assert ref["max_relative_difference"] < 1e-4
    assert tr["verified_rounds"] == [0, 6, 13]
    arts = {x["kind"]: x for x in st["artifacts"]}
    assert set(arts) == {
        "model_npz", "model_json", "inference_py", "predictions_csv", "training_curve_csv", "report_json",
    }  # fmt: skip
    blob = client.get(f"{V1}/artifacts/{arts['model_npz']['id']}/download", headers=h).content
    z = np.load(io.BytesIO(blob), allow_pickle=False)
    # replay centrally from the same prepared data
    from proofnet_api.tasks.prepared import load_task_images  # noqa: F401 (documented below)

    task_doc = mongo[s.mongodb_db]["tasks"].find_one({"_id": tid})
    assert task_doc is not None
    import gridfs

    buf = io.BytesIO()
    gridfs.GridFSBucket(mongo[s.mongodb_db]).download_to_stream(
        task_doc["prepared"]["file_id"], buf
    )
    with np.load(io.BytesIO(buf.getvalue()), allow_pickle=False) as d:
        x, y = d["X_train"], d["y_train"]
    w_central = central_replay(st["task"], x, y)
    rel = np.abs(z["weights"] - w_central).max() / np.abs(w_central).max()
    assert rel < 1e-3, rel

    # --- the model learned, and the report is honest and complete ---
    metrics = st["task"]["result"]["metrics"]
    assert metrics["accuracy"] > 0.6 and metrics["n_test"] == 27  # 180 images, 15% holdout
    rep = client.get(f"{V1}/artifacts/{arts['report_json']['id']}/download", headers=h).json()
    assert rep["architecture"]["classes"] == 3 and rep["training"]["steps"] == 14
    by_dev = {c["device_name"]: c for c in rep["contributions"]}
    assert by_dev[a.name]["rows"] == 24 * 14 and by_dev[b.name]["rows"] == 8 * 14
    assert any("saw the training images" in x for x in rep["limitations"])
    assert rep["metrics"]["computed_by"].startswith("aggregator")
    curve = client.get(
        f"{V1}/artifacts/{arts['training_curve_csv']['id']}/download", headers=h
    ).text
    assert curve.splitlines()[0].startswith("round,loss") and len(curve.splitlines()) == 15
    inf = client.get(f"{V1}/artifacts/{arts['inference_py']['id']}/download", headers=h).text
    assert "def predict_logits" in inf and "__main__" in inf
    assert tr["rounds"][-1]["loss"] < tr["rounds"][0]["loss"]


def test_training_works_with_one_device_and_adapts_when_a_second_joins(env: Env) -> None:
    client, _, _ = env
    h = signup(client)
    (a,) = make_workers(client, "solo", 1)
    only_these(env, [a])
    ds = upload_images(client, h, synth_zip(seed=5))
    st = run_training(env, [a], h, image_manifest(ds["id"], params={"steps": 6}))
    assert st["task"]["status"] == "completed", st["task"]["error"]
    tid = st["task"]["id"]
    tr = client.get(f"{V1}/tasks/{tid}/training", headers=h).json()
    assert all(len(r["devices"]) == 1 and r["devices"][0]["rows"] == 32 for r in tr["rounds"])


# ---------- failures inside training ----------
def test_worker_dying_mid_round_is_replaced_and_training_still_matches(env: Env) -> None:
    client, mongo, s = env
    h = signup(client)
    a, b = make_workers(client, "cdie", 2)
    only_these(env, [a, b])
    set_scores(env, [a, b], [3.0e6, 1.0e6])
    a.faults = Faults(die_after_start=True)  # the fast device vanishes in the first round
    ds = upload_images(client, h, synth_zip(seed=6))
    m = image_manifest(
        ds["id"], params={"steps": 6}, execution={"min_devices": 1, "max_devices": 2}
    )
    st = run_training(env, [a, b], h, m)
    assert st["task"]["status"] == "completed", st["task"]["error"]
    asgs = st["assignments"]
    assert any(x["status"] == "expired" and x["device_name"] == a.name for x in asgs)
    tid = st["task"]["id"]
    tr = client.get(f"{V1}/tasks/{tid}/training", headers=h).json()
    assert [r["round"] for r in tr["rounds"]] == list(range(6))  # every round closed exactly once
    # the rounds after the loss ran on the surviving device alone, with full batches
    assert all(sum(d["rows"] for d in r["devices"]) == 32 for r in tr["rounds"])
    assert st["task"]["result"]["reference_check"]["passed"] is True


def test_corrupt_gradient_is_rejected_inside_training(env: Env) -> None:
    client, mongo, s = env
    h = signup(client)
    a, b = make_workers(client, "ccor", 2)
    only_these(env, [a, b])
    set_scores(env, [a, b], [3.0e6, 1.0e6])
    a.faults = Faults(corrupt_result=True)
    ds = upload_images(client, h, synth_zip(seed=7))
    m = image_manifest(
        ds["id"], params={"steps": 5}, execution={"min_devices": 1, "max_devices": 2}
    )
    st = run_training(env, [a, b], h, m)
    assert st["task"]["status"] == "completed", st["task"]["error"]
    assert any(x["status"] == "rejected" for x in st["assignments"])
    dev = mongo[s.mongodb_db]["devices"].find_one({"_id": a.device_id})
    assert dev is not None and dev["stats"]["invalid_results"] >= 1


def test_scaled_gradient_passes_structure_but_fails_the_centralized_check(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A well-formed but wrong gradient (Part 2's threat model) is caught by the sampled
    centralized verification: the report says so instead of silently trusting it."""
    import base64

    client, _, _ = env
    h = signup(client)
    (a,) = make_workers(client, "cheat", 1)
    only_these(env, [a])
    original = core.map

    def cheating(
        x: np.ndarray, y: np.ndarray, w: np.ndarray, arch: dict[str, Any]
    ) -> dict[str, Any]:
        p = original(x, y, w, arch)
        g = core.decode_vector(p["grad_b64"], "<f4") * np.float32(1.5)
        p["grad_b64"] = base64.b64encode(g.astype("<f4").tobytes()).decode()
        return p

    monkeypatch.setattr(core, "map", cheating)  # the CLI worker calls core.map at run time
    ds = upload_images(client, h, synth_zip(seed=8))
    st = run_training(env, [a], h, image_manifest(ds["id"], params={"steps": 5}))
    assert st["task"]["status"] == "completed"
    ref = st["task"]["result"]["reference_check"]
    assert ref["passed"] is False and ref["max_relative_difference"] > 0.1


def test_cancel_training_mid_run(env: Env) -> None:
    client, _, _ = env
    h = signup(client)
    a, b = make_workers(client, "ccan", 2)
    only_these(env, [a, b])
    ds = upload_images(client, h, synth_zip(seed=9))
    r = client.post(
        f"{V1}/image-tasks", headers=h, json=image_manifest(ds["id"], params={"steps": 200})
    )
    tid = r.json()["id"]
    drive_many(
        [a, b],
        lambda: (
            status(client, h, tid)["task"]["status"] == "running"
            and len(status(client, h, tid)["assignments"]) >= 2
        ),
    )
    assert client.post(f"{V1}/tasks/{tid}/cancel", headers=h).json()["status"] == "cancelled"
    st = status(client, h, tid)
    assert st["task"]["status"] == "cancelled"
    assert all(c["status"] in ("completed", "cancelled") for c in st["chunks"])
    # the devices are free for new work
    ds2 = upload_images(client, h, synth_zip(seed=10))
    st2 = run_training(env, [a, b], h, image_manifest(ds2["id"], params={"steps": 5}))
    assert st2["task"]["status"] == "completed"


def test_stuck_round_is_resumed_by_the_reconciler(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the request path never closes the round (process died after the last chunk), the
    reconciler closes it: training still completes and every round closes exactly once."""
    from proofnet_api.worker_gateway import assignments

    class NoRequestPathClose:
        """The gateway's view of the training module, except it never closes rounds itself."""

        def __getattr__(self, name: str) -> Any:
            return getattr(training_service, name)

        @staticmethod
        async def advance_training(*_: Any, **__: Any) -> bool:
            return False

    from proofnet_api.training import service as training_service

    monkeypatch.setattr(assignments, "training", NoRequestPathClose())
    client, _, _ = env
    h = signup(client)
    (a,) = make_workers(client, "cres", 1)
    only_these(env, [a])
    ds = upload_images(client, h, synth_zip(seed=11))
    st = run_training(env, [a], h, image_manifest(ds["id"], params={"steps": 5}))
    assert st["task"]["status"] == "completed", st["task"]["error"]
    tr = client.get(f"{V1}/tasks/{st['task']['id']}/training", headers=h).json()
    assert [r["round"] for r in tr["rounds"]] == [0, 1, 2, 3, 4]


# ---------- start races (regression for a lost-round-0 bug found by repeated runs) ----------
def test_concurrent_starts_never_lose_round_zero(env: Env) -> None:
    import asyncio

    from proofnet_api.db import connect
    from proofnet_api.training.service import start_training

    client, mongo, s = env
    h = signup(client)
    (a,) = make_workers(client, "race", 1)
    only_these(env, [a])
    devs = mongo[s.mongodb_db]["devices"]
    devs.update_one({"_id": a.device_id}, {"$set": {"status": "disabled"}})  # keep the task queued
    ds = upload_images(client, h, synth_zip(seed=12))
    r = client.post(
        f"{V1}/image-tasks", headers=h, json=image_manifest(ds["id"], params={"steps": 5})
    )
    tid = r.json()["id"]
    devs.update_one({"_id": a.device_id}, {"$set": {"status": "idle"}})
    task = mongo[s.mongodb_db]["tasks"].find_one({"_id": tid})
    assert task is not None and task["status"] == "queued"

    async def race() -> list[bool]:
        db = await connect(s)
        try:
            return list(await asyncio.gather(*[start_training(db, s, task) for _ in range(4)]))
        finally:
            await db.close()

    results = asyncio.run(race())
    assert results.count(True) <= 1  # at most one winner (the reconciler may have won instead)
    time.sleep(1.5)
    chunks = list(mongo[s.mongodb_db]["chunks"].find({"task_id": tid, "round": 0}))
    assert len(chunks) == 1, [c["status"] for c in chunks]  # one intact set, nothing deleted
    st = status(client, h, tid)
    assert st["task"]["status"] == "running"
    drive_many([a], lambda: status(client, h, tid)["task"]["status"] == "completed")


def test_a_start_that_dies_after_the_claim_is_recovered(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import timedelta

    from proofnet_api.db import utcnow
    from proofnet_api.training import service

    client, mongo, s = env
    h = signup(client)
    (a,) = make_workers(client, "dead", 1)
    only_these(env, [a])
    calls: list[int] = []

    async def dies(*_: Any, **__: Any) -> None:
        calls.append(1)  # the process "crashes" right after the claim: no model, no chunks

    monkeypatch.setattr(service, "_open_first_round", dies)
    ds = upload_images(client, h, synth_zip(seed=13))
    tid = client.post(
        f"{V1}/image-tasks", headers=h, json=image_manifest(ds["id"], params={"steps": 5})
    ).json()["id"]
    end = time.time() + 20
    tasks = mongo[s.mongodb_db]["tasks"]
    while time.time() < end and not calls:
        a.heartbeat_once()
        time.sleep(0.2)
    assert calls, "the start never ran"
    doc = tasks.find_one({"_id": tid})
    assert doc is not None and doc["training"]["state"] == "starting"
    assert mongo[s.mongodb_db]["chunks"].count_documents({"task_id": tid}) == 0
    monkeypatch.undo()  # the "restarted process" has working code again
    tasks.update_one(
        {"_id": tid}, {"$set": {"training.starting_since": utcnow() - timedelta(seconds=120)}}
    )
    st = wait_for_completion(env, [a], h, tid)
    assert st["task"]["status"] == "completed", st["task"]["error"]


def wait_for_completion(
    env: Env, workers: list[Any], h: dict[str, str], tid: str
) -> dict[str, Any]:
    client, _, _ = env
    drive_many(
        workers,
        lambda: status(client, h, tid)["task"]["status"] in ("completed", "failed", "cancelled"),
        timeout=120,
    )
    return status(client, h, tid)
