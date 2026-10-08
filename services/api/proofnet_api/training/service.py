"""Iterative data-parallel training (ARCHITECTURE 4.5): rounds of chunks, gradient aggregation,
SGD update, sampled centralized verification.

A round = one global mini-batch split across the eligible devices in proportion to their measured
benchmark. The chunks of round r reference model version r. When every chunk of the round is
accepted, the backend adds the gradient sums, divides by the batch size, applies SGD+momentum and
creates round r+1. All transitions are conditional/idempotent and recoverable after a restart.
"""

import asyncio
import base64
import contextlib
import hashlib
from datetime import timedelta
from typing import Any

import numpy as np
from pymongo.errors import BulkWriteError, DuplicateKeyError

from proofnet_kernels.core import cnn as core
from proofnet_kernels.server import cnn as srv
from proofnet_kernels.server.common import deterministic_npz

from ..aggregation.service import aggregate_task
from ..config import Settings
from ..db import Db, utcnow
from ..events import emit
from ..ids import new_id
from ..scheduling.lifecycle import fail_task
from ..scheduling.planner import allocate_rows, device_score
from ..scheduling.scheduler import _eligible_devices, schedule_pass
from ..tasks.prepared import load_task_images

TASK_TYPE = core.KERNEL
CLOSING_STALE_SECONDS = 60
BATCH_SEED_OFFSET = 7


def is_iterative(task: dict[str, Any]) -> bool:
    return task.get("task_type") == TASK_TYPE


def _b64(a: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


def _unb64(s: str, dtype: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=dtype)


# ------------------------------------------------------------------ model states
async def save_state(db: Db, task_id: str, version: int, w: np.ndarray, v: np.ndarray) -> None:
    with contextlib.suppress(DuplicateKeyError):  # a recovered attempt recomputed the same state
        await db.col("model_states").insert_one(
            {
                "_id": f"{task_id}:v{version}",
                "task_id": task_id,
                "version": version,
                "weights_b64": _b64(w, "<f4"),
                "velocity_b64": _b64(v, "<f4"),
                "created_at": utcnow(),
            }
        )


async def load_state(db: Db, task_id: str, version: int) -> tuple[np.ndarray, np.ndarray]:
    doc = await db.col("model_states").find_one({"_id": f"{task_id}:v{version}"})
    if doc is None:
        raise RuntimeError(f"model state v{version} missing")
    return (
        _unb64(doc["weights_b64"], "<f4").astype(np.float32),
        _unb64(doc["velocity_b64"], "<f4").astype(np.float32),
    )


# ------------------------------------------------------------------ chunk inputs
def batch_idx(task: dict[str, Any], round_: int) -> np.ndarray:
    tr = task["training"]
    return srv.batch_indices(
        task["prepared"]["n_train"], tr["global_batch_size"], round_, tr["batch_seed"]
    )


def chunk_input_bytes(
    x_train: np.ndarray, y_train: np.ndarray, w: np.ndarray, idx: np.ndarray, a: int, b: int
) -> bytes:
    sel = idx[a:b]
    return deterministic_npz({"X": x_train[sel], "y": y_train[sel], "w": w.astype(np.float32)})


async def build_chunk_input(db: Db, task: dict[str, Any], chunk: dict[str, Any]) -> bytes:
    prepared = await load_task_images(db, task)
    w, _ = await load_state(db, task["_id"], chunk["round"])
    idx = batch_idx(task, chunk["round"])
    return await asyncio.to_thread(
        chunk_input_bytes,
        prepared.x_train,
        prepared.y_train,
        w,
        idx,
        chunk["row_start"],
        chunk["row_end"],
    )


def worker_params(task: dict[str, Any], chunk: dict[str, Any]) -> dict[str, Any]:
    return {"arch": task["training"]["arch"], "round": chunk["round"]}


def validate_partial(
    task: dict[str, Any], chunk: dict[str, Any], payload: dict[str, Any]
) -> list[str]:
    return srv.validate_partial(payload, chunk["n_rows"], task["training"]["n_params"])


# ------------------------------------------------------------------ planning a round
def split_batch(
    batch: int, devices: list[dict[str, Any]], max_devices: int
) -> list[tuple[dict[str, Any], int]]:
    """Global batch -> per-device slice sizes in proportion to benchmark (drop tiny shares)."""
    ordered = sorted(devices, key=lambda d: (-device_score(d), d["_id"]))
    k = max(1, min(len(ordered), max_devices, batch // srv.MIN_ROWS_PER_DEVICE))
    chosen = ordered[:k]
    rows = allocate_rows(batch, [device_score(d) for d in chosen])
    while len(chosen) > 1 and min(rows) < srv.MIN_ROWS_PER_DEVICE:
        chosen = chosen[:-1]  # the slowest device would get a trivial slice: drop it
        rows = allocate_rows(batch, [device_score(d) for d in chosen])
    return list(zip(chosen, rows, strict=True))


async def _make_round_chunks(
    db: Db, settings: Settings, task: dict[str, Any], round_: int, w: np.ndarray
) -> list[dict[str, Any]]:
    tr = task["training"]
    batch = tr["global_batch_size"]
    devices = await _eligible_devices(db, settings, task, rows=srv.MIN_ROWS_PER_DEVICE)
    split = split_batch(batch, devices, task["execution"]["max_devices"]) if devices else []
    if (
        not split
    ):  # nobody eligible right now: one chunk for the whole batch, assigned when possible
        split = [({"_id": None}, batch)]
    prepared = await load_task_images(db, task)
    idx = batch_idx(task, round_)
    d = task["prepared"]["n_features"]
    docs: list[dict[str, Any]] = []
    pos = 0
    for i, (dev, n) in enumerate(split):
        data = await asyncio.to_thread(
            chunk_input_bytes, prepared.x_train, prepared.y_train, w, idx, pos, pos + n
        )
        docs.append(
            {
                "_id": new_id("chk"),
                "task_id": task["_id"],
                "round": round_,
                "index": i,
                "role": "work",
                "row_start": pos,
                "row_end": pos + n,
                "n_rows": n,
                "work_units": n * d,
                "input_sha256": hashlib.sha256(data).hexdigest(),
                "preferred_device_id": dev["_id"],
                "status": "created",  # invisible to the scheduler until the round is released
                "attempt_count": 0,
                "max_attempts": settings.max_attempts,
                "excluded_device_ids": [],
                "accepted_assignment_id": None,
                "created_at": utcnow(),
                "pending_since": utcnow(),
            }
        )
        pos += n
    with contextlib.suppress(BulkWriteError):  # already created by a concurrent/recovered attempt
        await db.col("chunks").insert_many(docs, ordered=False)
    return docs


def plan_summary(
    task: dict[str, Any], docs: list[dict[str, Any]], devices_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    tr = task["training"]
    total = sum(c["n_rows"] for c in docs) or 1
    return {
        "kind": "iterative",
        "planner": "WeightedProportionalPlanner (per round)",
        "explanation": (
            "every round the global mini-batch is split across the eligible devices in proportion to "
            "their measured benchmark score; gradient sums are added, so the update equals "
            "centralized SGD on the same batch"
        ),
        "steps": tr["steps"],
        "global_batch_size": tr["global_batch_size"],
        "shares": [
            {
                "device_id": c["preferred_device_id"],
                "device_name": (devices_by_id.get(c["preferred_device_id"]) or {}).get("name"),
                "score_cells_per_sec": (
                    (devices_by_id.get(c["preferred_device_id"]) or {}).get("benchmark") or {}
                ).get("score_cells_per_sec"),
                "weight": c["n_rows"] / total,
                "rows": c["n_rows"],
                "n_chunks": 1,
                "est_seconds": None,
            }
            for c in docs
            if c["preferred_device_id"]
        ],
        "chunks": [],
    }


# ------------------------------------------------------------------ start
async def release_round(db: Db, task_id: str, round_: int) -> None:
    """Make a freshly created round visible to the scheduler (created -> pending)."""
    await db.col("chunks").update_many(
        {"task_id": task_id, "round": round_, "status": "created"},
        {"$set": {"status": "pending", "pending_since": utcnow()}},
    )


async def start_training(db: Db, settings: Settings, task: dict[str, Any]) -> bool:
    """queued -> running when >= min_devices are eligible. The conditional update is the claim:
    only the winner creates round 0 (a loser must never touch another attempt's chunks)."""
    devices = await _eligible_devices(db, settings, task, rows=srv.MIN_ROWS_PER_DEVICE)
    if len(devices) < task["execution"]["min_devices"]:
        return False
    params = srv.CnnParams.model_validate(task["params"])
    prepared = await load_task_images(db, task)
    arch = srv.build_arch(params, prepared.shape, len(prepared.class_names))
    n_params = core.n_params(arch)
    now = utcnow()
    training = {
        "arch": arch,
        "steps": params.steps,
        "global_batch_size": params.global_batch_size,
        "learning_rate": params.learning_rate,
        "momentum": params.momentum,
        "n_params": n_params,
        "batch_seed": params.split_seed + BATCH_SEED_OFFSET,
        "verify_rounds": srv.verify_round_set(params.steps, params.verify_rounds),
        "round": 0,
        "state": "starting",
        "starting_since": now,
        "model_version": 0,
    }
    timeout = max(settings.task_timeout_seconds, params.steps * 30)
    claim = await db.col("tasks").update_one(
        {"_id": task["_id"], "status": "queued"},
        {
            "$set": {
                "status": "running",
                "training": training,
                "started_at": now,
                "timeout_seconds": timeout,
            },
            "$push": {"status_history": {"status": "running", "at": now}},
        },
    )
    if not claim.modified_count:
        return False
    await _open_first_round(
        db, settings, {**task, "status": "running", "training": training}, devices
    )
    return True


async def _open_first_round(
    db: Db, settings: Settings, task: dict[str, Any], devices: list[dict[str, Any]]
) -> None:
    """Idempotent: safe to re-run if a previous attempt died after the claim."""
    task_id = task["_id"]
    tr = task["training"]
    arch = tr["arch"]
    w0 = core.init_weights(arch, int(task["params"]["init_seed"]))
    await save_state(db, task_id, 0, w0, np.zeros_like(w0))
    await db.col("chunks").delete_many({"task_id": task_id, "round": 0, "status": "created"})
    docs = await _make_round_chunks(db, settings, task, 0, w0)
    by_id = {d["_id"]: d for d in devices}
    res = await db.col("tasks").update_one(
        {"_id": task_id, "status": "running", "training.state": "starting"},
        {
            "$set": {
                "training.state": "collecting",
                "plan": {"created_at": utcnow(), **plan_summary(task, docs, by_id)},
            }
        },
    )
    if not res.modified_count:
        return
    await release_round(db, task_id, 0)
    await emit(
        db,
        "task_started",
        task_id=task_id,
        data={
            "kind": "training",
            "steps": tr["steps"],
            "batch": tr["global_batch_size"],
            "n_params": tr["n_params"],
            "devices": [d["name"] for d in devices[: task["execution"]["max_devices"]]],
        },
    )
    await schedule_pass(db, settings)


async def recover_start(db: Db, settings: Settings, task: dict[str, Any]) -> None:
    """A start that died after the claim: re-run the (idempotent) first-round creation."""
    tr = task["training"]
    if tr["state"] != "starting":
        return
    if (utcnow() - tr["starting_since"]) <= timedelta(seconds=CLOSING_STALE_SECONDS):
        return
    devices = await _eligible_devices(db, settings, task, rows=srv.MIN_ROWS_PER_DEVICE)
    await _open_first_round(db, settings, task, devices)


# ------------------------------------------------------------------ rounds
def _close_round_compute(
    task: dict[str, Any],
    payloads: list[dict[str, Any]],
    w: np.ndarray,
    v: np.ndarray,
    round_: int,
    verify: bool,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, Any]:
    """CPU part of closing a round (runs in a thread)."""
    tr = task["training"]
    merged = core.merge(payloads)
    grad_sum = core.decode_vector(merged["grad_sum_b64"], "<f8")
    if merged["n"] != tr["global_batch_size"]:
        raise RuntimeError("partial results do not cover the global batch")
    verification = None
    if verify:  # centralized recomputation of this round's whole batch
        idx = batch_idx(task, round_)
        _, _, ref = srv.reference_gradient(w, x_train[idx], y_train[idx], tr["arch"])
        verification = {"round": round_, **srv.compare_gradients(grad_sum, ref)}
    w2, v2 = srv.sgd_step(
        w, v, grad_sum, tr["global_batch_size"], tr["learning_rate"], tr["momentum"]
    )
    return {
        "w": w2,
        "v": v2,
        "loss": merged["loss_sum"] / merged["n"],
        "accuracy": merged["correct"] / merged["n"],
        "n": merged["n"],
        "verification": verification,
    }


async def advance_training(db: Db, settings: Settings, task_id: str, recover: bool = False) -> bool:
    """Close the current round if all its chunks are accepted. Idempotent and crash-safe."""
    task = await db.col("tasks").find_one({"_id": task_id, "status": "running"})
    if task is None or not task.get("training"):
        return False
    tr = task["training"]
    r = tr["round"]
    if tr["state"] == "closing":
        stale = (utcnow() - tr["closing_since"]) > timedelta(seconds=CLOSING_STALE_SECONDS)
        if not (recover and stale):
            return False
        await db.col("tasks").update_one(  # a previous attempt died mid-close: reopen
            {"_id": task_id, "training.state": "closing", "training.round": r},
            {"$set": {"training.state": "collecting"}},
        )
        task = await db.col("tasks").find_one({"_id": task_id, "status": "running"})
        if task is None:
            return False
        tr = task["training"]
    if tr["state"] != "collecting":
        return False
    chunks = [
        c async for c in db.col("chunks").find({"task_id": task_id, "round": r}).sort("index", 1)
    ]
    if not chunks or any(c["status"] != "completed" for c in chunks):
        return False
    claim = await db.col("tasks").update_one(
        {"_id": task_id, "status": "running", "training.state": "collecting", "training.round": r},
        {"$set": {"training.state": "closing", "training.closing_since": utcnow()}},
    )
    if not claim.modified_count:
        return False
    try:
        await _close_round(db, settings, task, chunks)
    except Exception as e:  # visible failure, never a silent stall
        await fail_task(db, task_id, f"training failed in round {r}: {e}", code="TRAINING_ERROR")
        return False
    return True


async def _close_round(
    db: Db, settings: Settings, task: dict[str, Any], chunks: list[dict[str, Any]]
) -> None:
    task_id = task["_id"]
    tr = task["training"]
    r = tr["round"]
    partials: list[dict[str, Any]] = []
    asgs: list[dict[str, Any]] = []
    for c in chunks:
        pr = await db.col("partial_results").find_one(
            {"assignment_id": c["accepted_assignment_id"], "acceptance": "accepted_unverified"}
        )
        asg = await db.col("assignments").find_one({"_id": c["accepted_assignment_id"]})
        if pr is None or asg is None:
            raise RuntimeError(f"no accepted partial for chunk {c['index']}")
        partials.append(pr["payload"])
        asgs.append(asg)
    w, v = await load_state(db, task_id, r)
    prepared = await load_task_images(db, task)
    out = await asyncio.to_thread(
        _close_round_compute,
        task,
        partials,
        w,
        v,
        r,
        r in tr["verify_rounds"],
        prepared.x_train,
        prepared.y_train,
    )
    await save_state(db, task_id, r + 1, out["w"], out["v"])
    # The gradients are consumed: keep digests/metadata, drop the ~150 KB payloads (free-tier storage)
    await db.col("partial_results").update_many(
        {"assignment_id": {"$in": [c["accepted_assignment_id"] for c in chunks]}},
        {"$unset": {"payload": ""}, "$set": {"payload_dropped": True}},
    )
    names = {
        d["_id"]: d["name"]
        async for d in db.col("devices").find({"_id": {"$in": [a["device_id"] for a in asgs]}})
    }
    devices = [
        {
            "device_id": a["device_id"],
            "name": names.get(a["device_id"]),
            "rows": c["n_rows"],
            "compute_ms": (a.get("timings") or {}).get("compute_ms"),
            "download_ms": (a.get("timings") or {}).get("download_ms"),
        }
        for a, c in zip(asgs, chunks, strict=True)
    ]
    await db.col("training_rounds").replace_one(
        {"_id": f"{task_id}:{r}"},
        {
            "_id": f"{task_id}:{r}",
            "task_id": task_id,
            "round": r,
            "loss": out["loss"],
            "accuracy": out["accuracy"],
            "n": out["n"],
            "devices": devices,
            "verification": out["verification"],
            "at": utcnow(),
        },
        upsert=True,
    )
    now = utcnow()
    last = r + 1 >= tr["steps"]
    if last:
        res = await db.col("tasks").update_one(
            {"_id": task_id, "status": "running", "training.state": "closing", "training.round": r},
            {
                "$set": {
                    "status": "aggregating",
                    "training.state": "done",
                    "training.round": r + 1,
                    "training.model_version": r + 1,
                },
                "$push": {"status_history": {"status": "aggregating", "at": now}},
            },
        )
        if res.modified_count:
            await emit(db, "task_aggregating", task_id=task_id)
            await _cleanup_states(db, task_id, keep=r + 1)
            await aggregate_task(db, task_id)
        return
    probe = {**task, "training": {**tr, "round": r + 1}}
    await db.col("chunks").delete_many({"task_id": task_id, "round": r + 1, "status": "created"})
    await _make_round_chunks(db, settings, probe, r + 1, out["w"])
    res = await db.col("tasks").update_one(
        {"_id": task_id, "status": "running", "training.state": "closing", "training.round": r},
        {
            "$set": {
                "training.state": "collecting",
                "training.round": r + 1,
                "training.model_version": r + 1,
            }
        },
    )
    if not res.modified_count:
        return
    await release_round(db, task_id, r + 1)  # only now can devices be assigned round r+1
    await _cleanup_states(db, task_id, keep=r)
    every = max(1, tr["steps"] // 10)
    if r % every == 0:
        await emit(
            db,
            "training_progress",
            task_id=task_id,
            data={
                "round": r + 1,
                "steps": tr["steps"],
                "loss": out["loss"],
                "accuracy": out["accuracy"],
            },
        )
    await schedule_pass(db, settings)


async def _cleanup_states(db: Db, task_id: str, keep: int) -> None:
    """Keep only the newest two model versions (storage on the free Atlas tier is small)."""
    await db.col("model_states").delete_many({"task_id": task_id, "version": {"$lt": keep}})


async def resume_training(db: Db, settings: Settings) -> None:
    """Reconciler: close rounds whose chunks are all accepted (e.g. after a restart)."""
    async for t in db.col("tasks").find({"status": "running", "training": {"$exists": True}}):
        if t["training"]["state"] == "starting":
            await recover_start(db, settings, t)
        else:
            await advance_training(db, settings, t["_id"], recover=True)
