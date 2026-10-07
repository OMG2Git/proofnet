"""Scheduler: start queued tasks (plan chunks) and assign pending chunks (ARCHITECTURE 6.3).

Every transition is a conditional atomic update; a failed precondition rolls back and moves on.
Run by the reconciler each pass and right after events that free capacity.
"""

import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Any

from ..config import Settings
from ..db import Db, utcnow
from ..events import emit
from ..ids import new_id
from ..tasks.prepared import load_task_prepared
from .planner import ChunkPlanner, SingleChunkPlanner
from .policies import AssignmentPolicy, OnePrimary, is_eligible

MIN_DEADLINE_S, MAX_DEADLINE_S = 60.0, 600.0

PLANNER: ChunkPlanner = SingleChunkPlanner()
ASSIGNMENT_POLICY: AssignmentPolicy = OnePrimary()


def deadline_seconds(est_seconds: float) -> float:
    """ARCHITECTURE 11: clamp(4 x estimated_seconds, 60 s, 600 s)."""
    return min(MAX_DEADLINE_S, max(MIN_DEADLINE_S, 4 * est_seconds))


async def _eligible_devices(
    db: Db,
    settings: Settings,
    task: dict[str, Any],
    rows: int,
    excluded: list[str] | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    now = now or utcnow()
    n_features = task["prepared"]["n_features"]
    cursor = db.col("devices").find({"status": "idle"})
    return [
        d
        async for d in cursor
        if is_eligible(
            d,
            now=now,
            settings=settings,
            rows=rows,
            n_features=n_features,
            excluded_device_ids=excluded,
            task=task,
        )
    ]


async def try_start_task(db: Db, settings: Settings, task: dict[str, Any]) -> bool:
    """queued -> running: plan chunks when enough eligible devices exist."""
    n_train, n_features = task["prepared"]["n_train"], task["prepared"]["n_features"]
    if task["execution"]["min_devices"] > 1:
        return False  # multi-device planning arrives in P5; the task waits (queue timeout applies)
    devices = await _eligible_devices(db, settings, task, rows=n_train)
    planned = PLANNER.plan(n_train, n_features, devices)
    if not planned:
        return False

    prepared = await load_task_prepared(db, task)
    chunks: list[dict[str, Any]] = []
    for p in planned:
        data = await asyncio.to_thread(prepared.chunk_npz_bytes, p.row_start, p.row_end)
        chunks.append(
            {
                "_id": new_id("chk"),
                "task_id": task["_id"],
                "index": p.index,
                "role": "work",
                "row_start": p.row_start,
                "row_end": p.row_end,
                "n_rows": p.n_rows,
                "work_units": p.n_rows * n_features,
                "input_sha256": hashlib.sha256(data).hexdigest(),
                "preferred_device_id": p.preferred_device_id,
                "status": "pending",
                "attempt_count": 0,
                "max_attempts": settings.max_attempts,
                "excluded_device_ids": [],
                "accepted_assignment_id": None,
                "created_at": utcnow(),
            }
        )
    await db.col("chunks").insert_many(chunks)
    now = utcnow()
    plan = {
        "created_at": now,
        "planner": type(PLANNER).__name__,
        "shares": [
            {
                "device_id": p.preferred_device_id,
                "chunk_index": p.index,
                "rows": p.n_rows,
                "est_seconds": p.est_seconds,
            }
            for p in planned
        ],
    }
    res = await db.col("tasks").update_one(
        {"_id": task["_id"], "status": "queued"},
        {
            "$set": {"status": "running", "plan": plan},
            "$push": {"status_history": {"status": "running", "at": now}},
        },
    )
    if not res.modified_count:  # lost a race: undo
        await db.col("chunks").delete_many({"_id": {"$in": [c["_id"] for c in chunks]}})
        return False
    await emit(
        db,
        "task_started",
        task_id=task["_id"],
        data={"chunks": len(chunks), "devices": [p.preferred_device_id for p in planned]},
    )
    return True


async def try_assign_chunk(
    db: Db, settings: Settings, chunk: dict[str, Any], now: datetime | None = None
) -> bool:
    now = now or utcnow()
    task = await db.col("tasks").find_one({"_id": chunk["task_id"], "status": "running"})
    if task is None:
        return False
    excluded = chunk.get("excluded_device_ids", [])
    candidates = await _eligible_devices(
        db, settings, task, rows=chunk["n_rows"], excluded=excluded, now=now
    )
    preferred = next((d for d in candidates if d["_id"] == chunk.get("preferred_device_id")), None)
    waited = (now - chunk["created_at"]) > timedelta(seconds=settings.preferred_wait_seconds)
    if preferred is not None:
        ordered = [preferred]
    elif (
        waited
        or chunk.get("preferred_device_id") in excluded
        or not chunk.get("preferred_device_id")
    ):
        ordered = sorted(
            candidates, key=lambda d: (-d["benchmark"]["score_cells_per_sec"], d["_id"])
        )
    else:
        ordered = []
    for device in ordered:
        for purpose in ASSIGNMENT_POLICY.purposes_for(chunk):
            if await _create_assignment(db, settings, task, chunk, device, purpose, now):
                return True
    return False


async def _create_assignment(
    db: Db,
    settings: Settings,
    task: dict[str, Any],
    chunk: dict[str, Any],
    device: dict[str, Any],
    purpose: str,
    now: datetime,
) -> bool:
    score = float(device["benchmark"]["score_cells_per_sec"])
    est = chunk["work_units"] / score
    asg_id = new_id("asg")
    attempt_no = chunk["attempt_count"] + 1
    asg = {
        "_id": asg_id,
        "task_id": task["_id"],
        "chunk_id": chunk["_id"],
        "device_id": device["_id"],
        "attempt_no": attempt_no,
        "purpose": purpose,
        "status": "assigned",
        "assigned_at": now,
        "started_at": None,
        "finished_at": None,
        "deadline_at": now + timedelta(seconds=deadline_seconds(est)),
        "timings": None,
        "runtime_fingerprint": None,
        "error": None,
        "partial_result_id": None,
        "work_units": chunk["work_units"],
    }
    await db.col("assignments").insert_one(asg)
    c = await db.col("chunks").update_one(
        {"_id": chunk["_id"], "status": "pending", "attempt_count": chunk["attempt_count"]},
        {"$set": {"status": "assigned"}, "$inc": {"attempt_count": 1}},
    )
    if not c.modified_count:
        await db.col("assignments").delete_one({"_id": asg_id})
        return False
    d = await db.col("devices").update_one(
        {"_id": device["_id"], "status": "idle"},
        {"$set": {"status": "busy", "current_assignment_id": asg_id}},
    )
    if not d.modified_count:  # roll back the chunk transition
        await db.col("chunks").update_one(
            {"_id": chunk["_id"], "status": "assigned"},
            {"$set": {"status": "pending"}, "$inc": {"attempt_count": -1}},
        )
        await db.col("assignments").delete_one({"_id": asg_id})
        return False
    await emit(
        db,
        "chunk_assigned",
        task_id=task["_id"],
        device_id=device["_id"],
        chunk_id=chunk["_id"],
        assignment_id=asg_id,
        data={"rows": chunk["n_rows"], "attempt": attempt_no, "est_seconds": est},
    )
    return True


async def schedule_pass(db: Db, settings: Settings) -> None:
    async for task in db.col("tasks").find({"status": "queued"}).sort("created_at", 1):
        await try_start_task(db, settings, task)
    async for chunk in (
        db.col("chunks").find({"status": "pending"}).sort([("created_at", 1), ("index", 1)])
    ):
        await try_assign_chunk(db, settings, chunk)
