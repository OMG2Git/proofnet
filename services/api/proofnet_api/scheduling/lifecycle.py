"""Assignment/chunk/task failure handling (ARCHITECTURE 10, 11).

Every transition is a conditional atomic update, so a repeat (reconciler re-run, duplicate request,
result racing with expiry) can never apply twice.
"""

from datetime import datetime, timedelta
from typing import Any

from ..config import Settings
from ..db import Db, utcnow
from ..events import emit

ACTIVE = ["assigned", "running"]


async def free_device(db: Db, device_id: str, assignment_id: str, stat: str | None = None) -> None:
    """Detach a finished assignment from its device; a busy device becomes idle again.

    An offline/disabled device keeps its status (it must start a new session to be usable)."""
    update: dict[str, Any] = {"$set": {"current_assignment_id": None}}
    if stat:
        update["$inc"] = {f"stats.{stat}": 1}
    await db.col("devices").update_one(
        {"_id": device_id, "current_assignment_id": assignment_id}, update
    )
    await db.col("devices").update_one(
        {"_id": device_id, "status": "busy", "current_assignment_id": None},
        {"$set": {"status": "idle"}},
    )


async def _close_open_work(db: Db, task_id: str, code: str, message: str, now: datetime) -> None:
    """Cancel pending/assigned chunks and active assignments of a task that is ending."""
    await db.col("chunks").update_many(
        {"task_id": task_id, "status": {"$in": ["pending", "assigned"]}},
        {"$set": {"status": "cancelled"}},
    )
    async for a in db.col("assignments").find({"task_id": task_id, "status": {"$in": ACTIVE}}):
        res = await db.col("assignments").update_one(
            {"_id": a["_id"], "status": {"$in": ACTIVE}},
            {
                "$set": {
                    "status": "cancelled",
                    "finished_at": now,
                    "error": {"code": code, "message": message},
                }
            },
        )
        if res.modified_count:
            await free_device(db, a["device_id"], a["_id"])
            await emit(
                db,
                "assignment_cancelled",
                task_id=task_id,
                device_id=a["device_id"],
                chunk_id=a["chunk_id"],
                assignment_id=a["_id"],
                data={"code": code, "message": message},
            )


async def end_task(
    db: Db, task_id: str, status: str, reason: str, *, code: str, from_states: list[str]
) -> bool:
    """queued/running(/aggregating) -> failed|cancelled, closing all open work. Idempotent."""
    now = utcnow()
    res = await db.col("tasks").update_one(
        {"_id": task_id, "status": {"$in": from_states}},
        {
            "$set": {"status": status, "error": reason},
            "$push": {"status_history": {"status": status, "at": now}},
        },
    )
    if not res.modified_count:
        return False
    await _close_open_work(db, task_id, code, reason, now)
    await emit(db, f"task_{status}", task_id=task_id, data={"error": reason, "code": code})
    return True


async def fail_task(db: Db, task_id: str, reason: str, code: str = "TASK_FAILED") -> bool:
    return await end_task(
        db, task_id, "failed", reason, code=code, from_states=["queued", "running", "aggregating"]
    )


async def cancel_task(db: Db, task_id: str) -> bool:
    """User cancel: only queued or running tasks (aggregating cannot be interrupted)."""
    return await end_task(
        db,
        task_id,
        "cancelled",
        "cancelled by the owner",
        code="CANCELLED",
        from_states=["queued", "running"],
    )


async def release_attempt(
    db: Db,
    asg: dict[str, Any],
    chunk: dict[str, Any],
    new_status: str,
    error: dict[str, str],
    from_status: list[str],
    *,
    device_stat: str,
) -> bool:
    """Assignment -> rejected/failed/expired; chunk -> pending (or failed after max attempts)."""
    now = utcnow()
    res = await db.col("assignments").update_one(
        {"_id": asg["_id"], "status": {"$in": from_status}},
        {"$set": {"status": new_status, "finished_at": now, "error": error}},
    )
    if not res.modified_count:
        return False
    fresh = await db.col("chunks").find_one({"_id": chunk["_id"]}) or chunk
    exhausted = fresh["attempt_count"] >= fresh["max_attempts"]
    if exhausted:
        await db.col("chunks").update_one(
            {"_id": chunk["_id"], "status": "assigned"}, {"$set": {"status": "failed"}}
        )
    else:
        await db.col("chunks").update_one(
            {"_id": chunk["_id"], "status": "assigned"},
            {
                "$set": {"status": "pending", "pending_since": now},
                "$addToSet": {"excluded_device_ids": asg["device_id"]},
            },
        )
    await free_device(db, asg["device_id"], asg["_id"], stat=device_stat)
    await emit(
        db,
        f"assignment_{new_status}",
        task_id=asg["task_id"],
        device_id=asg["device_id"],
        chunk_id=chunk["_id"],
        assignment_id=asg["_id"],
        data={**error, "attempt": fresh["attempt_count"], "max_attempts": fresh["max_attempts"]},
    )
    if exhausted:
        await fail_task(
            db,
            chunk["task_id"],
            f"chunk {chunk['index']} failed after {fresh['attempt_count']} attempts: "
            f"{error.get('message', error.get('code', ''))}",
            code="CHUNK_FAILED",
        )
    else:
        await emit(
            db,
            "chunk_requeued",
            task_id=asg["task_id"],
            chunk_id=chunk["_id"],
            data={"attempt": fresh["attempt_count"], "max_attempts": fresh["max_attempts"]},
        )
    return True


async def expire_assignments(db: Db, settings: Settings, now: datetime | None = None) -> int:
    """Lease/deadline expiry and offline-device expiry (ARCHITECTURE 11)."""
    now = now or utcnow()
    count = 0
    async for a in db.col("assignments").find({"status": {"$in": ACTIVE}}):
        dev = await db.col("devices").find_one({"_id": a["device_id"]})
        if now > a["deadline_at"]:
            code, msg = "DEADLINE", "the device did not finish before the deadline"
        elif dev is None or dev["status"] in ("offline", "disabled"):
            code, msg = "DEVICE_OFFLINE", "the device went offline"
        else:
            continue
        chunk = await db.col("chunks").find_one({"_id": a["chunk_id"]})
        if chunk is None:
            continue
        if await release_attempt(
            db, a, chunk, "expired", {"code": code, "message": msg}, ACTIVE, device_stat="expired"
        ):
            count += 1
    return count


async def expire_tasks(db: Db, settings: Settings, now: datetime | None = None) -> int:
    """Queue timeout (no eligible devices) and task timeout (running too long)."""
    now = now or utcnow()
    n = 0
    q_cut = now - timedelta(seconds=settings.queue_timeout_seconds)
    async for t in db.col("tasks").find({"status": "queued", "created_at": {"$lt": q_cut}}):
        if await fail_task(
            db,
            t["_id"],
            f"no eligible devices within {settings.queue_timeout_seconds} s (queue timeout)",
            code="QUEUE_TIMEOUT",
        ):
            n += 1
    r_cut = now - timedelta(seconds=settings.task_timeout_seconds)
    async for t in db.col("tasks").find({"status": "running", "started_at": {"$lt": r_cut}}):
        if await fail_task(
            db,
            t["_id"],
            f"task did not finish within {settings.task_timeout_seconds} s (task timeout)",
            code="TASK_TIMEOUT",
        ):
            n += 1
    return n
