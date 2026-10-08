"""Storage budget for the free Atlas tier (512 MB, writes are blocked at the limit).

What is large: uploaded raw files, prepared datasets (needed only while a task runs) and, for
iterative training, gradient payloads (~150 KB each). When the database nears its budget, release
what is no longer needed, oldest first, never touching data of queued/running tasks:
  1. gradient payloads of finished image-training tasks (digests and metadata stay),
  2. prepared datasets of finished tasks,
  3. raw uploaded datasets not referenced by an active task.
Artifacts (the results) and all metadata are kept.
"""

import logging
from typing import Any

from .config import Settings
from .db import Db, utcnow

log = logging.getLogger("proofnet.storage")
FINISHED = ["completed", "failed", "cancelled"]
ACTIVE = ["queued", "running", "aggregating"]


async def size_mb(db: Db) -> float:
    st = await db.db.command("dbStats")
    return float(st.get("dataSize", 0) + st.get("indexSize", 0)) / 1e6


async def _strip_gradient_payloads(db: Db) -> int:
    finished = [
        t["_id"]
        async for t in db.col("tasks").find(
            {"status": {"$in": FINISHED}, "training": {"$exists": True}}, {"_id": 1}
        )
    ]
    if not finished:
        return 0
    res = await db.col("partial_results").update_many(
        {"task_id": {"$in": finished}, "payload": {"$exists": True}},
        {"$unset": {"payload": ""}, "$set": {"payload_dropped": True}},
    )
    return int(res.modified_count)


async def _drop_prepared(db: Db, until_mb: float) -> int:
    n = 0
    cursor = (
        db.col("tasks")
        .find(
            {
                "status": {"$in": FINISHED},
                "prepared.file_id": {"$exists": True},
                "prepared.dropped": {"$ne": True},
            }
        )
        .sort("created_at", 1)
    )
    async for t in cursor:
        if await size_mb(db) <= until_mb:
            break
        try:
            await db.fs.delete(t["prepared"]["file_id"])
        except Exception:
            log.warning("prepared file of %s already gone", t["_id"])
        await db.col("tasks").update_one({"_id": t["_id"]}, {"$set": {"prepared.dropped": True}})
        n += 1
    return n


async def _drop_raw(db: Db, until_mb: float) -> int:
    in_use: set[str] = set()
    async for t in db.col("tasks").find({"status": {"$in": ACTIVE}}, {"dataset_id": 1}):
        in_use.add(t["dataset_id"])
    n = 0
    for coll, field in (("datasets", "raw_file_id"), ("image_datasets", "npz_file_id")):
        cursor = (
            db.col(coll)
            .find({field: {"$exists": True}, "raw_dropped": {"$ne": True}})
            .sort("created_at", 1)
        )
        async for d in cursor:
            if await size_mb(db) <= until_mb:
                return n
            if d["_id"] in in_use:
                continue
            try:
                await db.fs.delete(d[field])
            except Exception:
                log.warning("raw file of %s already gone", d["_id"])
            await db.col(coll).update_one({"_id": d["_id"]}, {"$set": {"raw_dropped": True}})
            n += 1
    return n


async def free_space(db: Db, settings: Settings, force: bool = False) -> dict[str, Any]:
    """Release space until the database is under 80 % of its budget (no-op when it is fine)."""
    before = await size_mb(db)
    budget = float(settings.storage_budget_mb)
    report: dict[str, Any] = {"before_mb": round(before, 1), "budget_mb": budget}
    if before <= budget and not force:
        report["freed"] = False
        return report
    target = budget * 0.8
    report["gradient_payloads_dropped"] = await _strip_gradient_payloads(db)
    report["prepared_dropped"] = await _drop_prepared(db, target)
    report["raw_datasets_dropped"] = await _drop_raw(db, target)
    report["after_mb"] = round(await size_mb(db), 1)
    report["freed"] = True
    report["at"] = utcnow().isoformat()
    log.warning("storage guard: %s", report)
    return report


async def ensure_capacity(db: Db, settings: Settings, need_mb: float) -> None:
    """Before accepting an upload: release old data if needed; refuse (507) if still too full."""
    from .errors import ProofNetError

    if await size_mb(db) + need_mb > settings.storage_budget_mb:
        await free_space(db, settings, force=True)
    if await size_mb(db) + need_mb > settings.storage_budget_mb * 1.1:
        raise ProofNetError(
            507, "STORAGE_FULL", "The free database tier is full; delete old tasks or try later"
        )
