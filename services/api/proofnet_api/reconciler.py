"""Idempotent maintenance loop (ARCHITECTURE 3.3, 11): offline marking, scheduling, aggregation.

Every pass is safe to re-run: transitions are conditional updates guarded by status and
last_seen_at, so a repeat (or a concurrent heartbeat) cannot apply one twice.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from .aggregation.service import aggregate_task, release_stale_claims
from .config import Settings
from .db import Db, utcnow
from .events import emit
from .scheduling.lifecycle import expire_assignments, expire_tasks
from .scheduling.scheduler import schedule_pass
from .training.service import resume_training

log = logging.getLogger("proofnet.reconciler")


async def mark_offline_devices(db: Db, settings: Settings, now: datetime | None = None) -> int:
    now = now or utcnow()
    cutoff = now - timedelta(seconds=settings.offline_after_seconds)
    stale = db.col("devices").find(
        {"status": {"$in": ["idle", "busy"]}, "last_seen_at": {"$lt": cutoff}}, {"_id": 1}
    )
    count = 0
    async for d in stale:
        # Re-check the precondition atomically: a heartbeat may have landed meanwhile.
        res = await db.col("devices").update_one(
            {"_id": d["_id"], "status": {"$in": ["idle", "busy"]}, "last_seen_at": {"$lt": cutoff}},
            {"$set": {"status": "offline"}},
        )
        if res.modified_count:
            count += 1
            await emit(db, "device_offline", device_id=d["_id"])
    return count


async def resume_aggregations(db: Db) -> None:
    """Re-run aggregation that stalled (e.g. the backend restarted mid-aggregation)."""
    await release_stale_claims(db)
    async for t in db.col("tasks").find({"status": "aggregating", "aggregation_claimed_at": None}):
        await aggregate_task(db, t["_id"])


async def run_once(db: Db, settings: Settings, now: datetime | None = None) -> None:
    await mark_offline_devices(db, settings, now)
    await expire_assignments(db, settings, now)  # deadlines, offline devices -> retry
    await expire_tasks(db, settings, now)  # queue and task timeouts
    await schedule_pass(db, settings)
    await resume_aggregations(db)
    await resume_training(db, settings)


async def run_forever(db: Db, settings: Settings) -> None:
    while True:
        try:
            await run_once(db, settings)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("reconciler pass failed")  # never die; next pass retries
        await asyncio.sleep(settings.reconciler_interval_seconds)
