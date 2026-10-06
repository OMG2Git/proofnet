"""Idempotent maintenance loop (ARCHITECTURE 3.3, 11). P2 scope: mark stale devices offline.

Every pass is safe to re-run: transitions are conditional updates guarded by status and
last_seen_at, so a repeat (or a concurrent heartbeat) cannot apply one twice.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from .config import Settings
from .db import Db, utcnow
from .events import emit

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


async def run_once(db: Db, settings: Settings, now: datetime | None = None) -> None:
    await mark_offline_devices(db, settings, now)
    # P6 adds: lease expiry, retries, timeouts, stuck aggregation.


async def run_forever(db: Db, settings: Settings) -> None:
    while True:
        try:
            await run_once(db, settings)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("reconciler pass failed")  # never die; next pass retries
        await asyncio.sleep(settings.reconciler_interval_seconds)
