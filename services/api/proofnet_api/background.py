"""Fire-and-forget helpers that keep strong references to background tasks."""

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

from fastapi import FastAPI

from .config import Settings
from .db import Db
from .scheduling.scheduler import schedule_pass

log = logging.getLogger("proofnet.background")


def spawn(app: FastAPI, coro: Coroutine[Any, Any, Any]) -> None:
    tasks: set[asyncio.Task[Any]] = app.state.background
    t = asyncio.create_task(coro)
    tasks.add(t)
    t.add_done_callback(tasks.discard)


def kick_scheduler(app: FastAPI, settings: Settings) -> None:
    """Run a scheduling pass now (capacity changed); the reconciler is the safety net."""
    db: Db = app.state.db

    async def go() -> None:
        try:
            await schedule_pass(db, settings)
        except Exception:  # the next reconciler pass retries
            log.exception("scheduling pass failed")

    spawn(app, go())
