"""Append-only event log (feeds dashboards; Part 2 audit trail)."""

from typing import Any

from ..db import Db, utcnow
from ..ids import new_id


async def emit(
    db: Db,
    type_: str,
    *,
    task_id: str | None = None,
    device_id: str | None = None,
    chunk_id: str | None = None,
    assignment_id: str | None = None,
    data: dict[str, Any] | None = None,
) -> None:
    await db.col("events").insert_one(
        {
            "_id": new_id("evt"),
            "ts": utcnow(),
            "type": type_,
            "task_id": task_id,
            "device_id": device_id,
            "chunk_id": chunk_id,
            "assignment_id": assignment_id,
            "data": data or {},
        }
    )
