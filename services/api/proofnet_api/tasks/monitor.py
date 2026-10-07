"""Task monitor endpoints: live status snapshot, event timeline, artifacts, downloads."""

import io
from typing import Any

from fastapi import APIRouter, Response

from ..contracts.api import (
    ArtifactOut,
    AssignmentOut,
    ChunkOut,
    EventOut,
    TaskStatus,
)
from ..db import Db, utcnow
from ..deps import DbDep, UserDep
from ..errors import ProofNetError
from .routes import task_out

router = APIRouter(tags=["tasks"])


async def _owned_task(db: Db, task_id: str, user_id: str) -> dict[str, Any]:
    t = await db.col("tasks").find_one({"_id": task_id, "owner_user_id": user_id})
    if t is None:
        raise ProofNetError(404, "NOT_FOUND", "Task not found")
    return t


def _artifact_out(a: dict[str, Any]) -> ArtifactOut:
    return ArtifactOut(
        id=a["_id"],
        task_id=a["task_id"],
        kind=a["kind"],
        filename=a["filename"],
        size_bytes=a["size_bytes"],
        sha256=a["sha256"],
        created_at=a["created_at"],
    )


@router.get("/tasks/{task_id}/status", response_model=TaskStatus)
async def task_status(task_id: str, db: DbDep, user: UserDep) -> TaskStatus:
    """Live snapshot: task, chunks, assignments (with device names), artifacts."""
    t = await _owned_task(db, task_id, user["_id"])
    chunks = [c async for c in db.col("chunks").find({"task_id": task_id}).sort("index", 1)]
    asgs = [
        a async for a in db.col("assignments").find({"task_id": task_id}).sort("assigned_at", 1)
    ]
    device_ids = list({a["device_id"] for a in asgs})
    names = {
        d["_id"]: d["name"] async for d in db.col("devices").find({"_id": {"$in": device_ids}})
    }
    arts = [a async for a in db.col("artifacts").find({"task_id": task_id}).sort("created_at", 1)]
    return TaskStatus(
        server_time=utcnow(),
        task=task_out(t),
        chunks=[
            ChunkOut(
                id=c["_id"],
                index=c["index"],
                role=c["role"],
                row_start=c["row_start"],
                row_end=c["row_end"],
                n_rows=c["n_rows"],
                status=c["status"],
                attempt_count=c["attempt_count"],
                max_attempts=c["max_attempts"],
                preferred_device_id=c.get("preferred_device_id"),
                accepted_assignment_id=c.get("accepted_assignment_id"),
            )
            for c in chunks
        ],
        assignments=[
            AssignmentOut(
                id=a["_id"],
                chunk_id=a["chunk_id"],
                device_id=a["device_id"],
                device_name=names.get(a["device_id"]),
                attempt_no=a["attempt_no"],
                purpose=a["purpose"],
                status=a["status"],
                assigned_at=a["assigned_at"],
                started_at=a.get("started_at"),
                finished_at=a.get("finished_at"),
                deadline_at=a["deadline_at"],
                timings=a.get("timings"),
                runtime_fingerprint=a.get("runtime_fingerprint"),
                error=a.get("error"),
            )
            for a in asgs
        ],
        artifacts=[_artifact_out(a) for a in arts],
    )


@router.get("/tasks/{task_id}/events", response_model=list[EventOut])
async def task_events(task_id: str, db: DbDep, user: UserDep) -> list[EventOut]:
    await _owned_task(db, task_id, user["_id"])
    cursor = db.col("events").find({"task_id": task_id}).sort("ts", 1)
    return [
        EventOut(
            id=e["_id"],
            ts=e["ts"],
            type=e["type"],
            task_id=e.get("task_id"),
            device_id=e.get("device_id"),
            chunk_id=e.get("chunk_id"),
            assignment_id=e.get("assignment_id"),
            data=e.get("data", {}),
        )
        async for e in cursor
    ]


@router.get("/tasks/{task_id}/artifacts", response_model=list[ArtifactOut])
async def task_artifacts(task_id: str, db: DbDep, user: UserDep) -> list[ArtifactOut]:
    await _owned_task(db, task_id, user["_id"])
    cursor = db.col("artifacts").find({"task_id": task_id}).sort("created_at", 1)
    return [_artifact_out(a) async for a in cursor]


_MEDIA = {
    "model_json": "application/json",
    "report_json": "application/json",
    "predictions_csv": "text/csv",
}


@router.get("/artifacts/{artifact_id}/download")
async def download_artifact(artifact_id: str, db: DbDep, user: UserDep) -> Response:
    a = await db.col("artifacts").find_one({"_id": artifact_id})
    if a is None:
        raise ProofNetError(404, "NOT_FOUND", "Artifact not found")
    await _owned_task(db, a["task_id"], user["_id"])  # owner only
    buf = io.BytesIO()
    await db.fs.download_to_stream(a["file_id"], buf)
    return Response(
        content=buf.getvalue(),
        media_type=_MEDIA.get(a["kind"], "application/octet-stream"),
        headers={"Content-Disposition": 'attachment; filename="' + a["filename"] + '"'},
    )
