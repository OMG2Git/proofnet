"""Task monitor endpoints: live status snapshot, event timeline, artifacts, downloads."""

import io
from typing import Any

from fastapi import APIRouter, Request, Response

from .. import cache
from ..config import Settings
from ..contracts.api import (
    ArtifactOut,
    AssignmentOut,
    ChunkOut,
    EventOut,
    TaskStatus,
)
from ..db import Db, utcnow
from ..deps import DbDep, SettingsDep, UserDep
from ..errors import ProofNetError
from ..scheduling.planner import MIN_CHUNK_ROWS
from ..scheduling.preview import device_reasons
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


async def _waiting_reasons(db: Db, settings: Settings, t: dict[str, Any]) -> list[str]:
    """Why a queued task has not started: the start policy and every device's blocker."""
    n_features = t["prepared"]["n_features"]
    reasons = await device_reasons(db, settings, t, MIN_CHUNK_ROWS, n_features)
    usable = await db.col("devices").count_documents({"status": {"$ne": "disabled"}})
    eligible = usable - len(reasons)
    need = t["execution"]["min_devices"]
    if eligible < need:
        reasons.insert(
            0, f"start policy needs {need} eligible device(s); {max(eligible, 0)} available"
        )
    elif not reasons:
        reasons.append("eligible devices are available; scheduling will start the task")
    return reasons


@router.get("/tasks/{task_id}/status", response_model=TaskStatus)
async def task_status(
    task_id: str, request: Request, db: DbDep, user: UserDep, settings: SettingsDep
) -> TaskStatus:
    """Live snapshot: task, chunks, assignments (with device names), artifacts. Cached ~1 s."""
    key = f"status:{task_id}"
    hit = cache.get(request.app, key)
    if hit is not None:
        owner, cached = hit
        if owner != user["_id"]:
            raise ProofNetError(404, "NOT_FOUND", "Task not found")
        return cached  # type: ignore[no-any-return]
    snapshot = await _build_status(db, settings, task_id, user["_id"])
    cache.put(request.app, key, (user["_id"], snapshot), settings.status_cache_seconds)
    return snapshot


async def _build_status(db: Db, settings: Settings, task_id: str, user_id: str) -> TaskStatus:
    t = await _owned_task(db, task_id, user_id)
    chunks = [c async for c in db.col("chunks").find({"task_id": task_id}).sort("index", 1)]
    asgs = [
        a async for a in db.col("assignments").find({"task_id": task_id}).sort("assigned_at", 1)
    ]
    device_ids = list({a["device_id"] for a in asgs})
    names = {
        d["_id"]: d["name"] async for d in db.col("devices").find({"_id": {"$in": device_ids}})
    }
    arts = [a async for a in db.col("artifacts").find({"task_id": task_id}).sort("created_at", 1)]
    waiting = await _waiting_reasons(db, settings, t) if t["status"] == "queued" else []
    return TaskStatus(
        server_time=utcnow(),
        task=task_out(t),
        waiting_reasons=waiting,
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


def describe_event(e: dict[str, Any], names: dict[str, str], chunk_index: dict[str, int]) -> str:
    """Human-readable line for the event feed (real events only)."""
    d = e.get("data") or {}
    dev = names.get(e.get("device_id") or "", "a device")
    idx = chunk_index.get(e.get("chunk_id") or "")
    chunk = f"chunk {idx}" if idx is not None else "a chunk"
    t = e["type"]
    if t == "task_created":
        return "Task created and queued"
    if t == "task_started":
        per = ", ".join(f"{k}: {v} rows" for k, v in (d.get("rows_per_device") or {}).items())
        n_dev = len(d.get("devices") or [])
        return f"Plan created: {d.get('chunks')} chunk(s) on {n_dev} device(s) ({per})"
    if t == "chunk_assigned":
        return f"{chunk.capitalize()} ({d.get('rows')} rows) assigned to {dev}"
    if t == "assignment_started":
        return f"{dev} started {chunk}"
    if t == "result_accepted":
        return f"{dev} finished {chunk} in {round(d.get('compute_ms') or 0)} ms"
    if t == "assignment_rejected":
        return f"{dev}'s result for {chunk} was rejected: {d.get('message', 'invalid result')}"
    if t == "assignment_failed":
        return f"{dev} failed {chunk}: {d.get('message', 'error')}"
    if t == "assignment_expired":
        return f"{dev} went offline - {chunk} will be reassigned"
    if t == "late_result":
        return f"Late result from {dev} for {chunk} ignored"
    if t == "task_aggregating":
        return "All chunks complete; merging partial results"
    if t == "task_completed":
        ok = d.get("reference_passed")
        return "Task completed; reference check " + ("passed" if ok else "FAILED")
    if t == "task_failed":
        return f"Task failed: {d.get('error', '')}"
    return str(t).replace("_", " ")


@router.get("/tasks/{task_id}/events", response_model=list[EventOut])
async def task_events(task_id: str, db: DbDep, user: UserDep) -> list[EventOut]:
    await _owned_task(db, task_id, user["_id"])
    events = [e async for e in db.col("events").find({"task_id": task_id}).sort("ts", 1)]
    dev_ids = list({e["device_id"] for e in events if e.get("device_id")})
    names = {d["_id"]: d["name"] async for d in db.col("devices").find({"_id": {"$in": dev_ids}})}
    chunk_index = {c["_id"]: c["index"] async for c in db.col("chunks").find({"task_id": task_id})}
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
            message=describe_event(e, names, chunk_index),
        )
        for e in events
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
