"""Live network summary for the /network dashboard (real device state only). Cached ~1 s."""

from fastapi import APIRouter, Request

from .. import cache
from ..contracts.api import NetworkDevice, NetworkSummary
from ..db import utcnow
from ..deps import DbDep, SettingsDep, UserDep
from ..trust.store import trust_of

router = APIRouter(prefix="/network", tags=["network"])

STATUSES = ("initializing", "idle", "busy", "offline", "disabled")


@router.get("/summary", response_model=NetworkSummary)
async def network_summary(
    request: Request, db: DbDep, _: UserDep, settings: SettingsDep
) -> NetworkSummary:
    hit = cache.get(request.app, "network")
    if hit is not None:
        return hit  # type: ignore[no-any-return]
    now = utcnow()
    docs = [d async for d in db.col("devices").find({}).sort("created_at", 1)]
    # Which chunk (of which task) each busy device holds right now.
    asg_ids = [d["current_assignment_id"] for d in docs if d.get("current_assignment_id")]
    asgs = {a["_id"]: a async for a in db.col("assignments").find({"_id": {"$in": asg_ids}})}
    chunks = {
        c["_id"]: c
        async for c in db.col("chunks").find(
            {"_id": {"$in": [a["chunk_id"] for a in asgs.values()]}}
        )
    }
    tasks = {
        t["_id"]: t
        async for t in db.col("tasks").find({"_id": {"$in": [a["task_id"] for a in asgs.values()]}})
    }
    profiles = {
        p["_id"]: p
        async for p in db.col("device_trust").find({"_id": {"$in": [d["_id"] for d in docs]}})
    }
    devices: list[NetworkDevice] = []
    counts = dict.fromkeys(STATUSES, 0)
    for d in docs:
        counts[d["status"]] = counts.get(d["status"], 0) + 1
        seen = d.get("last_seen_at")
        asg = asgs.get(d.get("current_assignment_id") or "")
        chunk = chunks.get(asg["chunk_id"]) if asg else None
        task = tasks.get(asg["task_id"]) if asg else None
        devices.append(
            NetworkDevice(
                id=d["_id"],
                name=d["name"],
                device_type=d["device_type"],
                status=d["status"],
                score_cells_per_sec=(d.get("benchmark") or {}).get("score_cells_per_sec"),
                runtime_kind=(d.get("runtime") or {}).get("kind"),
                trust_status=profiles[d["_id"]]["status"] if d["_id"] in profiles else None,
                trust=round(trust_of(profiles[d["_id"]]), 3) if d["_id"] in profiles else None,
                last_seen_age_seconds=(now - seen).total_seconds() if seen else None,
                current_assignment_id=d.get("current_assignment_id"),
                current_task_id=asg["task_id"] if asg else None,
                current_task_name=task["name"] if task else None,
                current_chunk_index=chunk["index"] if chunk else None,
                current_rows=chunk["n_rows"] if chunk else None,
            )
        )
    running = await db.col("tasks").count_documents({"status": {"$in": ["running", "aggregating"]}})
    summary = NetworkSummary(server_time=now, counts=counts, devices=devices, tasks_running=running)
    cache.put(request.app, "network", summary, settings.status_cache_seconds)
    return summary
