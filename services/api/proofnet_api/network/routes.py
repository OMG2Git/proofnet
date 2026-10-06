"""Live network summary for the /network dashboard (real device state only)."""

from fastapi import APIRouter

from ..contracts.api import NetworkDevice, NetworkSummary
from ..db import utcnow
from ..deps import DbDep, UserDep

router = APIRouter(prefix="/network", tags=["network"])

STATUSES = ("initializing", "idle", "busy", "offline", "disabled")


@router.get("/summary", response_model=NetworkSummary)
async def network_summary(db: DbDep, _: UserDep) -> NetworkSummary:
    now = utcnow()
    devices: list[NetworkDevice] = []
    counts = dict.fromkeys(STATUSES, 0)
    async for d in db.col("devices").find({}).sort("created_at", 1):
        counts[d["status"]] = counts.get(d["status"], 0) + 1
        seen = d.get("last_seen_at")
        devices.append(
            NetworkDevice(
                id=d["_id"],
                name=d["name"],
                device_type=d["device_type"],
                status=d["status"],
                score_cells_per_sec=(d.get("benchmark") or {}).get("score_cells_per_sec"),
                runtime_kind=(d.get("runtime") or {}).get("kind"),
                last_seen_age_seconds=(now - seen).total_seconds() if seen else None,
                current_assignment_id=d.get("current_assignment_id"),
            )
        )
    running = await db.col("tasks").count_documents({"status": {"$in": ["running", "aggregating"]}})
    return NetworkSummary(server_time=now, counts=counts, devices=devices, tasks_running=running)
