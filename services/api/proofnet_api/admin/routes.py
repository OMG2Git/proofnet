"""Admin: demo reset (ARCHITECTURE 13). Clears work data (tasks, results, verification records,
rewards), keeps users, devices, trust profiles (incl. quarantine), calibrations and security events."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from ..config import Settings
from ..db import Db
from ..deps import DbDep, SettingsDep, UserDep
from ..errors import ProofNetError
from ..events import emit

router = APIRouter(prefix="/admin", tags=["admin"])

# Collections that hold task/work data. users and devices are kept (so phones stay registered).
WORK_COLLECTIONS = (
    "tasks",
    "chunks",
    "assignments",
    "partial_results",
    "artifacts",
    "events",
    "datasets",
    "image_datasets",
    "model_states",
    "training_rounds",
    "verification_records",
    "reward_entries",
    "reward_events",
)


def is_admin(user: dict[str, Any], settings: Settings) -> bool:
    listed = {e.strip().lower() for e in settings.admin_emails.split(",") if e.strip()}
    return "admin" in user.get("roles", []) or user["email"].lower() in listed


async def admin_user(user: UserDep, settings: SettingsDep) -> dict[str, Any]:
    if not is_admin(user, settings):
        raise ProofNetError(403, "FORBIDDEN", "Admin only")
    return user


AdminDep = Annotated[dict[str, Any], Depends(admin_user)]


async def reset_demo_data(db: Db) -> dict[str, int]:
    """Delete all tasks/datasets/results (and their GridFS files); free every device."""
    counts: dict[str, int] = {}
    for name in WORK_COLLECTIONS:
        counts[name] = (await db.col(name).delete_many({})).deleted_count
    files = (await db.col("fs.files").delete_many({})).deleted_count
    await db.col("fs.chunks").delete_many({})
    counts["files"] = files
    # a device that was working goes back to idle; offline/disabled devices keep their status
    freed = await db.col("devices").update_many({"status": "busy"}, {"$set": {"status": "idle"}})
    await db.col("devices").update_many({}, {"$set": {"current_assignment_id": None}})
    counts["devices_freed"] = freed.modified_count
    return counts


@router.post("/demo/reset")
async def demo_reset(db: DbDep, _: AdminDep) -> dict[str, Any]:
    counts = await reset_demo_data(db)
    await emit(db, "demo_reset", data=counts)
    return {"reset": True, "deleted": counts}


@router.get("/me")
async def admin_me(user: UserDep, settings: SettingsDep) -> dict[str, bool]:
    """Lets the UI show admin-only controls."""
    return {"admin": is_admin(user, settings)}
