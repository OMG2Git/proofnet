from typing import Any

from fastapi import APIRouter, Request

from ..auth.security import hash_device_token
from ..contracts.api import DeviceOut, DevicePatch, DeviceRegistered, DeviceRegisterRequest
from ..db import Db, utcnow
from ..deps import DbDep, SettingsDep, UserDep
from ..errors import ProofNetError
from ..events import emit
from ..ids import new_id, new_token
from ..security.quarantine import record_event
from ..security.ratelimit import enforce_auth_limit

router = APIRouter(prefix="/devices", tags=["devices"])

EMPTY_STATS = {
    "succeeded": 0,
    "failed": 0,
    "expired": 0,
    "invalid_results": 0,
    "cells_processed": 0,
}


def device_out(d: dict[str, Any]) -> DeviceOut:
    return DeviceOut(
        id=d["_id"],
        name=d["name"],
        device_type=d["device_type"],
        status=d["status"],
        last_seen_at=d.get("last_seen_at"),
        capabilities=d.get("capabilities", {}),
        runtime=d.get("runtime"),
        benchmark=d.get("benchmark"),
        stats=d.get("stats", EMPTY_STATS),
    )


async def _owned(db: Db, device_id: str, user_id: str) -> dict[str, Any]:
    d = await db.col("devices").find_one({"_id": device_id, "owner_user_id": user_id})
    if d is None:
        raise ProofNetError(404, "NOT_FOUND", "Device not found")
    return d


@router.post("", response_model=DeviceRegistered, status_code=201)
async def register_device(
    body: DeviceRegisterRequest, request: Request, db: DbDep, user: UserDep, settings: SettingsDep
) -> DeviceRegistered:
    await enforce_auth_limit(request, db, settings, "register")
    if await db.col("devices").count_documents({"owner_user_id": user["_id"]}) >= (
        settings.max_devices_per_user
    ):
        await record_event(
            db, "device_cap_reached", severity="warning", user_id=user["_id"], data={}
        )
        raise ProofNetError(
            409, "DEVICE_LIMIT", f"At most {settings.max_devices_per_user} devices per account"
        )
    token = new_token()
    doc = {
        "_id": new_id("dev"),
        "owner_user_id": user["_id"],
        "name": body.name,
        "device_type": body.device_type.value,
        "capabilities": body.capabilities.model_dump(),
        "runtime": None,
        "benchmark": None,
        "status": "initializing",
        "session_id": None,
        "last_seen_at": None,
        "current_assignment_id": None,
        "token_hash": hash_device_token(token),
        "stats": dict(EMPTY_STATS),
        "created_at": utcnow(),
    }
    await db.col("devices").insert_one(doc)
    await emit(db, "device_registered", device_id=doc["_id"], data={"name": body.name})
    return DeviceRegistered(device=device_out(doc), device_token=token)


@router.get("/mine", response_model=list[DeviceOut])
async def my_devices(db: DbDep, user: UserDep) -> list[DeviceOut]:
    cursor = db.col("devices").find({"owner_user_id": user["_id"]}).sort("created_at", 1)
    return [device_out(d) async for d in cursor]


@router.patch("/{device_id}", response_model=DeviceOut)
async def patch_device(device_id: str, body: DevicePatch, db: DbDep, user: UserDep) -> DeviceOut:
    await _owned(db, device_id, user["_id"])
    if body.name is not None:
        await db.col("devices").update_one({"_id": device_id}, {"$set": {"name": body.name}})
    if body.disabled is True:
        await db.col("devices").update_one(
            {"_id": device_id, "status": {"$in": ["idle", "offline", "initializing"]}},
            {"$set": {"status": "disabled"}},
        )
        await emit(db, "device_disabled", device_id=device_id)
    elif body.disabled is False:
        # disabled -> initializing; the device must start a new session to become idle
        await db.col("devices").update_one(
            {"_id": device_id, "status": "disabled"},
            {"$set": {"status": "initializing", "session_id": None}},
        )
        await emit(db, "device_enabled", device_id=device_id)
    return device_out(await _owned(db, device_id, user["_id"]))
