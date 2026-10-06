"""Worker gateway. P2: runtime manifest/bundle, session, heartbeat (no dispatch yet)."""

from fastapi import APIRouter, Request, Response

from ..contracts import HeartbeatRequest, HeartbeatResponse
from ..contracts.api import RuntimeManifest, SessionRequest, SessionResponse
from ..db import utcnow
from ..deps import DbDep, DeviceDep, SettingsDep
from ..errors import ProofNetError
from ..events import emit
from ..ids import new_id
from .runtime import NUMPY_VERSION, PYODIDE_VERSION, KernelBundle

router = APIRouter(tags=["worker"])


def _bundle(request: Request) -> KernelBundle:
    b: KernelBundle = request.app.state.bundle
    return b


@router.get("/runtime/manifest", response_model=RuntimeManifest)
async def runtime_manifest(
    request: Request, _: DeviceDep, settings: SettingsDep
) -> RuntimeManifest:
    b = _bundle(request)
    return RuntimeManifest(
        pyodide_version=PYODIDE_VERSION,
        numpy_version=NUMPY_VERSION,
        kernel_bundle_version=b.version,
        kernel_bundle_sha256=b.sha256,
        heartbeat_idle_ms=settings.heartbeat_idle_ms,
        heartbeat_busy_ms=settings.heartbeat_busy_ms,
        offline_after_seconds=settings.offline_after_seconds,
    )


@router.get("/runtime/kernels/{version}")
async def kernel_bundle(version: str, request: Request, _: DeviceDep) -> Response:
    b = _bundle(request)
    if version != b.version:
        raise ProofNetError(404, "NOT_FOUND", f"Unknown kernel bundle version '{version}'")
    return Response(content=b.data, media_type="application/zip", headers={"ETag": b.sha256})


@router.post("/worker/session", response_model=SessionResponse)
async def start_session(
    body: SessionRequest, db: DbDep, device: DeviceDep, settings: SettingsDep
) -> SessionResponse:
    if device["status"] == "disabled":
        raise ProofNetError(403, "DEVICE_DISABLED", "This device has been disabled by its owner")
    session_id = new_id("pr")  # opaque session id
    caps = {**device.get("capabilities", {}), **body.capabilities.model_dump(exclude_none=True)}
    await db.col("devices").update_one(
        {"_id": device["_id"], "status": {"$ne": "disabled"}},
        {
            "$set": {
                "status": "idle",
                "session_id": session_id,
                "last_seen_at": utcnow(),
                "runtime": body.runtime.model_dump(),
                "benchmark": {**body.benchmark.model_dump(), "measured_at": utcnow()},
                "capabilities": caps,
                "current_assignment_id": None,
            }
        },
    )
    await emit(
        db,
        "device_session_started",
        device_id=device["_id"],
        data={"score": body.benchmark.score_cells_per_sec},
    )
    return SessionResponse(
        session_id=session_id,
        heartbeat_idle_ms=settings.heartbeat_idle_ms,
        heartbeat_busy_ms=settings.heartbeat_busy_ms,
    )


@router.post("/worker/heartbeat", response_model=HeartbeatResponse)
async def heartbeat(
    body: HeartbeatRequest, db: DbDep, device: DeviceDep, settings: SettingsDep
) -> HeartbeatResponse:
    now = utcnow()
    if device["status"] == "disabled" or device.get("session_id") != body.session_id:
        # Stale or unknown session: tell the worker to start a new one (no state change).
        return HeartbeatResponse(
            server_time=now,
            next_heartbeat_ms=settings.heartbeat_idle_ms,
            directives=[{"type": "reregister"}],
        )
    update: dict[str, object] = {"last_seen_at": now}
    if body.battery is not None:
        update["capabilities.battery"] = body.battery
    # offline -> idle recovery is conditional so concurrent heartbeats apply it once
    recovered = await db.col("devices").update_one(
        {"_id": device["_id"], "session_id": body.session_id, "status": "offline"},
        {"$set": {**update, "status": "idle"}},
    )
    if recovered.modified_count:
        await emit(db, "device_online", device_id=device["_id"])
    else:
        await db.col("devices").update_one(
            {"_id": device["_id"], "session_id": body.session_id}, {"$set": update}
        )
    busy = device["status"] == "busy"
    return HeartbeatResponse(
        server_time=now,
        next_heartbeat_ms=settings.heartbeat_busy_ms if busy else settings.heartbeat_idle_ms,
        directives=[],
    )
