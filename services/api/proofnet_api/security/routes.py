"""Security API: event log, overview, manual quarantine / reinstatement (admin)."""

from datetime import timedelta
from typing import Any

from fastapi import APIRouter

from ..admin.routes import AdminDep, is_admin
from ..config import Settings
from ..contracts.part2 import ReasonBody, SecurityEvent, SecurityOverview
from ..db import utcnow
from ..deps import DbDep, SettingsDep, UserDep
from ..errors import ProofNetError
from ..trust import store
from ..trust.routes import device_view
from .quarantine import quarantine_device, reinstate_device

router = APIRouter(prefix="/security", tags=["security"])


def _event(e: dict[str, Any]) -> SecurityEvent:
    return SecurityEvent(
        id=e["_id"],
        ts=e["ts"],
        kind=e["kind"],
        severity=e["severity"],
        device_id=e.get("device_id"),
        task_id=e.get("task_id"),
        data=e.get("data", {}),
    )


def controls(s: Settings) -> list[dict[str, str]]:
    """What protects the system (shown on the dashboard so the claims can be checked in code)."""
    return [
        {
            "name": "Audit by recomputation",
            "detail": "The backend recomputes sampled results itself; colluding devices cannot "
            "vouch for each other.",
        },
        {
            "name": "Adaptive audit probability",
            "detail": f"New devices: every result for their first {s.audit_probation_results} "
            f"results; then decaying to a floor of {s.audit_floor:.0%}; suspicion raises it "
            "back towards 100%.",
        },
        {
            "name": "Evidence-based quarantine",
            "detail": "Accusation only when betting evidence passes a threshold calibrated so an "
            f"honest device is wrongly accused with lifetime probability <= {s.pwav_alpha:g}.",
        },
        {
            "name": "End-to-end reference check",
            "detail": "Every finished task is compared with a centralized recomputation; a mismatch "
            "triggers forensics, quarantine and re-computation of the bad chunk.",
        },
        {
            "name": "Reward clawback",
            "detail": "Work that was never verified is revoked when its device is quarantined; "
            "rewards scale with earned trust.",
        },
        {
            "name": "Sybil resistance",
            "detail": "A new device inherits half of the suspicion of its owner's other devices.",
        },
        {
            "name": "Login lockout and rate limits",
            "detail": f"{s.login_max_failures} failed sign-ins lock the account for "
            f"{s.login_lockout_seconds} s; sign-up/sign-in limited to "
            f"{s.rate_limit_auth_per_minute} requests/min per address.",
        },
        {
            "name": "No code execution, no pickle",
            "detail": "Workers only run versioned kernels from the server's own bundle; results "
            "are JSON with a SHA-256 over canonical bytes; device tokens are stored hashed.",
        },
        {
            "name": "Tenant isolation",
            "detail": "A device can only touch its own assignments; users only their own tasks, "
            "datasets and rewards.",
        },
    ]


@router.get("/overview", response_model=SecurityOverview)
async def overview(db: DbDep, user: UserDep, settings: SettingsDep) -> SecurityOverview:
    admin = is_admin(user, settings)
    own = [
        d["_id"] async for d in db.col("devices").find({"owner_user_id": user["_id"]}, {"_id": 1})
    ]
    q: dict[str, Any] = (
        {} if admin else {"$or": [{"user_id": user["_id"]}, {"device_id": {"$in": own}}]}
    )
    events = [_event(e) async for e in db.col("security_events").find(q).sort("ts", -1).limit(60)]
    since = utcnow() - timedelta(hours=24)
    counts: dict[str, int] = {}
    async for r in await db.col("security_events").aggregate(
        [{"$match": {**q, "ts": {"$gte": since}}}, {"$group": {"_id": "$kind", "n": {"$sum": 1}}}]
    ):
        counts[r["_id"]] = r["n"]
    dq: dict[str, Any] = {"status": store.STATUS_QUARANTINED}
    if not admin:
        dq["_id"] = {"$in": own}
    quarantined = []
    async for p in db.col("device_trust").find(dq):
        d = await db.col("devices").find_one({"_id": p["_id"]})
        if d:
            quarantined.append(device_view(p, d, settings))
    return SecurityOverview(
        quarantined_devices=quarantined,
        events=events,
        counts=counts,
        controls=controls(settings),
    )


@router.post("/devices/{device_id}/reinstate")
async def reinstate(
    device_id: str, body: ReasonBody, db: DbDep, admin: AdminDep
) -> dict[str, bool]:
    ok = await reinstate_device(db, device_id, by_user_id=admin["_id"], reason=body.reason)
    if not ok:
        raise ProofNetError(409, "CONFLICT", "Device is not quarantined")
    return {"reinstated": True}


@router.post("/devices/{device_id}/quarantine")
async def quarantine(
    device_id: str, body: ReasonBody, db: DbDep, admin: AdminDep, settings: SettingsDep
) -> dict[str, bool]:
    device = await db.col("devices").find_one({"_id": device_id})
    if device is None:
        raise ProofNetError(404, "NOT_FOUND", "Device not found")
    await store.get_profile(db, device, settings)
    ok = await quarantine_device(
        db,
        device_id,
        reason=f"manual: {body.reason}",
        source=f"admin:{admin['email']}",
        settings=settings,
    )
    if not ok:
        raise ProofNetError(409, "CONFLICT", "Device is already quarantined")
    return {"quarantined": True}
