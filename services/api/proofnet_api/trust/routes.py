"""Trust dashboard API: per-device evidence, calibration, verification records, simulator."""

import asyncio
from typing import Any

from fastapi import APIRouter

from proofnet_kernels.server.registry import get_kernel

from ..admin.routes import is_admin
from ..config import Settings
from ..contracts.part2 import (
    CalibrationClass,
    DeviceTrust,
    SimulateRequest,
    SimulationOut,
    TrustEvent,
    TrustOverview,
    VerificationRecord,
)
from ..db import Db
from ..deps import DbDep, SettingsDep, UserDep
from ..errors import ProofNetError
from . import pwav, simulator, store

router = APIRouter(prefix="/trust", tags=["trust"])


def device_view(profile: dict[str, Any], device: dict[str, Any], s: Settings) -> DeviceTrust:
    ep = store.eparams(s)
    summ = pwav.summary(profile["e_state"], ep)
    n = max(summ["n"], 1)
    h = pwav.threshold(n, ep)
    ev = float(profile["evidence"])
    return DeviceTrust(
        device_id=device["_id"],
        device_name=device["name"],
        runtime_kind=(device.get("runtime") or {}).get("kind"),
        status=profile["status"],
        results_seen=profile["results_seen"],
        audits=profile["audits"],
        n_clean=profile["n_clean"],
        exceedances=profile["exceedances"],
        evidence=ev,
        threshold=h,
        log10_ratio=max(summ["log10_ratio"], -50.0),
        suspicion=profile["suspicion"],
        memory=profile["memory"],
        trust=store.trust_of(profile),
        audit_probability=store.audit_probability(profile, s),
        reward_multiplier=pwav.reward_multiplier(store.trust_of(profile), s.reward_base),
        quarantine=profile.get("quarantine"),
        history=[TrustEvent(**e) for e in reversed(profile.get("history", [])[-25:])],
    )


def record_view(r: dict[str, Any], names: dict[str, str]) -> VerificationRecord:
    return VerificationRecord(
        id=r["_id"],
        at=r["created_at"],
        task_id=r["task_id"],
        device_id=r["device_id"],
        device_name=names.get(r["device_id"]),
        mode=r["mode"],
        audit_probability=r.get("audit_probability"),
        draw=r.get("draw"),
        audited=r["audited"],
        decision=r["decision"],
        discrepancy=r.get("discrepancy"),
        tolerance=r.get("tolerance"),
        tolerance_source=r.get("tolerance_source"),
        exceeded=r.get("exceeded"),
        evidence=r.get("evidence"),
        suspicion=r.get("suspicion"),
        class_key=r.get("class_key"),
    )


async def calibration_view(db: Db, s: Settings) -> list[CalibrationClass]:
    out: list[CalibrationClass] = []
    p = 1.0 - s.pwav_q0
    need = pwav.min_samples(p, s.pwav_gamma)
    async for c in db.col("calibrations").find({}):
        key = c["_id"]
        kernel_key = key.split("|")[0]
        try:
            k = get_kernel(kernel_key)
            hard: float | None = float(
                getattr(k.server, "GRADIENT_TOLERANCE", None) or k.server.TOLERANCE
            )
            floor: float | None = float(k.server.AUDIT_FLOOR)
        except (KeyError, AttributeError):
            hard = floor = None
        samples = sorted(float(x) for x in c.get("samples", []))
        tol = (
            store.tolerance_for(c, hard, floor, s)
            if hard is not None and floor is not None
            else None
        )
        out.append(
            CalibrationClass(
                class_key=key,
                n=len(samples),
                limit=c.get("limit"),
                min_samples=need,
                p=p,
                gamma=s.pwav_gamma,
                margin=s.pwav_margin,
                tolerance_floor=floor,
                tolerance_hard=hard,
                tolerance=tol["tolerance"] if tol else None,
                source=tol["source"] if tol else "unknown",
                sample_min=samples[0] if samples else None,
                sample_median=samples[len(samples) // 2] if samples else None,
                sample_max=samples[-1] if samples else None,
            )
        )
    return out


@router.get("/overview", response_model=TrustOverview)
async def overview(db: DbDep, user: UserDep, settings: SettingsDep) -> TrustOverview:
    admin = is_admin(user, settings)
    dq: dict[str, Any] = {} if admin else {"owner_user_id": user["_id"]}
    devices = {d["_id"]: d async for d in db.col("devices").find(dq)}
    profiles = {
        p["_id"]: p async for p in db.col("device_trust").find({"_id": {"$in": list(devices)}})
    }
    views = [
        device_view(profiles[i], d, settings)
        for i, d in sorted(devices.items(), key=lambda kv: kv[1]["name"])
        if i in profiles
    ]
    names = {i: d["name"] for i, d in devices.items()}
    recent = [
        record_view(r, names)
        async for r in db.col("verification_records")
        .find({"device_id": {"$in": list(devices)}})
        .sort("created_at", -1)
        .limit(40)
    ]
    n_status = {
        k: sum(1 for v in views if v.status == k)
        for k in ("probation", "trusted", "watch", "quarantined")
    }
    results = sum(v.results_seen for v in views)
    audits = sum(v.audits for v in views)
    return TrustOverview(
        devices=len(views),
        **n_status,
        results_seen=results,
        audits=audits,
        exceedances=sum(v.exceedances for v in views),
        audit_rate=audits / results if results else 0.0,
        parameters={
            "q0": settings.pwav_q0,
            "p": 1.0 - settings.pwav_q0,
            "gamma": settings.pwav_gamma,
            "alpha": settings.pwav_alpha,
            "margin": settings.pwav_margin,
            "audit_floor": settings.audit_floor,
            "audit_initial": settings.audit_initial,
            "probation_results": float(settings.audit_probation_results),
            "memory_decay": settings.audit_memory_decay,
            "reward_base": settings.reward_base,
        },
        devices_detail=views,
        calibration=await calibration_view(db, settings),
        recent_records=recent,
    )


@router.get("/devices/{device_id}", response_model=DeviceTrust)
async def device_detail(
    device_id: str, db: DbDep, user: UserDep, settings: SettingsDep
) -> DeviceTrust:
    device = await db.col("devices").find_one({"_id": device_id})
    if device is None or (device["owner_user_id"] != user["_id"] and not is_admin(user, settings)):
        raise ProofNetError(404, "NOT_FOUND", "Device not found")
    profile = await store.get_profile(db, device, settings)
    return device_view(profile, device, settings)


@router.get("/records", response_model=list[VerificationRecord])
async def records(
    db: DbDep, user: UserDep, settings: SettingsDep, task_id: str | None = None, limit: int = 100
) -> list[VerificationRecord]:
    admin = is_admin(user, settings)
    q: dict[str, Any] = {}
    if task_id:
        task = await db.col("tasks").find_one({"_id": task_id})
        if task is None or (task["owner_user_id"] != user["_id"] and not admin):
            raise ProofNetError(404, "NOT_FOUND", "Task not found")
        q["task_id"] = task_id
    elif not admin:
        mine = [
            d["_id"]
            async for d in db.col("devices").find({"owner_user_id": user["_id"]}, {"_id": 1})
        ]
        q["device_id"] = {"$in": mine}
    docs = [
        r
        async for r in db.col("verification_records")
        .find(q)
        .sort("created_at", -1)
        .limit(max(1, min(limit, 300)))
    ]
    ids = list({r["device_id"] for r in docs})
    names = {d["_id"]: d["name"] async for d in db.col("devices").find({"_id": {"$in": ids}})}
    return [record_view(r, names) for r in docs]


@router.post("/simulate", response_model=SimulationOut)
async def simulate(body: SimulateRequest, user: UserDep) -> SimulationOut:
    """Population simulation of the audit policy (pure model; never touches live data)."""
    kw = body.model_dump(exclude={"seed"}) | {"seed": body.seed}
    out = await asyncio.to_thread(simulator.simulate_all, **_sim_kwargs(kw))
    return SimulationOut(request=body, **out)


def _sim_kwargs(kw: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "honest",
        "attackers",
        "rounds",
        "policy",
        "alpha",
        "q0",
        "audit_floor",
        "attack_strength",
        "cheat_rate",
        "sleeper_after",
        "fixed_rate",
        "seed",
    ]
    return {k: kw[k] for k in keys}
