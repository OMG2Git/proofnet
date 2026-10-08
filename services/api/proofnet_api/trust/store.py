"""Persistent trust state: per-device profile (evidence, suspicion memory, status) and per-class
calibration of honest discrepancies. Pure PWAV math lives in `pwav`; this module only stores it.

Collections:
  device_trust   one document per device (_id = device id)
  calibrations   one document per class key "kernel@version|runtime" with the honest samples
"""

from typing import Any

from ..config import Settings
from ..db import Db, utcnow
from . import pwav

HISTORY_CAP = 60
STATUS_PROBATION, STATUS_TRUSTED = "probation", "trusted"
STATUS_WATCH, STATUS_QUARANTINED = "watch", "quarantined"
WATCH_AT = 0.25  # suspicion (or memory) at which a device is flagged "watch"


def eparams(s: Settings) -> pwav.EParams:
    return pwav.EParams(q0=s.pwav_q0, alpha=s.pwav_alpha)


def aparams(s: Settings) -> pwav.AuditParams:
    return pwav.AuditParams(
        floor=s.audit_floor,
        initial=s.audit_initial,
        probation_results=s.audit_probation_results,
        tau=s.audit_tau,
        memory_decay=s.audit_memory_decay,
    )


def class_key(task: dict[str, Any], device: dict[str, Any]) -> str:
    kind = (device.get("runtime") or {}).get("kind") or "unknown"
    return f"{task['task_type']}@{task['kernel_version']}|{kind}"


def derive_status(profile: dict[str, Any], s: Settings) -> str:
    if profile["status"] == STATUS_QUARANTINED:
        return STATUS_QUARANTINED  # sticky: only an explicit reinstatement clears it
    if profile["results_seen"] < s.audit_probation_results:
        return STATUS_PROBATION
    if max(profile["suspicion"], profile["memory"]) >= WATCH_AT:
        return STATUS_WATCH
    return STATUS_TRUSTED


def trust_of(profile: dict[str, Any]) -> float:
    return pwav.trust_score(profile["n_clean"], max(profile["suspicion"], profile["memory"]))


async def user_memory(db: Db, user_id: str, exclude_device: str) -> float:
    """A new device inherits part of the suspicion of its owner's other devices, so re-registering
    does not wipe a bad record (Sybil/whitewashing resistance)."""
    best = 0.0
    async for p in db.col("device_trust").find(
        {"user_id": user_id, "_id": {"$ne": exclude_device}}
    ):
        mem = 1.0 if p["status"] == STATUS_QUARANTINED else p["memory"]
        best = max(best, mem)
    return 0.5 * best


async def get_profile(db: Db, device: dict[str, Any], s: Settings) -> dict[str, Any]:
    p = await db.col("device_trust").find_one({"_id": device["_id"]})
    if p is not None:
        return p
    now = utcnow()
    inherited = await user_memory(db, device["owner_user_id"], device["_id"])
    doc = {
        "_id": device["_id"],
        "user_id": device["owner_user_id"],
        "status": STATUS_PROBATION,
        "results_seen": 0,
        "audits": 0,
        "n_clean": 0,
        "exceedances": 0,
        "rejected": 0,
        "e_state": pwav.new_state(),
        "evidence": 0.0,
        "suspicion": 0.0,
        "memory": inherited,
        "inherited_memory": inherited,
        "quarantine": None,
        "history": [{"at": now, "event": "created", "inherited_memory": inherited}],
        "created_at": now,
        "updated_at": now,
    }
    await db.col("device_trust").update_one({"_id": doc["_id"]}, {"$setOnInsert": doc}, upsert=True)
    return await db.col("device_trust").find_one({"_id": device["_id"]}) or doc


def audit_probability(profile: dict[str, Any], s: Settings) -> float:
    return pwav.audit_probability(
        profile["results_seen"], profile["n_clean"], profile["memory"], aparams(s)
    )


def history_entry(event: str, **data: Any) -> dict[str, Any]:
    return {"at": utcnow(), "event": event, **data}


# ------------------------------------------------------------------ calibration
async def get_calibration(db: Db, key: str) -> dict[str, Any]:
    return await db.col("calibrations").find_one({"_id": key}) or {
        "_id": key,
        "samples": [],
        "n": 0,
        "limit": None,
    }


def tolerance_for(cal: dict[str, Any], hard: float, floor: float, s: Settings) -> dict[str, Any]:
    """tolerance = min(hard, max(margin * L, floor)); `hard` until the class has enough samples.
    `hard` is the kernel's fixed numerical bound, so a poisoned calibration can never loosen it."""
    limit = cal.get("limit")
    if limit is None:
        return {"tolerance": hard, "limit": None, "source": "kernel_default"}
    return {
        "tolerance": min(hard, max(s.pwav_margin * limit, floor)),
        "limit": limit,
        "source": "calibrated",
    }


async def add_calibration_sample(
    db: Db, key: str, value: float, hard: float, s: Settings
) -> dict[str, Any]:
    """Record an honest discrepancy (only values within the hard bound are ever accepted)."""
    if value != value or value < 0 or value > hard:
        return await get_calibration(db, key)
    value = max(value, 1e-300)
    await db.col("calibrations").update_one(
        {"_id": key},
        {
            "$push": {"samples": {"$each": [value], "$slice": -s.calibration_cap}},
            "$inc": {"n": 1},
            "$set": {"updated_at": utcnow()},
        },
        upsert=True,
    )
    cal = await get_calibration(db, key)
    samples = [float(x) for x in cal["samples"]]
    limit = pwav.tolerance_limit(samples, 1.0 - s.pwav_q0, s.pwav_gamma)
    await db.col("calibrations").update_one({"_id": key}, {"$set": {"limit": limit}})
    cal["limit"] = limit
    return cal
