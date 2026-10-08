"""Quarantine, reinstatement and the security event log (ARCHITECTURE 17.6).

Quarantine is permanent until an explicit reinstatement. It (1) stops all new assignments, (2)
releases the device's active work, (3) claws back rewards for work that was never verified, and
(4) re-audits that device's unverified results in still-running tasks, re-queuing wrong ones.
"""

from typing import Any

from ..config import Settings
from ..db import Db, utcnow
from ..events import emit
from ..ids import new_id
from ..rewards import ledger
from ..scheduling.lifecycle import ACTIVE, release_attempt
from ..training import service as training
from ..trust import store
from ..verification import ACCEPTED_UNVERIFIED, VERIFIED
from ..verification.recompute import discrepancy, expected_partial, kernel_limits

REJECTED_RETRO = "rejected_retroactive"


async def record_event(
    db: Db,
    kind: str,
    *,
    severity: str = "info",
    user_id: str | None = None,
    device_id: str | None = None,
    task_id: str | None = None,
    data: dict[str, Any] | None = None,
) -> None:
    """Append-only security log (shown on /security)."""
    await db.col("security_events").insert_one(
        {
            "_id": new_id("sec"),
            "ts": utcnow(),
            "kind": kind,
            "severity": severity,  # info | warning | critical
            "user_id": user_id,
            "device_id": device_id,
            "task_id": task_id,
            "data": data or {},
        }
    )


async def quarantine_device(
    db: Db,
    device_id: str,
    *,
    reason: str,
    source: str,
    task_id: str | None = None,
    settings: Settings | None = None,
    keep_assignment: str | None = None,
) -> bool:
    """Idempotent: returns False if the device was already quarantined. `keep_assignment` is an
    assignment the caller is settling itself (the result that triggered the accusation)."""
    now = utcnow()
    res = await db.col("device_trust").update_one(
        {"_id": device_id, "status": {"$ne": store.STATUS_QUARANTINED}},
        {
            "$set": {
                "status": store.STATUS_QUARANTINED,
                "quarantine": {"at": now, "reason": reason, "source": source, "task_id": task_id},
                "updated_at": now,
            },
            "$push": {
                "history": {
                    "$each": [store.history_entry("quarantined", reason=reason, source=source)],
                    "$slice": -store.HISTORY_CAP,
                }
            },
        },
    )
    if not res.modified_count:
        return False
    device = await db.col("devices").find_one_and_update(
        {"_id": device_id}, {"$set": {"quarantined": True}}
    )
    await record_event(
        db,
        "device_quarantined",
        severity="critical",
        user_id=(device or {}).get("owner_user_id"),
        device_id=device_id,
        task_id=task_id,
        data={"reason": reason, "source": source},
    )
    await emit(
        db, "device_quarantined", device_id=device_id, task_id=task_id, data={"reason": reason}
    )

    # (2) release active work so another device takes it
    async for a in db.col("assignments").find({"device_id": device_id, "status": {"$in": ACTIVE}}):
        if a["_id"] == keep_assignment:
            continue
        chunk = await db.col("chunks").find_one({"_id": a["chunk_id"]})
        if chunk is not None:
            await release_attempt(
                db,
                a,
                chunk,
                "expired",
                {"code": "QUARANTINED", "message": "device quarantined"},
                ACTIVE,
                device_stat="expired",
            )
    # (3) clawback of never-verified rewards
    claw = await ledger.revoke_unverified(
        db, device_id, "device quarantined: unverified work forfeited"
    )
    if claw["entries"]:
        await record_event(
            db,
            "rewards_revoked",
            severity="warning",
            user_id=(device or {}).get("owner_user_id"),
            device_id=device_id,
            data=claw,
        )
    # (4) retroactive audit of unverified results in running tasks
    if settings is not None:
        await retro_audit(db, settings, device_id)
    return True


async def retro_audit(db: Db, settings: Settings, device_id: str) -> dict[str, int]:
    """Recompute this device's earlier unverified results. Wrong ones are rejected and, in a
    still-running (non-iterative) task, the chunk is re-queued. In finished/iterative tasks the
    task is flagged instead (a merged round cannot be un-merged)."""
    checked = reopened = flagged = 0
    async for pr in db.col("partial_results").find(
        {"device_id": device_id, "acceptance": ACCEPTED_UNVERIFIED}
    ):
        task = await db.col("tasks").find_one({"_id": pr["task_id"]})
        chunk = await db.col("chunks").find_one({"_id": pr["chunk_id"]})
        if task is None or chunk is None:
            continue
        if training.is_iterative(task):  # gradient payloads are dropped after a round closes
            flagged += 1
            await db.col("tasks").update_one(
                {"_id": task["_id"]},
                {
                    "$push": {
                        "integrity_flags": {
                            "at": utcnow(),
                            "device_id": device_id,
                            "chunk_index": chunk["index"],
                            "note": "device later quarantined; this unverified round was not re-run",
                        }
                    }
                },
            )
            continue
        checked += 1
        lim = kernel_limits(task)
        expected = await expected_partial(db, task, chunk)
        disc = discrepancy(task, pr["payload"], expected)
        key = pr.get("class_key") or "unknown"
        cal = await store.get_calibration(db, key)
        tol = store.tolerance_for(cal, lim.hard, lim.floor, settings)["tolerance"]
        if disc <= tol:
            await db.col("partial_results").update_one(
                {"_id": pr["_id"]},
                {"$set": {"acceptance": VERIFIED, "verified_retroactively": True}},
            )
            continue
        await db.col("partial_results").update_one(
            {"_id": pr["_id"]}, {"$set": {"acceptance": REJECTED_RETRO, "retro_discrepancy": disc}}
        )
        await ledger.revoke_entry(db, pr["assignment_id"], "failed retroactive audit")
        await db.col("verification_records").insert_one(
            {
                "_id": new_id("vr"),
                "task_id": task["_id"],
                "chunk_id": chunk["_id"],
                "assignment_id": pr["assignment_id"],
                "device_id": device_id,
                "user_id": (await _owner(db, device_id)),
                "class_key": key,
                "mode": "retroactive",
                "audited": True,
                "decision": REJECTED_RETRO,
                "discrepancy": disc,
                "tolerance": tol,
                "exceeded": True,
                "created_at": utcnow(),
            }
        )
        if task["status"] == "running" and not training.is_iterative(task):
            res = await db.col("chunks").update_one(
                {"_id": chunk["_id"], "status": "completed"},
                {
                    "$set": {
                        "status": "pending",
                        "pending_since": utcnow(),
                        "accepted_assignment_id": None,
                    },
                    "$addToSet": {"excluded_device_ids": device_id},
                    "$inc": {"max_attempts": 1},
                },
            )
            if res.modified_count:
                reopened += 1
                await emit(
                    db,
                    "chunk_reopened",
                    task_id=task["_id"],
                    chunk_id=chunk["_id"],
                    device_id=device_id,
                    data={"reason": "retroactive audit failed", "discrepancy": disc},
                )
                continue
        flagged += 1
        await db.col("tasks").update_one(
            {"_id": task["_id"]},
            {
                "$push": {
                    "integrity_flags": {
                        "at": utcnow(),
                        "device_id": device_id,
                        "chunk_index": chunk["index"],
                        "note": "contains a result later proven wrong; not re-run",
                    }
                }
            },
        )
    out = {"checked": checked, "reopened": reopened, "flagged": flagged}
    if checked:
        await record_event(db, "retroactive_audit", device_id=device_id, data=out)
    return out


async def _owner(db: Db, device_id: str) -> str | None:
    d = await db.col("devices").find_one({"_id": device_id}, {"owner_user_id": 1})
    return d["owner_user_id"] if d else None


async def reinstate_device(db: Db, device_id: str, *, by_user_id: str, reason: str) -> bool:
    """Manual decision by an administrator: clears quarantine, keeps a high suspicion memory and
    a fresh probation, so the device is audited heavily again."""
    now = utcnow()
    res = await db.col("device_trust").update_one(
        {"_id": device_id, "status": store.STATUS_QUARANTINED},
        {
            "$set": {
                "status": store.STATUS_PROBATION,
                "quarantine": None,
                "memory": 0.6,
                "suspicion": 0.0,
                "e_state": {"n": 0, "exceed": 0, "R": []},
                "evidence": 0.0,
                "results_seen": 0,
                "updated_at": now,
            },
            "$push": {
                "history": {
                    "$each": [store.history_entry("reinstated", by=by_user_id, reason=reason)],
                    "$slice": -store.HISTORY_CAP,
                }
            },
        },
    )
    if not res.modified_count:
        return False
    dev = await db.col("devices").find_one_and_update(
        {"_id": device_id}, {"$set": {"quarantined": False}}
    )
    await record_event(
        db,
        "device_reinstated",
        severity="warning",
        user_id=(dev or {}).get("owner_user_id"),
        device_id=device_id,
        data={"by": by_user_id, "reason": reason},
    )
    return True
