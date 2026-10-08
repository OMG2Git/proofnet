"""Post-hoc forensics: when the end-to-end reference check of a finished task fails, some
unaudited chunk must be wrong. Recompute every chunk, name the culprits, quarantine their
devices, revoke their rewards and re-run the affected chunks (the task returns to `running`)."""

from typing import Any

from ..config import Settings
from ..db import Db, utcnow
from ..events import emit
from ..ids import new_id
from ..rewards import ledger
from ..security.quarantine import quarantine_device, record_event
from . import REJECTED_VERIFICATION
from .recompute import discrepancy, expected_partial, kernel_limits

REJECTED_FORENSIC = "rejected_forensic"


async def heal_after_failed_check(
    db: Db,
    settings: Settings,
    task: dict[str, Any],
    chunks: list[dict[str, Any]],
    partials: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Returns the culprits; when there are any, their chunks are re-queued and the task is
    moved aggregating -> running (idempotent: guarded by the transition)."""
    lim = kernel_limits(task)
    culprits: list[dict[str, Any]] = []
    for chunk, pr in zip(chunks, partials, strict=True):
        expected = await expected_partial(db, task, chunk)
        disc = discrepancy(task, pr["payload"], expected)
        if disc > lim.hard:  # a deterministic proof: honest runtimes never get near this bound
            culprits.append({"chunk": chunk, "pr": pr, "discrepancy": disc})
    if not culprits:
        return []
    res = await db.col("tasks").update_one(
        {"_id": task["_id"], "status": "aggregating"},
        {
            "$set": {"status": "running", "aggregation_claimed_at": None},
            "$push": {"status_history": {"status": "running", "at": utcnow()}},
        },
    )
    if not res.modified_count:
        return []
    await record_event(
        db,
        "reference_check_failed",
        severity="critical",
        user_id=task["owner_user_id"],
        task_id=task["_id"],
        data={"culprit_chunks": [c["chunk"]["index"] for c in culprits]},
    )
    for c in culprits:
        chunk, pr = c["chunk"], c["pr"]
        await db.col("partial_results").update_one(
            {"_id": pr["_id"]}, {"$set": {"acceptance": REJECTED_FORENSIC}}
        )
        await db.col("chunks").update_one(
            {"_id": chunk["_id"], "status": "completed"},
            {
                "$set": {
                    "status": "pending",
                    "pending_since": utcnow(),
                    "accepted_assignment_id": None,
                },
                "$addToSet": {"excluded_device_ids": pr["device_id"]},
                "$inc": {"max_attempts": 1},
            },
        )
        await ledger.revoke_entry(db, pr["assignment_id"], "failed the end-to-end reference check")
        await db.col("verification_records").insert_one(
            {
                "_id": new_id("vr"),
                "task_id": task["_id"],
                "chunk_id": chunk["_id"],
                "assignment_id": pr["assignment_id"],
                "device_id": pr["device_id"],
                "user_id": task["owner_user_id"],
                "class_key": pr.get("class_key"),
                "mode": "forensic",
                "audited": True,
                "decision": REJECTED_VERIFICATION,
                "discrepancy": c["discrepancy"],
                "tolerance": lim.hard,
                "exceeded": True,
                "created_at": utcnow(),
            }
        )
        await db.col("device_trust").update_one(
            {"_id": pr["device_id"]}, {"$inc": {"exceedances": 1, "rejected": 1}}
        )
        await emit(
            db,
            "chunk_reopened",
            task_id=task["_id"],
            chunk_id=chunk["_id"],
            device_id=pr["device_id"],
            data={
                "reason": "reference check failed; chunk recomputed",
                "discrepancy": c["discrepancy"],
            },
        )
        await quarantine_device(
            db,
            pr["device_id"],
            reason=(
                f"returned a wrong result for chunk {chunk['index']} that was not audited; "
                f"found by the end-to-end reference check (discrepancy {c['discrepancy']:.3g})"
            ),
            source="forensics",
            task_id=task["_id"],
            settings=settings,
        )
    return culprits
