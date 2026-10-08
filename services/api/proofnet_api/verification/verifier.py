"""The verifier: decides, per incoming result, whether to audit it, audits by backend
recomputation, and updates the device's trust state (ARCHITECTURE 17).

Order inside result intake: structural checks -> verify (this module) -> assignment succeeded.
A failed audit therefore rejects the attempt exactly like a structural failure: the chunk is
re-queued for another device and the contributor only ever sees a generic INVALID_RESULT.
"""

import hashlib
import hmac
from dataclasses import dataclass
from typing import Any

from ..config import Settings
from ..db import Db, utcnow
from ..events import emit
from ..ids import new_id
from ..trust import pwav, store
from . import ACCEPTED_UNVERIFIED, REJECTED_VERIFICATION, VERIFIED, normalize_mode
from .recompute import discrepancy, expected_partial, kernel_limits


@dataclass
class Decision:
    acceptance: str
    audited: bool
    record_id: str | None = None
    exceeded: bool = False
    quarantined: bool = False
    discrepancy: float | None = None
    tolerance: float | None = None


def audit_draw(settings: Settings, assignment_id: str) -> float:
    """Reproducible uniform draw in [0, 1) per assignment (HMAC with a server-side key): the
    worker cannot predict it, and a record can be re-checked later with the same key."""
    key = hashlib.sha256(("proofnet-audit|" + settings.jwt_secret).encode()).digest()
    mac = hmac.new(key, assignment_id.encode(), hashlib.sha256).digest()
    return int.from_bytes(mac[:8], "big") / 2**64


async def verify_result(
    db: Db,
    settings: Settings,
    task: dict[str, Any],
    chunk: dict[str, Any],
    asg: dict[str, Any],
    device: dict[str, Any],
    payload: dict[str, Any],
) -> Decision:
    """Returns the decision. Updates the trust profile; never raises on a wrong result (the caller
    rejects the attempt when `acceptance == REJECTED_VERIFICATION`)."""
    mode = normalize_mode(task.get("verification_policy"), settings.verification_default_mode)
    profile = await store.get_profile(db, device, settings)
    if mode == "off":
        await db.col("device_trust").update_one(
            {"_id": device["_id"]}, {"$inc": {"results_seen": 1}, "$set": {"updated_at": utcnow()}}
        )
        return Decision(ACCEPTED_UNVERIFIED, audited=False)

    prob = 1.0 if mode == "full" else store.audit_probability(profile, settings)
    draw = audit_draw(settings, asg["_id"])
    audited = draw < prob
    now = utcnow()
    record = {
        "_id": new_id("vr"),
        "task_id": task["_id"],
        "chunk_id": chunk["_id"],
        "assignment_id": asg["_id"],
        "device_id": device["_id"],
        "user_id": device["owner_user_id"],
        "class_key": store.class_key(task, device),
        "mode": mode,
        "audit_probability": prob,
        "draw": draw,
        "audited": audited,
        "results_seen_before": profile["results_seen"],
        "created_at": now,
    }

    if not audited:
        await db.col("verification_records").insert_one({**record, "decision": ACCEPTED_UNVERIFIED})
        await db.col("device_trust").update_one(
            {"_id": device["_id"]},
            {
                "$inc": {"results_seen": 1},
                "$set": {"updated_at": now},
            },
        )
        return Decision(ACCEPTED_UNVERIFIED, audited=False, record_id=record["_id"])

    # ---------------- audit by recomputation (collusion-immune: the oracle is the backend)
    lim = kernel_limits(task)
    expected = await expected_partial(db, task, chunk)
    disc = discrepancy(task, payload, expected)
    cal = await store.get_calibration(db, record["class_key"])
    tol = store.tolerance_for(cal, lim.hard, lim.floor, settings)
    exceeded = disc > tol["tolerance"]
    z = int(exceeded)

    ep = store.eparams(settings)
    ap = store.aparams(settings)
    state = profile["e_state"]
    summ = pwav.update(state, z, ep)
    memory = pwav.decay_memory(profile["memory"], summ["suspicion"], ap)
    accused = bool(summ["accused"])

    # only results from devices not under suspicion feed the honest-noise calibration
    if (
        not exceeded
        and profile["status"] != store.STATUS_QUARANTINED
        and max(profile["suspicion"], profile["memory"]) < store.WATCH_AT
    ):
        await store.add_calibration_sample(db, record["class_key"], disc, lim.hard, settings)

    new_profile = {
        **profile,
        "results_seen": profile["results_seen"] + 1,
        "n_clean": profile["n_clean"] + (0 if exceeded else 1),
        "suspicion": summ["suspicion"],
        "memory": memory,
    }
    status = store.derive_status(new_profile, settings)  # quarantine itself is applied below
    update: dict[str, Any] = {
        "$set": {
            "e_state": state,
            "evidence": summ["evidence"],
            "suspicion": summ["suspicion"],
            "memory": memory,
            "status": status,
            "updated_at": now,
        },
        "$inc": {
            "results_seen": 1,
            "audits": 1,
            "n_clean": 0 if exceeded else 1,
            "exceedances": z,
            "rejected": z,
        },
        "$push": {
            "history": {
                "$each": [
                    store.history_entry(
                        "audit_fail" if exceeded else "audit_pass",
                        discrepancy=disc,
                        tolerance=tol["tolerance"],
                        suspicion=summ["suspicion"],
                        task_id=task["_id"],
                    )
                ],
                "$slice": -store.HISTORY_CAP,
            }
        },
    }
    await db.col("device_trust").update_one({"_id": device["_id"]}, update)

    decision_name = REJECTED_VERIFICATION if exceeded else VERIFIED
    await db.col("verification_records").insert_one(
        {
            **record,
            "decision": decision_name,
            "discrepancy": disc,
            "tolerance": tol["tolerance"],
            "tolerance_source": tol["source"],
            "calibration_limit": tol["limit"],
            "calibration_n": cal.get("n", 0),
            "exceeded": exceeded,
            "evidence": summ["evidence"],
            "threshold": summ["threshold"],
            "suspicion": summ["suspicion"],
            "accused": accused,
        }
    )
    await emit(
        db,
        "result_audited",
        task_id=task["_id"],
        device_id=device["_id"],
        chunk_id=chunk["_id"],
        assignment_id=asg["_id"],
        data={
            "passed": not exceeded,
            "discrepancy": disc,
            "tolerance": tol["tolerance"],
            "suspicion": summ["suspicion"],
        },
    )
    if accused:
        from ..security.quarantine import quarantine_device

        await quarantine_device(
            db,
            device["_id"],
            reason=(
                f"evidence {summ['evidence']:.3g} reached the accusation threshold "
                f"{summ['threshold']:.3g} after {summ['n']} audits "
                f"(lifetime false-accusation bound {settings.pwav_alpha:g})"
            ),
            source="pwav",
            task_id=task["_id"],
            settings=settings,
            keep_assignment=asg["_id"],
        )
    return Decision(
        decision_name,
        audited=True,
        record_id=record["_id"],
        exceeded=exceeded,
        quarantined=accused,
        discrepancy=disc,
        tolerance=tol["tolerance"],
    )
