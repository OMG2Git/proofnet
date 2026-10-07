"""Assignment protocol (ARCHITECTURE 8.3, 9.2, 11): directive, start, input, result, fail.

All transitions are conditional atomic updates; every write endpoint is idempotent per assignment.
"""

import asyncio
from typing import Any

from fastapi import APIRouter, Request, Response

from proofnet_kernels.core.serialize import payload_sha256
from proofnet_kernels.server.registry import Kernel, get_kernel

from ..aggregation.service import aggregate_task
from ..background import kick_scheduler, spawn
from ..config import Settings
from ..contracts import AssignmentPayload, FailRequest, PartialResultEnvelope, RunDirective
from ..contracts.api import ResultAck
from ..db import Db, utcnow
from ..deps import DbDep, DeviceDep, SettingsDep
from ..errors import ProofNetError
from ..events import emit
from ..ids import new_id
from ..tasks.prepared import load_task_prepared
from ..verification import DEFAULT_HOOK

router = APIRouter(prefix="/worker/assignments", tags=["worker"])


def worker_params(task: dict[str, Any]) -> dict[str, Any]:
    if task["task_type"] == "gaussian_nb_train":
        return {"n_classes": len(task["prepared"]["class_labels"])}
    return {}


async def run_directive_for(db: Db, device: dict[str, Any]) -> RunDirective | None:
    """The `run` directive for a device's assignment that has not been started yet."""
    asg = await db.col("assignments").find_one(
        {"device_id": device["_id"], "status": "assigned"}, sort=[("assigned_at", 1)]
    )
    if asg is None:
        return None
    chunk = await db.col("chunks").find_one({"_id": asg["chunk_id"]})
    task = await db.col("tasks").find_one({"_id": asg["task_id"]})
    if chunk is None or task is None:
        return None
    return RunDirective(
        assignment=AssignmentPayload(
            assignment_id=asg["_id"],
            task_id=task["_id"],
            chunk_id=chunk["_id"],
            kernel=task["task_type"],
            kernel_version=task["kernel_version"],
            params=worker_params(task),
            input_url=f"/worker/assignments/{asg['_id']}/input",
            input_sha256=chunk["input_sha256"],
            n_rows=chunk["n_rows"],
            n_features=task["prepared"]["n_features"],
            deadline_at=asg["deadline_at"],
        )
    )


async def _own_assignment(db: Db, assignment_id: str, device: dict[str, Any]) -> dict[str, Any]:
    asg = await db.col("assignments").find_one({"_id": assignment_id})
    # Workers can only access their own assignments: unknown and foreign look identical.
    if asg is None or asg["device_id"] != device["_id"]:
        raise ProofNetError(404, "NOT_FOUND", "Assignment not found")
    return asg


def _kick(request: Request, settings: Settings) -> None:
    kick_scheduler(request.app, settings)


def _spawn(request: Request, coro: Any) -> None:
    spawn(request.app, coro)


@router.post("/{assignment_id}/start", response_model=ResultAck)
async def start_assignment(assignment_id: str, db: DbDep, device: DeviceDep) -> ResultAck:
    asg = await _own_assignment(db, assignment_id, device)
    now = utcnow()
    res = await db.col("assignments").update_one(
        {"_id": assignment_id, "status": "assigned"},
        {"$set": {"status": "running", "started_at": now}},
    )
    if res.modified_count:
        await emit(
            db,
            "assignment_started",
            task_id=asg["task_id"],
            device_id=device["_id"],
            chunk_id=asg["chunk_id"],
            assignment_id=assignment_id,
        )
        return ResultAck(assignment_id=assignment_id, status="running")
    current = await db.col("assignments").find_one({"_id": assignment_id})
    if current and current["status"] == "running":
        return ResultAck(assignment_id=assignment_id, status="running")  # idempotent
    raise ProofNetError(
        409, "CONFLICT", f"Assignment is {current['status'] if current else 'gone'}; cannot start"
    )


@router.get("/{assignment_id}/input")
async def assignment_input(assignment_id: str, db: DbDep, device: DeviceDep) -> Response:
    asg = await _own_assignment(db, assignment_id, device)
    if asg["status"] not in ("assigned", "running"):
        raise ProofNetError(409, "CONFLICT", f"Assignment is {asg['status']}")
    chunk = await db.col("chunks").find_one({"_id": asg["chunk_id"]})
    task = await db.col("tasks").find_one({"_id": asg["task_id"]})
    assert chunk is not None and task is not None
    prepared = await load_task_prepared(db, task)
    data = await asyncio.to_thread(prepared.chunk_npz_bytes, chunk["row_start"], chunk["row_end"])
    import hashlib

    digest = hashlib.sha256(data).hexdigest()
    if digest != chunk["input_sha256"]:  # would mean the prepared data changed: never serve it
        raise ProofNetError(500, "INPUT_DIGEST_MISMATCH", "Chunk input does not match its digest")
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"X-Input-SHA256": digest, "ETag": digest},
    )


def _validate_partial(
    kernel: Kernel, task: dict[str, Any], chunk: dict[str, Any], payload: dict[str, Any]
) -> list[str]:
    n_features = task["prepared"]["n_features"]
    if task["task_type"] == "gaussian_nb_train":
        n_classes = len(task["prepared"]["class_labels"])
        return list(kernel.server.validate_partial(payload, chunk["n_rows"], n_features, n_classes))
    return list(kernel.server.validate_partial(payload, chunk["n_rows"], n_features))


async def _release_attempt(
    db: Db,
    asg: dict[str, Any],
    chunk: dict[str, Any],
    new_status: str,
    error: dict[str, str],
    from_status: list[str],
    *,
    device_stat: str,
) -> bool:
    """Assignment -> rejected/failed/expired; chunk -> pending (or failed); device -> idle."""
    now = utcnow()
    res = await db.col("assignments").update_one(
        {"_id": asg["_id"], "status": {"$in": from_status}},
        {"$set": {"status": new_status, "finished_at": now, "error": error}},
    )
    if not res.modified_count:
        return False
    exhausted = chunk["attempt_count"] >= chunk["max_attempts"]
    if exhausted:
        await db.col("chunks").update_one(
            {"_id": chunk["_id"], "status": "assigned"}, {"$set": {"status": "failed"}}
        )
        t = await db.col("tasks").update_one(
            {"_id": chunk["task_id"], "status": "running"},
            {
                "$set": {
                    "status": "failed",
                    "error": f"chunk {chunk['index']} failed after {chunk['attempt_count']} attempts: "
                    f"{error.get('message', '')}",
                },
                "$push": {"status_history": {"status": "failed", "at": now}},
            },
        )
        if t.modified_count:
            await emit(db, "task_failed", task_id=chunk["task_id"], data={"error": error})
    else:
        await db.col("chunks").update_one(
            {"_id": chunk["_id"], "status": "assigned"},
            {"$set": {"status": "pending"}, "$addToSet": {"excluded_device_ids": asg["device_id"]}},
        )
    inc = {f"stats.{device_stat}": 1}
    await db.col("devices").update_one(
        {"_id": asg["device_id"], "current_assignment_id": asg["_id"]},
        {"$set": {"status": "idle", "current_assignment_id": None}, "$inc": inc},
    )
    await emit(
        db,
        f"assignment_{new_status}",
        task_id=asg["task_id"],
        device_id=asg["device_id"],
        chunk_id=chunk["_id"],
        assignment_id=asg["_id"],
        data=error,
    )
    return True


@router.post("/{assignment_id}/result", response_model=ResultAck)
async def submit_result(
    assignment_id: str,
    body: PartialResultEnvelope,
    request: Request,
    db: DbDep,
    device: DeviceDep,
    settings: SettingsDep,
) -> ResultAck:
    asg = await _own_assignment(db, assignment_id, device)
    if asg["status"] == "succeeded":  # duplicate submission: no state change, no double count
        pr = await db.col("partial_results").find_one({"assignment_id": assignment_id})
        return ResultAck(
            assignment_id=assignment_id,
            status="succeeded",
            acceptance=pr["acceptance"] if pr else None,
        )
    if asg["status"] != "running":
        # Late result (expired/cancelled/...): never merged; kept as an event for Part 2.
        await emit(
            db,
            "late_result",
            task_id=asg["task_id"],
            device_id=device["_id"],
            chunk_id=asg["chunk_id"],
            assignment_id=assignment_id,
            data={"assignment_status": asg["status"], "payload_sha256": body.payload_sha256},
        )
        raise ProofNetError(
            409, "LATE_RESULT", f"Assignment is {asg['status']}; result not accepted"
        )

    chunk = await db.col("chunks").find_one({"_id": asg["chunk_id"]})
    task = await db.col("tasks").find_one({"_id": asg["task_id"]})
    assert chunk is not None and task is not None
    kernel = get_kernel(f"{task['task_type']}@{task['kernel_version']}")

    errors: list[str] = []
    if body.kernel != task["task_type"] or body.kernel_version != task["kernel_version"]:
        errors.append("kernel/version mismatch")
    if body.input_sha256 != chunk["input_sha256"]:
        errors.append("input_sha256 mismatch")
    if body.n_rows != chunk["n_rows"]:
        errors.append("n_rows mismatch")
    try:
        if body.payload_sha256 != payload_sha256(body.payload):
            errors.append("payload_sha256 mismatch")
    except (TypeError, ValueError):
        errors.append("payload is not canonical JSON")
    if not errors:
        errors = _validate_partial(kernel, task, chunk, body.payload)

    pr_id = new_id("pr")
    pr_doc = {
        "_id": pr_id,
        "assignment_id": assignment_id,
        "chunk_id": chunk["_id"],
        "task_id": task["_id"],
        "device_id": device["_id"],
        "kernel": body.kernel,
        "kernel_version": body.kernel_version,
        "payload": body.payload,
        "payload_sha256": body.payload_sha256,
        "structural": {"ok": not errors, "errors": errors},
        "acceptance": "rejected_structural" if errors else "pending",
        "received_at": utcnow(),
    }
    await db.col("partial_results").insert_one(pr_doc)

    if errors:
        msg = "; ".join(errors)
        await _release_attempt(
            db,
            asg,
            chunk,
            "rejected",
            {"code": "INVALID_RESULT", "message": msg},
            ["running"],
            device_stat="invalid_results",
        )
        _kick(request, settings)
        raise ProofNetError(422, "INVALID_RESULT", "Result rejected", errors)

    now = utcnow()
    won = await db.col("assignments").update_one(
        {"_id": assignment_id, "status": "running"},
        {
            "$set": {
                "status": "succeeded",
                "finished_at": now,
                "timings": body.timings.model_dump(),
                "runtime_fingerprint": body.runtime.model_dump(),
                "partial_result_id": pr_id,
            }
        },
    )
    if not won.modified_count:  # lost the race with expiry/cancel
        await db.col("partial_results").delete_one({"_id": pr_id})
        raise ProofNetError(409, "LATE_RESULT", "Assignment is no longer running")

    acceptance = DEFAULT_HOOK.on_partial(task, chunk, body.payload)  # Part 2 insertion point
    await db.col("partial_results").update_one({"_id": pr_id}, {"$set": {"acceptance": acceptance}})
    await db.col("chunks").update_one(
        {"_id": chunk["_id"], "status": "assigned"},
        {"$set": {"status": "completed", "accepted_assignment_id": assignment_id}},
    )
    cells = chunk["work_units"]
    await db.col("devices").update_one(
        {"_id": device["_id"], "current_assignment_id": assignment_id},
        {
            "$set": {"status": "idle", "current_assignment_id": None},
            "$inc": {"stats.succeeded": 1, "stats.cells_processed": cells},
        },
    )
    await emit(
        db,
        "result_accepted",
        task_id=task["_id"],
        device_id=device["_id"],
        chunk_id=chunk["_id"],
        assignment_id=assignment_id,
        data={"acceptance": acceptance, "compute_ms": body.timings.compute_ms},
    )
    remaining = await db.col("chunks").count_documents(
        {"task_id": task["_id"], "status": {"$ne": "completed"}}
    )
    if remaining == 0:
        t = await db.col("tasks").update_one(
            {"_id": task["_id"], "status": "running"},
            {
                "$set": {"status": "aggregating"},
                "$push": {"status_history": {"status": "aggregating", "at": utcnow()}},
            },
        )
        if t.modified_count:
            await emit(db, "task_aggregating", task_id=task["_id"])
            _spawn(request, aggregate_task(db, task["_id"]))
    _kick(request, settings)
    return ResultAck(assignment_id=assignment_id, status="succeeded", acceptance=acceptance)


@router.post("/{assignment_id}/fail", response_model=ResultAck)
async def fail_assignment(
    assignment_id: str,
    body: FailRequest,
    request: Request,
    db: DbDep,
    device: DeviceDep,
    settings: SettingsDep,
) -> ResultAck:
    asg = await _own_assignment(db, assignment_id, device)
    if asg["status"] == "failed":
        return ResultAck(assignment_id=assignment_id, status="failed")  # idempotent
    chunk = await db.col("chunks").find_one({"_id": asg["chunk_id"]})
    assert chunk is not None
    ok = await _release_attempt(
        db,
        asg,
        chunk,
        "failed",
        {"code": body.code, "message": body.message},
        ["assigned", "running"],
        device_stat="failed",
    )
    if not ok:
        raise ProofNetError(409, "CONFLICT", f"Assignment is {asg['status']}; cannot fail")
    _kick(request, settings)
    return ResultAck(assignment_id=assignment_id, status="failed")
