"""Task catalog, validation, preparation and lifecycle entry (P2: up to status=queued)."""

import asyncio
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Request

from proofnet_kernels.server.common import KernelValidationError
from proofnet_kernels.server.registry import Kernel, get_kernel

from ..background import kick_scheduler
from ..contracts import TaskManifest
from ..contracts.api import TaskOut, TaskTypeInfo, ValidationResult
from ..datasets.routes import load_dataset_frame
from ..db import Db, utcnow
from ..deps import DbDep, SettingsDep, UserDep
from ..errors import ProofNetError
from ..events import emit
from ..ids import new_id
from ..scheduling.preview import build_plan_preview
from .prepared import prime_prepared

router = APIRouter(tags=["tasks"])

DESCRIPTIONS = {
    "gaussian_nb_train": "Gaussian Naive Bayes classification (exact sufficient-statistics merge).",
    "linear_ridge_train": "Linear / Ridge regression (exact sufficient-statistics merge).",
}


def _kernel(manifest: TaskManifest) -> Kernel:
    return get_kernel(f"{manifest.task_type.value}@{manifest.kernel_version}")


async def _dataset(db: Db, dataset_id: str, user_id: str) -> dict[str, Any]:
    d = await db.col("datasets").find_one({"_id": dataset_id, "owner_user_id": user_id})
    if d is None:
        raise ProofNetError(404, "NOT_FOUND", "Dataset not found")
    return d


def task_out(t: dict[str, Any]) -> TaskOut:
    return TaskOut(
        id=t["_id"],
        name=t["name"],
        task_type=t["task_type"],
        kernel_version=t["kernel_version"],
        dataset_id=t["dataset_id"],
        params=t["params"],
        execution=t["execution"],
        validation=t["validation"],
        prepared={k: v for k, v in t["prepared"].items() if k != "file_id"},
        plan=t.get("plan"),
        status=t["status"],
        status_history=t["status_history"],
        result=t.get("result"),
        error=t.get("error"),
        verification_policy=t["verification_policy"],
        created_at=t["created_at"],
    )


@router.get("/task-types", response_model=list[TaskTypeInfo])
async def task_types(_: UserDep) -> list[TaskTypeInfo]:
    out = []
    for k in ("gaussian_nb_train@1", "linear_ridge_train@1"):
        kernel = get_kernel(k)
        params_model = kernel.server.PARAMS_MODEL
        out.append(
            TaskTypeInfo(
                task_type=kernel.task_type,
                kernel_version=kernel.version,
                description=DESCRIPTIONS[kernel.task_type],
                params_schema=params_model.model_json_schema(),
            )
        )
    return out


@router.post("/tasks/validate", response_model=ValidationResult)
async def validate_task(
    manifest: TaskManifest, db: DbDep, user: UserDep, settings: SettingsDep
) -> ValidationResult:
    dataset = await _dataset(db, manifest.dataset_id, user["_id"])
    report = _kernel(manifest).server.validate(dataset["profile"], manifest.params)
    preview = None
    if report.ok:
        preview = await build_plan_preview(
            db,
            settings,
            dataset["profile"],
            manifest.params.model_dump(mode="json"),
            manifest.execution.model_dump(mode="json"),
        )
    return ValidationResult(**asdict(report), plan_preview=preview)


@router.post("/tasks", response_model=TaskOut, status_code=201)
async def create_task(
    manifest: TaskManifest, request: Request, db: DbDep, user: UserDep, settings: SettingsDep
) -> TaskOut:
    kernel = _kernel(manifest)
    dataset = await _dataset(db, manifest.dataset_id, user["_id"])
    report = kernel.server.validate(dataset["profile"], manifest.params)
    if not report.ok:
        raise ProofNetError(422, "VALIDATION_FAILED", "Task validation failed", report.errors)
    df = await load_dataset_frame(db, dataset)
    try:
        prepared = await asyncio.to_thread(kernel.server.prepare, df, manifest.params)
    except KernelValidationError as e:
        raise ProofNetError(
            422, "VALIDATION_FAILED", "Task validation failed", e.report.errors
        ) from None
    npz = await asyncio.to_thread(prepared.to_npz_bytes)
    sha = await asyncio.to_thread(prepared.sha256)
    task_id = new_id("tsk")
    file_id = await db.fs.upload_from_stream(
        f"{task_id}_prepared.npz",
        npz,
        metadata={"kind": "prepared", "task_id": task_id, "sha256": sha},
    )
    now = utcnow()
    doc = {
        "_id": task_id,
        "owner_user_id": user["_id"],
        "name": manifest.name,
        "task_type": manifest.task_type.value,
        "kernel_version": manifest.kernel_version,
        "dataset_id": manifest.dataset_id,
        "params": manifest.params.model_dump(mode="json"),
        "execution": manifest.execution.model_dump(mode="json"),
        "validation": {**asdict(report), "n_rows_dropped": prepared.n_dropped},
        "prepared": {
            "file_id": file_id,
            "sha256": sha,
            "n_train": prepared.n_train,
            "n_test": int(len(prepared.y_test)),
            "n_features": prepared.n_features,
            "class_labels": prepared.class_labels,
            "feature_names": prepared.feature_names,
        },
        "plan": None,
        "status": "queued",
        "status_history": [{"status": "queued", "at": now}],
        "result": None,
        "error": None,
        "verification_policy": {"mode": "none"},
        "created_at": now,
    }
    await db.col("tasks").insert_one(doc)
    prime_prepared(task_id, prepared)
    await emit(db, "task_created", task_id=task_id, data={"task_type": doc["task_type"]})
    kick_scheduler(request.app, settings)
    return task_out(doc)


@router.get("/tasks", response_model=list[TaskOut])
async def list_tasks(db: DbDep, user: UserDep) -> list[TaskOut]:
    cursor = db.col("tasks").find({"owner_user_id": user["_id"]}).sort("created_at", -1)
    return [task_out(t) async for t in cursor]


@router.get("/tasks/{task_id}", response_model=TaskOut)
async def get_task(task_id: str, db: DbDep, user: UserDep) -> TaskOut:
    t = await db.col("tasks").find_one({"_id": task_id, "owner_user_id": user["_id"]})
    if t is None:
        raise ProofNetError(404, "NOT_FOUND", "Task not found")
    return task_out(t)
