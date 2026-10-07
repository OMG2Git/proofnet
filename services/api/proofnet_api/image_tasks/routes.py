"""Image pipeline (separate from the CSV workloads): image dataset upload, CNN task validation and
creation, training curve. The CSV endpoints are untouched."""

import asyncio
import hashlib
import io
from dataclasses import asdict
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, File, Query, Request, UploadFile

from proofnet_kernels.server import cnn as cnn_srv
from proofnet_kernels.server import images
from proofnet_kernels.server.common import KernelValidationError

from ..background import kick_scheduler
from ..contracts import ImageTaskManifest
from ..contracts.api import (
    ImageDatasetOut,
    TaskOut,
    TrainingRound,
    TrainingStatus,
    ValidationResult,
)
from ..db import Db, utcnow
from ..deps import DbDep, SettingsDep, UserDep
from ..errors import ProofNetError
from ..events import emit
from ..ids import new_id
from ..scheduling.planner import device_score
from ..scheduling.scheduler import _eligible_devices
from ..tasks.prepared import prime_images
from ..tasks.routes import task_out
from ..training.service import split_batch

router = APIRouter(tags=["images"])

_CHUNK = 1024 * 1024


def _dataset_out(d: dict[str, Any]) -> ImageDatasetOut:
    return ImageDatasetOut(
        id=d["_id"],
        filename=d["filename"],
        size_bytes=d["size_bytes"],
        profile=d["profile"],
        created_at=d["created_at"],
    )


async def _owned_dataset(db: Db, dataset_id: str, user_id: str) -> dict[str, Any]:
    d = await db.col("image_datasets").find_one({"_id": dataset_id, "owner_user_id": user_id})
    if d is None:
        raise ProofNetError(404, "NOT_FOUND", "Image dataset not found")
    return d


@router.post("/image-datasets", response_model=ImageDatasetOut, status_code=201)
async def upload_image_dataset(
    db: DbDep,
    settings: SettingsDep,
    user: UserDep,
    file: Annotated[UploadFile, File()],
    side: Annotated[int, Query(description="images are resized to side x side")] = 28,
) -> ImageDatasetOut:
    """A .zip of class folders (png/jpg), or a Kaggle MNIST-style pixel .csv."""
    limit = settings.max_upload_mb * 1024 * 1024
    chunks: list[bytes] = []
    size = 0
    while block := await file.read(_CHUNK):
        size += len(block)
        if size > limit:
            raise ProofNetError(
                413, "PAYLOAD_TOO_LARGE", f"upload exceeds the {settings.max_upload_mb} MB limit"
            )
        chunks.append(block)
    raw = b"".join(chunks)
    name = (file.filename or "images.zip").lower()
    try:
        if name.endswith(".csv"):
            ds = await asyncio.to_thread(images.from_pixel_csv, raw, side)
        else:
            ds = await asyncio.to_thread(images.from_zip, raw, side)
    except KernelValidationError as e:
        raise ProofNetError(422, "INVALID_IMAGES", "; ".join(e.report.errors)) from None
    profile = await asyncio.to_thread(images.profile_images, ds)
    npz = await asyncio.to_thread(images.to_npz, ds)
    ds_id = new_id("img")
    file_id = await db.fs.upload_from_stream(
        f"{ds_id}_images.npz", npz, metadata={"kind": "image_dataset", "dataset_id": ds_id}
    )
    doc = {
        "_id": ds_id,
        "owner_user_id": user["_id"],
        "filename": file.filename or "images.zip",
        "size_bytes": size,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "npz_file_id": file_id,
        "profile": profile,
        "created_at": utcnow(),
    }
    await db.col("image_datasets").insert_one(doc)
    await emit(
        db, "image_dataset_uploaded", data={"dataset_id": ds_id, "images": profile["n_images"]}
    )
    return _dataset_out(doc)


@router.get("/image-datasets", response_model=list[ImageDatasetOut])
async def list_image_datasets(db: DbDep, user: UserDep) -> list[ImageDatasetOut]:
    cur = db.col("image_datasets").find({"owner_user_id": user["_id"]}).sort("created_at", -1)
    return [_dataset_out(d) async for d in cur]


@router.get("/image-datasets/{dataset_id}", response_model=ImageDatasetOut)
async def get_image_dataset(dataset_id: str, db: DbDep, user: UserDep) -> ImageDatasetOut:
    return _dataset_out(await _owned_dataset(db, dataset_id, user["_id"]))


async def _preview(
    db: Db, settings: Any, profile: dict[str, Any], manifest: ImageTaskManifest
) -> dict[str, Any]:
    n_train = profile["n_images"] - max(
        1, round(profile["n_images"] * manifest.params.test_fraction)
    )
    d = int(np.prod(profile["shape"]))
    pseudo = {"prepared": {"n_features": d}, "execution": manifest.execution.model_dump()}
    devices = await _eligible_devices(db, settings, pseudo, rows=cnn_srv.MIN_ROWS_PER_DEVICE)
    ready = len(devices) >= manifest.execution.min_devices
    split = (
        split_batch(manifest.params.global_batch_size, devices, manifest.execution.max_devices)
        if devices
        else []
    )
    return {
        "estimated": True,
        "kind": "iterative",
        "n_train": n_train,
        "steps": manifest.params.steps,
        "global_batch_size": manifest.params.global_batch_size,
        "eligible_devices": len(devices),
        "min_devices": manifest.execution.min_devices,
        "ready_to_start": ready,
        "message": None
        if ready
        else f"needs {manifest.execution.min_devices} eligible device(s); {len(devices)} available now",
        "explanation": "every round the global mini-batch is split across devices in proportion to "
        "their measured benchmark; devices only ever receive their slice of one mini-batch plus the "
        "current weights, never the dataset",
        "shares": [
            {
                "device_name": dev["name"],
                "score_cells_per_sec": device_score(dev),
                "weight": rows / manifest.params.global_batch_size,
                "rows": rows,
                "n_chunks": 1,
                "est_seconds": None,
            }
            for dev, rows in split
        ],
        "not_eligible": [],
    }


@router.post("/image-tasks/validate", response_model=ValidationResult)
async def validate_image_task(
    manifest: ImageTaskManifest, db: DbDep, user: UserDep, settings: SettingsDep
) -> ValidationResult:
    ds = await _owned_dataset(db, manifest.dataset_id, user["_id"])
    report = cnn_srv.validate(ds["profile"], manifest.params)
    preview = await _preview(db, settings, ds["profile"], manifest) if report.ok else None
    return ValidationResult(**asdict(report), plan_preview=preview)


@router.post("/image-tasks", response_model=TaskOut, status_code=201)
async def create_image_task(
    manifest: ImageTaskManifest,
    request: Request,
    db: DbDep,
    user: UserDep,
    settings: SettingsDep,
) -> TaskOut:
    ds_doc = await _owned_dataset(db, manifest.dataset_id, user["_id"])
    report = cnn_srv.validate(ds_doc["profile"], manifest.params)
    if not report.ok:
        raise ProofNetError(422, "VALIDATION_FAILED", "Task validation failed", report.errors)
    buf = io.BytesIO()
    await db.fs.download_to_stream(ds_doc["npz_file_id"], buf)

    def build() -> tuple[cnn_srv.ImagePrepared, bytes, str]:
        with np.load(io.BytesIO(buf.getvalue()), allow_pickle=False) as z:
            ds = images.ImageSet(z["X"], z["y"], list(ds_doc["profile"]["classes"]))
        prepared = cnn_srv.prepare(ds, manifest.params)
        data = prepared.to_npz_bytes()
        return prepared, data, hashlib.sha256(data).hexdigest()

    try:
        prepared, npz, sha = await asyncio.to_thread(build)
    except KernelValidationError as e:
        raise ProofNetError(
            422, "VALIDATION_FAILED", "Task validation failed", e.report.errors
        ) from None
    task_id = new_id("tsk")
    file_id = await db.fs.upload_from_stream(
        f"{task_id}_prepared_images.npz", npz, metadata={"kind": "prepared", "task_id": task_id}
    )
    now = utcnow()
    shape = prepared.shape
    doc = {
        "_id": task_id,
        "owner_user_id": user["_id"],
        "name": manifest.name,
        "task_type": manifest.task_type,
        "kernel_version": manifest.kernel_version,
        "dataset_id": manifest.dataset_id,
        "params": manifest.params.model_dump(mode="json"),
        "execution": manifest.execution.model_dump(mode="json"),
        "validation": asdict(report),
        "prepared": {
            "file_id": file_id,
            "sha256": sha,
            "n_train": prepared.n_train,
            "n_test": int(len(prepared.y_test)),
            "n_features": int(np.prod(shape)),
            "image_shape": shape,
            "class_labels": prepared.class_names,
            "feature_names": [],
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
    prime_images(task_id, prepared)
    await emit(db, "task_created", task_id=task_id, data={"task_type": manifest.task_type})
    kick_scheduler(request.app, settings)
    return task_out(doc)


@router.get("/tasks/{task_id}/training", response_model=TrainingStatus)
async def training_status(task_id: str, db: DbDep, user: UserDep) -> TrainingStatus:
    t = await db.col("tasks").find_one({"_id": task_id, "owner_user_id": user["_id"]})
    if t is None or not t.get("training"):
        raise ProofNetError(404, "NOT_FOUND", "No training run for this task")
    tr = t["training"]
    rounds = [
        r
        async for r in db.col("training_rounds")
        .find({"task_id": task_id})
        .sort("round", 1)
        .limit(5000)
    ]
    return TrainingStatus(
        steps=tr["steps"],
        round=tr["round"],
        state=tr["state"],
        global_batch_size=tr["global_batch_size"],
        learning_rate=tr["learning_rate"],
        n_params=tr["n_params"],
        rounds=[
            TrainingRound(
                round=r["round"],
                loss=r["loss"],
                accuracy=r["accuracy"],
                n=r["n"],
                devices=r["devices"],
                verification=r.get("verification"),
            )
            for r in rounds
        ],
        verified_rounds=[r["round"] for r in rounds if r.get("verification")],
    )
