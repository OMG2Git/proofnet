"""Aggregation (ARCHITECTURE 9.3): coverage check -> merge -> finalize -> metrics ->
reference check -> artifacts. Idempotent: guarded by the running->aggregating transition and a
claim; the reconciler re-runs aggregation that stalls (backend restart)."""

import asyncio
import csv
import hashlib
import io
import platform
import sys
from datetime import timedelta
from typing import Any

import joblib
import numpy as np
import sklearn

from proofnet_kernels.server.common import PreparedData
from proofnet_kernels.server.params import GaussianNBParams, LinearRidgeParams
from proofnet_kernels.server.registry import get_kernel

from ..db import Db, utcnow
from ..events import emit
from ..ids import new_id
from ..tasks.prepared import load_task_images, load_task_prepared
from ..verification import ACCEPTED_UNVERIFIED

CLAIM_STALE_SECONDS = 120

LIMITATIONS = [
    "Results are unverified: a contributor could return well-formed but wrong statistics "
    "(verification is Part 2).",
    "Contributors receive the raw rows of their partition; do not upload sensitive data.",
    "Holdout metrics are computed by the aggregator, not by contributor devices.",
    "Research prototype; not production-grade secure remote execution.",
]


class AggregationError(RuntimeError):
    pass


def check_coverage(chunks: list[dict[str, Any]], n_train: int) -> None:
    """Row ranges must tile [0, n_train) exactly, in chunk-index order."""
    pos = 0
    for c in sorted(chunks, key=lambda c: c["index"]):
        if c["row_start"] != pos or c["row_end"] <= c["row_start"]:
            raise AggregationError(f"chunk ranges do not tile [0, {n_train}) at {pos}")
        pos = c["row_end"]
    if pos != n_train:
        raise AggregationError(f"chunk ranges cover [0, {pos}) but n_train is {n_train}")


def _compute(
    task: dict[str, Any], prepared: PreparedData, payloads: list[dict[str, Any]]
) -> dict[str, Any]:
    """CPU-bound part, run in a thread: merge, finalize, reference, artifact bytes."""
    kernel = get_kernel(f"{task['task_type']}@{task['kernel_version']}")
    merged = kernel.core.merge(payloads)
    if task["task_type"] == "gaussian_nb_train":
        params: Any = GaussianNBParams.model_validate(task["params"])
        final = kernel.server.finalize(merged, prepared, params)
        reference = kernel.server.reference(prepared, len(prepared.class_labels))
    else:
        params = LinearRidgeParams.model_validate(task["params"])
        final = kernel.server.finalize(merged, prepared, params)
        reference = kernel.server.reference(prepared)
    check = kernel.server.compare(merged, reference)

    model_buf = io.BytesIO()
    joblib.dump(final.model, model_buf)
    pred_buf = io.StringIO()
    w = csv.writer(pred_buf, lineterminator="\n")
    w.writerow(["row", "y_true", "y_pred"])
    labels = prepared.class_labels
    for i, (yt, yp) in enumerate(zip(prepared.y_test, final.y_pred, strict=True)):
        if labels:
            w.writerow([i, labels[int(yt)], labels[int(yp)]])
        else:
            w.writerow([i, repr(float(yt)), repr(float(yp))])
    return {
        "final": final,
        "check": {**check, "n_train": prepared.n_train},
        "joblib": model_buf.getvalue(),
        "predictions": pred_buf.getvalue().encode("utf-8"),
    }


async def aggregate_task(db: Db, task_id: str) -> bool:
    """Claim and run aggregation for a task in `aggregating`. Returns True if completed."""
    claim = await db.col("tasks").update_one(
        {"_id": task_id, "status": "aggregating", "aggregation_claimed_at": None},
        {"$set": {"aggregation_claimed_at": utcnow()}},
    )
    if not claim.modified_count:
        return False
    task = await db.col("tasks").find_one({"_id": task_id})
    assert task is not None
    try:
        return await _aggregate(db, task)
    except Exception as e:  # any failure is explicit and visible
        now = utcnow()
        res = await db.col("tasks").update_one(
            {"_id": task_id, "status": "aggregating"},
            {
                "$set": {"status": "failed", "error": f"aggregation failed: {e}"},
                "$push": {"status_history": {"status": "failed", "at": now}},
            },
        )
        if res.modified_count:
            await emit(db, "task_failed", task_id=task_id, data={"error": str(e)})
        return False


async def _store_artifacts(db: Db, task_id: str, files: list[tuple[str, str, bytes]]) -> list[str]:
    ids: list[str] = []
    for kind, filename, data in files:
        file_id = await db.fs.upload_from_stream(
            f"{task_id}_{filename}", data, metadata={"kind": "artifact", "task_id": task_id}
        )
        art_id = new_id("art")
        await db.col("artifacts").insert_one(
            {
                "_id": art_id,
                "task_id": task_id,
                "kind": kind,
                "filename": filename,
                "file_id": file_id,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "created_at": utcnow(),
            }
        )
        ids.append(art_id)
    return ids


async def _aggregate_cnn(db: Db, task: dict[str, Any]) -> bool:
    """Final step of an iterative CNN task: evaluate the final model on the aggregator-held holdout,
    summarise the training run and the centralized gradient checks, store artifacts."""
    import csv
    import json

    from proofnet_kernels.server import cnn as cnn_srv

    from ..training.service import load_state

    task_id = task["_id"]
    tr = task["training"]
    arch = tr["arch"]
    prepared = await load_task_images(db, task)
    w, _ = await load_state(db, task_id, tr["steps"])
    metrics, pred = await asyncio.to_thread(cnn_srv.evaluate, w, prepared, arch)
    arts = cnn_srv.model_artifacts(w, arch, prepared.class_names)

    rounds = [
        r async for r in db.col("training_rounds").find({"task_id": task_id}).sort("round", 1)
    ]
    names = prepared.class_names
    pbuf = io.StringIO()
    pw = csv.writer(pbuf, lineterminator="\n")
    pw.writerow(["row", "y_true", "y_pred"])
    for i, (yt, yp) in enumerate(zip(prepared.y_test, pred, strict=True)):
        pw.writerow([i, names[int(yt)], names[int(yp)]])
    cbuf = io.StringIO()
    cw = csv.writer(cbuf, lineterminator="\n")
    cw.writerow(["round", "loss", "batch_accuracy", "devices", "max_compute_ms"])
    for r in rounds:
        comp = [d.get("compute_ms") or 0 for d in r["devices"]]
        cw.writerow(
            [
                r["round"],
                r["loss"],
                r["accuracy"],
                len(r["devices"]),
                round(max(comp), 1) if comp else 0,
            ]
        )

    contrib: dict[str, dict[str, Any]] = {}
    for r in rounds:
        for d in r["devices"]:
            c = contrib.setdefault(
                d["device_id"],
                {"device_id": d["device_id"], "device_name": d["name"], "rounds": 0, "rows": 0,
                 "compute_ms_total": 0.0, "download_ms_total": 0.0},
            )  # fmt: skip
            c["rounds"] += 1
            c["rows"] += d["rows"]
            c["compute_ms_total"] += d.get("compute_ms") or 0
            c["download_ms_total"] += d.get("download_ms") or 0
    verifs = [r["verification"] for r in rounds if r.get("verification")]
    ref = {
        "description": "per-round gradient sums from the devices vs centralized recomputation of "
        "the same global batch (verified on sampled rounds)",
        "verified_rounds": [v["round"] for v in verifs],
        "max_relative_difference": max((v["max_rel_diff"] for v in verifs), default=None),
        "tolerance": cnn_srv.GRADIENT_TOLERANCE,
        "passed": all(v["within_tolerance"] for v in verifs) if verifs else None,
    }
    n_train = task["prepared"]["n_train"]
    training_summary = {
        "steps": tr["steps"],
        "global_batch_size": tr["global_batch_size"],
        "epochs": round(tr["steps"] * tr["global_batch_size"] / n_train, 2),
        "learning_rate": tr["learning_rate"],
        "momentum": tr["momentum"],
        "n_params": tr["n_params"],
        "first_loss": rounds[0]["loss"] if rounds else None,
        "final_loss": rounds[-1]["loss"] if rounds else None,
    }
    report = {
        "task": {
            "id": task_id,
            "name": task["name"],
            "task_type": task["task_type"],
            "kernel_version": task["kernel_version"],
            "params": task["params"],
            "execution": task["execution"],
        },
        "architecture": arch,
        "training": training_summary,
        "data": {
            "n_train": n_train,
            "n_test": task["prepared"]["n_test"],
            "image_shape": task["prepared"]["image_shape"],
            "class_names": names,
            "prepared_sha256": task["prepared"]["sha256"],
        },
        "plan": task.get("plan"),
        "contributions": sorted(contrib.values(), key=lambda c: -c["rows"]),
        "metrics": {
            **{k: v for k, v in metrics.items() if k != "class_names"},
            "computed_by": "aggregator (holdout evaluation of the final model, not distributed)",
        },
        "reference_check": ref,
        "gradient_verifications": verifs,
        "versions": {
            "numpy": np.__version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "acceptance": ACCEPTED_UNVERIFIED,
        "limitations": LIMITATIONS
        + [
            "Contributors saw the training images of the mini-batches they computed on.",
            "Only sampled rounds are re-checked against a centralized gradient; other rounds "
            "are accepted unverified.",
        ],
        "generated_at": utcnow().isoformat(),
    }
    files: list[tuple[str, str, bytes]] = [
        ("model_npz", "model.npz", arts["model.npz"]),
        ("model_json", "model.json", arts["model.json"]),
        ("inference_py", "inference.py", arts["inference.py"]),
        ("predictions_csv", "predictions.csv", pbuf.getvalue().encode()),
        ("training_curve_csv", "training_curve.csv", cbuf.getvalue().encode()),
    ]
    report["artifact_digests"] = {n: hashlib.sha256(b).hexdigest() for _, n, b in files}
    files.append(("report_json", "report.json", json.dumps(report, indent=2, default=str).encode()))
    artifact_ids = await _store_artifacts(db, task_id, files)
    res = await db.col("tasks").update_one(
        {"_id": task_id, "status": "aggregating"},
        {
            "$set": {
                "status": "completed",
                "result": {
                    "artifact_ids": artifact_ids,
                    "metrics": {k: v for k, v in metrics.items() if k != "class_names"},
                    "reference_check": ref,
                    "training": training_summary,
                },
            },
            "$push": {"status_history": {"status": "completed", "at": utcnow()}},
        },
    )
    if res.modified_count:
        await emit(
            db,
            "task_completed",
            task_id=task_id,
            data={"reference_passed": ref["passed"], "accuracy": metrics["accuracy"]},
        )
    return bool(res.modified_count)


async def _aggregate(db: Db, task: dict[str, Any]) -> bool:
    task_id = task["_id"]
    if task.get("task_type") == "cnn_image_train":
        return await _aggregate_cnn(db, task)
    chunks = [c async for c in db.col("chunks").find({"task_id": task_id}).sort("index", 1)]
    if not chunks or any(c["status"] != "completed" for c in chunks):
        raise AggregationError("not all chunks are completed")
    check_coverage(chunks, task["prepared"]["n_train"])

    partials: list[dict[str, Any]] = []
    for c in chunks:
        pr = await db.col("partial_results").find_one(
            {"assignment_id": c["accepted_assignment_id"], "acceptance": ACCEPTED_UNVERIFIED}
        )
        if pr is None:
            raise AggregationError(f"no accepted partial result for chunk {c['index']}")
        partials.append(pr)

    prepared = await load_task_prepared(db, task)
    out = await asyncio.to_thread(_compute, task, prepared, [p["payload"] for p in partials])
    final, check = out["final"], out["check"]

    asgs = {
        a["_id"]: a
        async for a in db.col("assignments").find(
            {"_id": {"$in": [c["accepted_assignment_id"] for c in chunks]}}
        )
    }
    devs = {
        d["_id"]: d
        async for d in db.col("devices").find(
            {"_id": {"$in": [a["device_id"] for a in asgs.values()]}}
        )
    }
    contributions = []
    for c in chunks:
        a = asgs[c["accepted_assignment_id"]]
        contributions.append(
            {
                "chunk_index": c["index"],
                "row_start": c["row_start"],
                "row_end": c["row_end"],
                "rows": c["n_rows"],
                "device_id": a["device_id"],
                "device_name": devs.get(a["device_id"], {}).get("name"),
                "device_benchmark_cells_per_sec": (
                    devs.get(a["device_id"], {}).get("benchmark") or {}
                ).get("score_cells_per_sec"),
                "attempt_no": a["attempt_no"],
                "timings_ms": a.get("timings"),
                "runtime_fingerprint": a.get("runtime_fingerprint"),
                "input_sha256": c["input_sha256"],
            }
        )
    plan = task.get("plan") or {}
    report = {
        "task": {
            "id": task_id,
            "name": task["name"],
            "task_type": task["task_type"],
            "kernel_version": task["kernel_version"],
            "params": task["params"],
            "execution": task["execution"],
        },
        "validation": task["validation"],
        "data": {
            "n_train": task["prepared"]["n_train"],
            "n_test": task["prepared"]["n_test"],
            "n_features": task["prepared"]["n_features"],
            "feature_names": task["prepared"]["feature_names"],
            "class_labels": task["prepared"]["class_labels"],
            "prepared_sha256": task["prepared"]["sha256"],
        },
        "plan": {**plan, "created_at": plan.get("created_at")},
        "contributions": contributions,
        "metrics": {
            **final.metrics,
            "computed_by": "aggregator (holdout evaluation, not distributed)",
        },
        "reference_check": {
            "description": "distributed merged statistics vs centralized recomputation",
            "max_relative_difference": check["max_rel_diff"],
            "tolerance": check["tolerance"],
            "passed": bool(check["within_tolerance"]),
            "per_field": check["fields"],
        },
        "versions": {
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "acceptance": ACCEPTED_UNVERIFIED,
        "limitations": LIMITATIONS,
        "generated_at": utcnow().isoformat(),
    }
    import json

    files: list[tuple[str, str, bytes]] = [
        ("model_joblib", "model.joblib", out["joblib"]),
        ("model_json", "model.json", json.dumps(final.model_json, indent=2).encode("utf-8")),
        ("predictions_csv", "predictions.csv", out["predictions"]),
    ]
    report["artifact_digests"] = {name: hashlib.sha256(b).hexdigest() for _, name, b in files}
    files.append(
        ("report_json", "report.json", json.dumps(report, indent=2, default=str).encode("utf-8"))
    )

    artifact_ids: list[str] = []
    for kind, filename, data in files:
        file_id = await db.fs.upload_from_stream(
            f"{task_id}_{filename}", data, metadata={"kind": "artifact", "task_id": task_id}
        )
        art_id = new_id("art")
        await db.col("artifacts").insert_one(
            {
                "_id": art_id,
                "task_id": task_id,
                "kind": kind,
                "filename": filename,
                "file_id": file_id,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "created_at": utcnow(),
            }
        )
        artifact_ids.append(art_id)

    now = utcnow()
    res = await db.col("tasks").update_one(
        {"_id": task_id, "status": "aggregating"},
        {
            "$set": {
                "status": "completed",
                "result": {
                    "artifact_ids": artifact_ids,
                    "metrics": final.metrics,
                    "reference_check": report["reference_check"],
                },
            },
            "$push": {"status_history": {"status": "completed", "at": now}},
        },
    )
    if res.modified_count:
        await emit(
            db,
            "task_completed",
            task_id=task_id,
            data={"reference_passed": report["reference_check"]["passed"]},
        )
    return bool(res.modified_count)


async def release_stale_claims(db: Db, now: Any = None) -> int:
    """Reconciler helper: aggregation claimed long ago (backend restarted) is retried."""
    now = now or utcnow()
    res = await db.col("tasks").update_many(
        {
            "status": "aggregating",
            "aggregation_claimed_at": {"$lt": now - timedelta(seconds=CLAIM_STALE_SECONDS)},
        },
        {"$set": {"aggregation_claimed_at": None}},
    )
    return int(res.modified_count)
