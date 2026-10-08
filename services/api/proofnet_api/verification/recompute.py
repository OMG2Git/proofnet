"""Backend recomputation of one chunk (the audit oracle) and the discrepancy measure.

The MVP workloads are cheap enough to recompute exactly on the backend, so audits use backend
recomputation as ground truth: it is immune to collusion between devices (ARCHITECTURE 17).
"""

import asyncio
import io
from dataclasses import dataclass
from typing import Any

import numpy as np

from proofnet_kernels.core import cnn as core_cnn
from proofnet_kernels.server import cnn as srv_cnn
from proofnet_kernels.server.registry import get_kernel

from ..db import Db
from ..tasks.prepared import load_task_prepared
from ..training import service as training


@dataclass(frozen=True)
class KernelLimits:
    hard: float  # beyond this a result is wrong (numerical noise never reaches it)
    floor: float  # smallest tolerance the trust system will use


def kernel_limits(task: dict[str, Any]) -> KernelLimits:
    k = get_kernel(f"{task['task_type']}@{task['kernel_version']}")
    hard = srv_cnn.GRADIENT_TOLERANCE if training.is_iterative(task) else k.server.TOLERANCE
    return KernelLimits(hard=float(hard), floor=float(k.server.AUDIT_FLOOR))


async def expected_partial(db: Db, task: dict[str, Any], chunk: dict[str, Any]) -> dict[str, Any]:
    """What an honest device must return for this chunk, recomputed from the stored data."""
    if training.is_iterative(task):
        data = await training.build_chunk_input(db, task, chunk)
        arch = task["training"]["arch"]

        def run_cnn() -> dict[str, Any]:
            with np.load(io.BytesIO(data), allow_pickle=False) as z:
                return core_cnn.map(z["X"], z["y"], z["w"], arch)

        return await asyncio.to_thread(run_cnn)
    prepared = await load_task_prepared(db, task)
    k = get_kernel(f"{task['task_type']}@{task['kernel_version']}")
    a, b = chunk["row_start"], chunk["row_end"]

    def run_csv() -> dict[str, Any]:
        x, y = prepared.x_train[a:b], prepared.y_train[a:b]
        if task["task_type"] == "gaussian_nb_train":
            return dict(k.core.map(x, y, len(task["prepared"]["class_labels"])))
        return dict(k.core.map(x, y))

    return await asyncio.to_thread(run_csv)


def discrepancy(task: dict[str, Any], payload: dict[str, Any], expected: dict[str, Any]) -> float:
    """Normwise relative discrepancy between a device's partial result and the recomputation."""
    if training.is_iterative(task):
        got = core_cnn.decode_vector(payload["grad_b64"], "<f4").astype(np.float64)
        ref = core_cnn.decode_vector(expected["grad_b64"], "<f4").astype(np.float64)
        grad = srv_cnn.compare_gradients(got, ref)["max_rel_diff"]
        loss = abs(payload["loss_sum"] - expected["loss_sum"]) / max(abs(expected["loss_sum"]), 1.0)
        return float(max(grad, loss))
    k = get_kernel(f"{task['task_type']}@{task['kernel_version']}")
    return float(k.server.compare(payload, expected)["max_rel_diff"])
