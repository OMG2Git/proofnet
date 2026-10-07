"""Load a task's prepared data (pickle-free .npz in GridFS), with a small bounded cache.

Chunk inputs are sliced from this array on demand (ARCHITECTURE 3.5); without a cache every input
request would re-read the file from Atlas. The cache is byte-bounded so it is safe on a 512 MB host.
"""

import asyncio
import io
from collections import OrderedDict
from typing import Any

from proofnet_kernels.server.common import PreparedData, load_prepared

from ..db import Db

CACHE_MAX_BYTES = 96 * 1024 * 1024
_cache: OrderedDict[str, tuple[PreparedData, int]] = OrderedDict()


def _nbytes(p: PreparedData) -> int:
    return int(p.x_train.nbytes + p.y_train.nbytes + p.x_test.nbytes + p.y_test.nbytes)


def prime_prepared(task_id: str, prepared: PreparedData) -> None:
    """Remember freshly prepared data (it is about to be needed for planning and inputs)."""
    size = _nbytes(prepared)
    if size > CACHE_MAX_BYTES:
        return
    _cache.pop(task_id, None)
    _cache[task_id] = (prepared, size)
    while sum(s for _, s in _cache.values()) > CACHE_MAX_BYTES:
        _cache.popitem(last=False)


async def load_task_prepared(db: Db, task: dict[str, Any]) -> PreparedData:
    hit = _cache.get(task["_id"])
    if hit is not None:
        _cache.move_to_end(task["_id"])
        return hit[0]
    prep = task["prepared"]
    buf = io.BytesIO()
    await db.fs.download_to_stream(prep["file_id"], buf)
    loaded = await asyncio.to_thread(
        load_prepared, buf.getvalue(), prep["feature_names"], prep["class_labels"]
    )
    prime_prepared(task["_id"], loaded)
    return loaded
