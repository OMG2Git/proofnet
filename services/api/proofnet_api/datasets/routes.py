"""Dataset upload (raw CSV -> GridFS), profiling, retrieval."""

import asyncio
import hashlib
import io
from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, File, UploadFile

from proofnet_kernels.server.common import MAX_ROWS, profile_dataframe

from ..contracts.api import DatasetOut
from ..db import Db, utcnow
from ..deps import DbDep, SettingsDep, UserDep
from ..errors import ProofNetError
from ..events import emit
from ..ids import new_id

router = APIRouter(prefix="/datasets", tags=["datasets"])

_CHUNK = 1024 * 1024


def parse_csv(raw: bytes) -> pd.DataFrame:
    """Parse a UTF-8 CSV with header; reads at most MAX_ROWS + 1 rows (limit enforced later)."""
    try:
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8", nrows=MAX_ROWS + 1)
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as e:
        raise ProofNetError(422, "INVALID_CSV", f"Could not parse CSV: {e}") from None
    if df.shape[1] == 0:
        raise ProofNetError(422, "INVALID_CSV", "CSV has no columns")
    return df


async def load_dataset_frame(db: Db, dataset: dict[str, Any]) -> pd.DataFrame:
    buf = io.BytesIO()
    await db.fs.download_to_stream(dataset["raw_file_id"], buf)
    return await asyncio.to_thread(parse_csv, buf.getvalue())


def dataset_out(d: dict[str, Any]) -> DatasetOut:
    return DatasetOut(
        id=d["_id"],
        filename=d["filename"],
        size_bytes=d["size_bytes"],
        sha256=d["sha256"],
        profile=d["profile"],
        created_at=d["created_at"],
    )


@router.post("", response_model=DatasetOut, status_code=201)
async def upload_dataset(
    db: DbDep, settings: SettingsDep, user: UserDep, file: Annotated[UploadFile, File()]
) -> DatasetOut:
    limit = settings.max_upload_mb * 1024 * 1024
    chunks: list[bytes] = []
    size = 0
    while block := await file.read(_CHUNK):
        size += len(block)
        if size > limit:
            raise ProofNetError(
                413, "PAYLOAD_TOO_LARGE", f"CSV exceeds the {settings.max_upload_mb} MB limit"
            )
        chunks.append(block)
    raw = b"".join(chunks)
    if not raw:
        raise ProofNetError(422, "INVALID_CSV", "Uploaded file is empty")
    df = await asyncio.to_thread(parse_csv, raw)
    profile = await asyncio.to_thread(profile_dataframe, df)
    if profile["n_rows"] > MAX_ROWS:
        raise ProofNetError(422, "TOO_MANY_ROWS", f"CSV has more than {MAX_ROWS} rows")
    dataset_id = new_id("ds")
    sha = hashlib.sha256(raw).hexdigest()
    file_id = await db.fs.upload_from_stream(
        file.filename or "dataset.csv",
        raw,
        metadata={"kind": "dataset_raw", "dataset_id": dataset_id, "owner_user_id": user["_id"]},
    )
    doc = {
        "_id": dataset_id,
        "owner_user_id": user["_id"],
        "filename": file.filename or "dataset.csv",
        "size_bytes": size,
        "sha256": sha,
        "raw_file_id": file_id,
        "profile": profile,
        "created_at": utcnow(),
    }
    await db.col("datasets").insert_one(doc)
    await emit(db, "dataset_uploaded", data={"dataset_id": dataset_id, "rows": profile["n_rows"]})
    return dataset_out(doc)


@router.get("/{dataset_id}", response_model=DatasetOut)
async def get_dataset(dataset_id: str, db: DbDep, user: UserDep) -> DatasetOut:
    d = await db.col("datasets").find_one({"_id": dataset_id, "owner_user_id": user["_id"]})
    if d is None:
        raise ProofNetError(404, "NOT_FOUND", "Dataset not found")
    return dataset_out(d)
