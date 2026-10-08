"""Storage guard for the free Atlas tier: release old data over budget, never active data."""

import asyncio
import secrets
from collections.abc import Iterator
from datetime import timedelta
from typing import Any

import pytest
from pymongo import MongoClient

from proofnet_api.config import Settings
from proofnet_api.db import connect, utcnow
from proofnet_api.errors import ProofNetError
from proofnet_api.storage import ensure_capacity, free_space, size_mb


@pytest.fixture()
def dbname() -> Iterator[str]:
    name = f"proofnet_test_storage_{secrets.token_hex(4)}"
    yield name
    MongoClient(Settings().mongodb_uri).drop_database(name)


def settings(dbname: str, budget_mb: float) -> Settings:
    return Settings(
        mongodb_uri=Settings().mongodb_uri,
        mongodb_db=dbname,
        jwt_secret="test-secret-test-secret-test-secret-123",
        storage_budget_mb=int(budget_mb) if budget_mb >= 1 else 1,
    )


async def seed(db: Any) -> dict[str, Any]:
    """Old finished tasks (prepared files + gradient payloads), an active task, datasets."""
    blob = b"x" * (1_000_000)
    now = utcnow()
    ids: dict[str, Any] = {"prepared": []}
    for i in range(3):
        fid = await db.fs.upload_from_stream(f"prep{i}", blob)
        ids["prepared"].append(fid)
        await db.col("tasks").insert_one(
            {
                "_id": f"tsk_done{i}",
                "status": "completed",
                "created_at": now - timedelta(days=3 - i),
                "prepared": {"file_id": fid},
                "training": {"steps": 1},
                "dataset_id": f"ds_old{i}",
            }
        )
        await db.col("partial_results").insert_one(
            {"_id": f"pr{i}", "task_id": f"tsk_done{i}", "payload": {"g": "y" * 200_000}}
        )
    active_fid = await db.fs.upload_from_stream("prep_active", blob)
    ids["active_prepared"] = active_fid
    await db.col("tasks").insert_one(
        {
            "_id": "tsk_run",
            "status": "running",
            "created_at": now,
            "prepared": {"file_id": active_fid},
            "dataset_id": "ds_in_use",
        }
    )
    for name in ("ds_old0", "ds_in_use"):
        raw = await db.fs.upload_from_stream(name, blob)
        ids[name] = raw
        await db.col("datasets").insert_one(
            {"_id": name, "raw_file_id": raw, "created_at": now - timedelta(days=9)}
        )
    return ids


def test_free_space_releases_old_data_but_never_active_data(dbname: str) -> None:
    async def go() -> dict[str, Any]:
        s = settings(dbname, 1)  # a 1 MB budget: everything above it is released
        db = await connect(s)
        try:
            ids = await seed(db)
            assert await size_mb(db) > 5
            report = await free_space(db, s)
            report["ids"] = ids
            report["tasks"] = {t["_id"]: t async for t in db.col("tasks").find({})}
            report["datasets"] = {d["_id"]: d async for d in db.col("datasets").find({})}
            report["partials"] = [p async for p in db.col("partial_results").find({})]
            report["files"] = {f["_id"] async for f in db.db["fs.files"].find({})}
            return report
        finally:
            await db.close()

    r = asyncio.run(go())
    assert r["freed"] is True and r["after_mb"] < r["before_mb"]
    assert r["gradient_payloads_dropped"] == 3
    assert all("payload" not in p and p["payload_dropped"] for p in r["partials"])  # metadata kept
    done = [r["tasks"][f"tsk_done{i}"] for i in range(3)]
    assert all(t["prepared"]["dropped"] is True for t in done)  # finished tasks lose prepared data
    assert not any(fid in r["files"] for fid in r["ids"]["prepared"])
    assert (
        r["tasks"]["tsk_run"]["status"] == "running"
        and "dropped" not in r["tasks"]["tsk_run"]["prepared"]
    )
    assert r["ids"]["active_prepared"] in r["files"]  # an active task's data is never touched
    assert r["datasets"]["ds_in_use"].get("raw_dropped") is None  # used by the running task
    assert r["ids"]["ds_in_use"] in r["files"]


def test_free_space_is_a_noop_under_budget(dbname: str) -> None:
    async def go() -> dict[str, Any]:
        s = settings(dbname, 300)
        db = await connect(s)
        try:
            await seed(db)
            return await free_space(db, s)
        finally:
            await db.close()

    r = asyncio.run(go())
    assert r["freed"] is False and "prepared_dropped" not in r


def test_uploads_are_refused_with_507_when_nothing_can_be_freed(dbname: str) -> None:
    async def go() -> None:
        s = settings(dbname, 1)
        db = await connect(s)
        try:
            # only active data (cannot be released) fills the budget
            blob = b"x" * 3_000_000
            fid = await db.fs.upload_from_stream("active", blob)
            await db.col("tasks").insert_one(
                {"_id": "t", "status": "running", "prepared": {"file_id": fid}, "dataset_id": "d"}
            )
            with pytest.raises(ProofNetError) as e:
                await ensure_capacity(db, s, need_mb=1)
            assert e.value.status == 507 and e.value.code == "STORAGE_FULL"
        finally:
            await db.close()

    asyncio.run(go())


def test_startup_survives_a_database_that_blocks_writes(
    dbname: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full free-tier database rejects index creation; the app must still start (logins work)."""
    from proofnet_api import db as dbmod

    async def blocked(_: Any) -> None:
        raise RuntimeError("Over the quota: writes are blocked")

    monkeypatch.setattr(dbmod, "ensure_indexes", blocked)

    async def go() -> bool:
        db = await connect(settings(dbname, 300))
        try:
            await db.client.admin.command("ping")
            return True
        finally:
            await db.close()

    assert asyncio.run(go()) is True
