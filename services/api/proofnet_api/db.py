"""MongoDB (async PyMongo) + GridFS. The only persistent store (D4)."""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from gridfs.asynchronous import AsyncGridFSBucket
from pymongo import ASCENDING, AsyncMongoClient
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase

from .config import Settings


def utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass
class Db:
    client: AsyncMongoClient[dict[str, Any]]
    db: AsyncDatabase[dict[str, Any]]
    fs: AsyncGridFSBucket
    indexes_ready: bool = False

    def col(self, name: str) -> AsyncCollection[dict[str, Any]]:
        return self.db[name]

    async def close(self) -> None:
        await self.client.close()


async def connect(settings: Settings) -> Db:
    client: AsyncMongoClient[dict[str, Any]] = AsyncMongoClient(
        settings.mongodb_uri, tz_aware=True, serverSelectionTimeoutMS=15000
    )
    db = client[settings.mongodb_db]
    handle = Db(client, db, AsyncGridFSBucket(db))
    await try_ensure_indexes(handle)
    return handle


async def try_ensure_indexes(h: Db) -> bool:
    """Create indexes; on failure (database full, network hiccup) keep serving and let the
    reconciler retry. Unique indexes matter (e.g. one account per email), so this must converge."""
    if h.indexes_ready:
        return True
    try:
        await ensure_indexes(h)
        h.indexes_ready = True
    except Exception:
        logging.getLogger("proofnet.db").exception("could not ensure indexes (will retry)")
    return h.indexes_ready


async def ensure_indexes(h: Db) -> None:
    """Minimum indexes from ARCHITECTURE 12 (plus uniqueness for login/device auth)."""
    await h.col("users").create_index("email", unique=True)
    await h.col("devices").create_index([("status", ASCENDING), ("last_seen_at", ASCENDING)])
    await h.col("devices").create_index("token_hash", unique=True)
    await h.col("devices").create_index("owner_user_id")
    await h.col("datasets").create_index("owner_user_id")
    await h.col("tasks").create_index([("owner_user_id", ASCENDING), ("created_at", ASCENDING)])
    await h.col("chunks").create_index([("task_id", ASCENDING), ("status", ASCENDING)])
    await h.col("assignments").create_index([("device_id", ASCENDING), ("status", ASCENDING)])
    await h.col("assignments").create_index("chunk_id")
    await h.col("events").create_index([("task_id", ASCENDING), ("ts", ASCENDING)])
    await h.col("image_datasets").create_index("owner_user_id")
    await h.col("training_rounds").create_index([("task_id", ASCENDING), ("round", ASCENDING)])
    await h.col("model_states").create_index("task_id")
    # iterative tasks create one chunk set per round; (task, round, index) must be unique
    await h.col("chunks").create_index(
        [("task_id", ASCENDING), ("round", ASCENDING), ("index", ASCENDING)],
        unique=True,
        partialFilterExpression={"round": {"$exists": True}},
    )
