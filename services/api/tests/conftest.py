"""API tests run against a real MongoDB (MONGODB_URI from the environment / .env).

Each test session uses a throwaway database (proofnet_test_<random>) that is dropped afterwards.
"""

import os
import secrets
from collections.abc import Iterator
from typing import Any

import pytest
from api_helpers import signup
from fastapi.testclient import TestClient
from pymongo import MongoClient

from proofnet_api.config import Settings
from proofnet_api.main import create_app

os.environ.setdefault("RATE_LIMIT_AUTH_PER_MINUTE", "1000000")  # tests sign in constantly
_BASE = Settings()  # reads .env / environment


@pytest.fixture(scope="session")
def test_db_name() -> Iterator[str]:
    name = f"proofnet_test_{secrets.token_hex(4)}"
    yield name
    MongoClient(_BASE.mongodb_uri).drop_database(name)


@pytest.fixture(scope="session")
def settings(test_db_name: str) -> Settings:
    return Settings(
        mongodb_uri=_BASE.mongodb_uri,
        mongodb_db=test_db_name,
        jwt_secret="test-secret-test-secret-test-secret-123",
        status_cache_seconds=0,
        offline_after_seconds=20,
        reconciler_interval_seconds=3600,  # tests drive reconciler passes explicitly
    )


@pytest.fixture(scope="session")
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture(scope="session")
def mongo(settings: Settings) -> Iterator[MongoClient[dict[str, Any]]]:
    c: MongoClient[dict[str, Any]] = MongoClient(settings.mongodb_uri, tz_aware=True)
    yield c
    c.close()


@pytest.fixture()
def user_headers(client: TestClient) -> dict[str, str]:
    return signup(client)


@pytest.fixture(scope="module")
def env() -> Iterator[tuple[TestClient, MongoClient[dict[str, Any]], Settings]]:
    """Isolated database per module (so devices from other tests cannot take the work)."""
    s = Settings(
        mongodb_uri=_BASE.mongodb_uri,
        mongodb_db=f"proofnet_test_e2e_{secrets.token_hex(4)}",
        jwt_secret="test-secret-test-secret-test-secret-123",
        status_cache_seconds=0,
        reconciler_interval_seconds=1,
    )
    mongo: MongoClient[dict[str, Any]] = MongoClient(s.mongodb_uri, tz_aware=True)
    try:
        with TestClient(create_app(s)) as c:
            yield c, mongo, s
    finally:
        mongo.drop_database(s.mongodb_db)
        mongo.close()


def _make_env(**over: Any) -> Iterator[tuple[TestClient, MongoClient[dict[str, Any]], Settings]]:
    s = Settings(
        mongodb_uri=_BASE.mongodb_uri,
        mongodb_db=f"proofnet_test_rel_{secrets.token_hex(4)}",
        jwt_secret="test-secret-test-secret-test-secret-123",
        status_cache_seconds=0,
        reconciler_interval_seconds=1,
        **over,
    )
    mongo: MongoClient[dict[str, Any]] = MongoClient(s.mongodb_uri, tz_aware=True)
    try:
        with TestClient(create_app(s)) as c:
            yield c, mongo, s
    finally:
        mongo.drop_database(s.mongodb_db)
        mongo.close()


@pytest.fixture(scope="module")
def fast_env() -> Iterator[tuple[TestClient, MongoClient[dict[str, Any]], Settings]]:
    """Short timeouts so failure handling can be exercised in seconds."""
    yield from _make_env(
        offline_after_seconds=2,
        exclusion_relax_seconds=2,
        preferred_wait_seconds=1,
        queue_timeout_seconds=4,
    )


@pytest.fixture(scope="module")
def timeout_env() -> Iterator[tuple[TestClient, MongoClient[dict[str, Any]], Settings]]:
    yield from _make_env(offline_after_seconds=30, task_timeout_seconds=3)
