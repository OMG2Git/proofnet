"""API tests run against a real MongoDB (MONGODB_URI from the environment / .env).

Each test session uses a throwaway database (proofnet_test_<random>) that is dropped afterwards.
"""

import secrets
from collections.abc import Iterator
from typing import Any

import pytest
from api_helpers import signup
from fastapi.testclient import TestClient
from pymongo import MongoClient

from proofnet_api.config import Settings
from proofnet_api.main import create_app

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
