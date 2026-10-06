import time

import pytest
from fastapi.testclient import TestClient

from proofnet_api.auth.security import (
    create_access_token,
    decode_access_token,
    hash_device_token,
    hash_password,
    verify_device_token,
    verify_password,
)
from proofnet_api.config import Settings
from proofnet_api.errors import ProofNetError
from proofnet_api.ids import PREFIXES, new_id, new_token
from proofnet_api.main import app

client = TestClient(app)


def test_ids_prefixed_unique_and_time_ordered() -> None:
    a = new_id("tsk")
    time.sleep(0.002)
    b = new_id("tsk")
    assert a.startswith("tsk_") and a != b and a < b
    assert len({new_id("dev") for _ in range(1000)}) == 1000
    for p in PREFIXES:
        assert new_id(p).startswith(p + "_")
    with pytest.raises(ValueError):
        new_id("xxx")


def test_password_hashing() -> None:
    h = hash_password("correct horse")
    assert h != "correct horse"
    assert verify_password("correct horse", h)
    assert not verify_password("wrong", h)
    assert not verify_password("x", "not-a-hash")


def test_jwt_roundtrip_expiry_and_tamper() -> None:
    tok = create_access_token("usr_1", ["user"], "test-secret-test-secret-test-secret-123", 5)
    assert decode_access_token(tok, "test-secret-test-secret-test-secret-123")["sub"] == "usr_1"
    with pytest.raises(ProofNetError):
        decode_access_token(tok, "other-secret-other-secret-other-secret-1")
    old = create_access_token(
        "usr_1", ["user"], "test-secret-test-secret-test-secret-123", 1, now=time.time() - 3600
    )
    with pytest.raises(ProofNetError):
        decode_access_token(old, "test-secret-test-secret-test-secret-123")
    with pytest.raises(ProofNetError):
        decode_access_token(tok[:-2] + "xx", "test-secret-test-secret-test-secret-123")


def test_device_token_hash() -> None:
    t = new_token()
    h = hash_device_token(t)
    assert len(h) == 64 and t not in h
    assert verify_device_token(t, h) and not verify_device_token(t + "x", h)


def test_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "http://a.com, https://b.com/")
    monkeypatch.setenv("MONGODB_DB", "x")
    s = Settings()
    assert s.cors_origin_list == ["http://a.com", "https://b.com"] and s.mongodb_db == "x"
    assert s.offline_after_seconds == 20 and s.max_attempts == 3


def test_error_format_validation_and_404(client: TestClient) -> None:
    r = client.post("/api/v1/auth/signup", json={"email": "bad"})
    assert r.status_code == 422
    body = r.json()["error"]
    assert body["code"] == "VALIDATION_FAILED" and body["details"]
    r = client.get("/api/v1/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "NOT_FOUND"


def test_cors_allows_configured_origin() -> None:
    r = client.options(
        "/api/v1/health",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
    )
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"
