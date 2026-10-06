"""Shared helpers for API tests."""

import secrets
from typing import Any

from fastapi.testclient import TestClient

SESSION_BODY: dict[str, Any] = {
    "runtime": {"kind": "cpython", "python": "3.12", "numpy": "2.4.6", "bundle": "1"},
    "benchmark": {"score_cells_per_sec": 2.5e6, "bench_version": "bench_v1"},
}


def signup(client: TestClient, email: str | None = None) -> dict[str, str]:
    email = email or f"u{secrets.token_hex(4)}@example.com"
    r = client.post(
        "/api/v1/auth/signup",
        json={"email": email, "password": "password123", "display_name": "Tester"},
    )
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def register_device(
    client: TestClient, headers: dict[str, str], name: str = "d"
) -> tuple[str, dict[str, str]]:
    r = client.post(
        "/api/v1/devices", headers=headers, json={"name": name, "device_type": "laptop"}
    )
    assert r.status_code == 201, r.text
    body = r.json()
    return body["device"]["id"], {"Authorization": f"Bearer {body['device_token']}"}
