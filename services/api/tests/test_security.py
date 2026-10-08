"""Security hardening and trust APIs: lockout, rate limits, headers, tenant isolation, admin-only
quarantine/reinstatement, dashboards' API, simulator, ledger check."""

import secrets
from collections.abc import Iterator
from typing import Any

import pytest
from e2e_helpers import V1, Env, make_workers, only_these
from fastapi.testclient import TestClient
from pymongo import MongoClient

from proofnet_api.config import Settings
from proofnet_api.main import create_app
from proofnet_api.security.ratelimit import RateLimiter

ADMIN = "boss@example.com"


@pytest.fixture(scope="module")
def senv() -> Iterator[Env]:
    s = Settings(
        mongodb_uri=Settings().mongodb_uri,
        mongodb_db=f"proofnet_test_sec_{secrets.token_hex(4)}",
        jwt_secret="test-secret-test-secret-test-secret-123",
        status_cache_seconds=0,
        reconciler_interval_seconds=1,
        admin_emails=ADMIN,
        rate_limit_auth_per_minute=40,
        login_max_failures=3,
        login_lockout_seconds=2,
    )
    mongo: MongoClient[dict[str, Any]] = MongoClient(s.mongodb_uri, tz_aware=True)
    try:
        with TestClient(create_app(s)) as c:
            yield c, mongo, s
    finally:
        mongo.drop_database(s.mongodb_db)
        mongo.close()


def col(env: Env, name: str) -> Any:
    _, mongo, s = env
    return mongo[s.mongodb_db][name]


def mk_user(client: TestClient, email: str, pw: str = "password123") -> dict[str, str]:
    r = client.post(f"{V1}/auth/signup", json={"email": email, "password": pw, "display_name": "T"})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


# ---------------------------------------------------------------- unit: limiter
def test_sliding_window_limiter() -> None:
    lim = RateLimiter()
    assert [lim.check("k", 3, 10.0, now=t) for t in (0.0, 1.0, 2.0)] == [0.0, 0.0, 0.0]
    wait = lim.check("k", 3, 10.0, now=3.0)
    assert 6.9 < wait <= 7.0  # the first hit leaves the window at t = 10
    assert lim.check("other", 3, 10.0, now=3.0) == 0.0  # keys are independent
    assert lim.check("k", 3, 10.0, now=10.5) == 0.0  # slot freed
    for i in range(30000):  # memory stays bounded under key spraying
        lim.check(f"spray{i}", 1, 1.0, now=100.0 + i * 0.001)
    assert len(lim.hits) < 26000


# ---------------------------------------------------------------- lockout / rate limit / headers
def test_account_locks_after_repeated_failures_then_recovers(senv: Env) -> None:
    client, _, _ = senv
    mk_user(client, "lock@example.com")
    body = {"email": "lock@example.com", "password": "wrong-password"}
    codes = [client.post(f"{V1}/auth/login", json=body).status_code for _ in range(3)]
    assert codes == [401, 401, 401]
    r = client.post(f"{V1}/auth/login", json={**body, "password": "password123"})
    assert (
        r.status_code == 429 and r.json()["error"]["code"] == "ACCOUNT_LOCKED"
    )  # even the right one
    import time

    time.sleep(2.3)
    ok = client.post(f"{V1}/auth/login", json={**body, "password": "password123"})
    assert ok.status_code == 200
    kinds = {e["kind"] for e in col(senv, "security_events").find({})}
    assert "login_lockout" in kinds


def test_unknown_accounts_are_throttled_the_same_way(senv: Env) -> None:
    client, _, _ = senv
    body = {"email": "ghost@example.com", "password": "x" * 10}
    codes = [client.post(f"{V1}/auth/login", json=body).status_code for _ in range(4)]
    assert codes[:3] == [401, 401, 401] and codes[3] == 429  # no user enumeration via behaviour


def test_security_headers_present(senv: Env) -> None:
    client, _, _ = senv
    r = client.get(f"{V1}/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["cache-control"] == "no-store"
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]


# ---------------------------------------------------------------- isolation + admin
def test_foreign_assignment_probe_is_404_and_logged(senv: Env) -> None:
    client, _, s = senv
    mk_user(client, "iso@example.com")
    a, b = make_workers(client, "iso", 2)
    only_these(senv, [a, b])
    col(senv, "assignments").insert_one(
        {
            "_id": "asg_foreign",
            "device_id": b.device_id,
            "task_id": "t",
            "chunk_id": "c",
            "status": "assigned",
        }
    )
    r = client.post(f"{V1}/worker/assignments/asg_foreign/start", headers=a._dev())
    assert r.status_code == 404
    ev = col(senv, "security_events").find_one({"kind": "foreign_assignment_access"})
    assert ev is not None and ev["device_id"] == a.device_id and ev["severity"] == "warning"
    del s


def test_quarantine_and_reinstate_are_admin_only_and_effective(senv: Env) -> None:
    client, _, _ = senv
    admin = mk_user(client, ADMIN)
    user = mk_user(client, "plain@example.com")
    (w,) = make_workers(client, "qa", 1)
    only_these(senv, [w])
    url = f"{V1}/security/devices/{w.device_id}"
    assert client.post(f"{url}/quarantine", headers=user, json={"reason": "x"}).status_code == 403
    assert (
        client.post(f"{url}/quarantine", headers=admin, json={"reason": "demo"}).status_code == 200
    )
    assert (
        client.post(f"{url}/quarantine", headers=admin, json={"reason": "again"}).status_code == 409
    )
    dev = col(senv, "devices").find_one({"_id": w.device_id})
    assert dev["quarantined"] is True
    hb = client.get(f"{V1}/devices/mine", headers=user)
    assert hb.status_code == 200
    ov = client.get(f"{V1}/security/overview", headers=admin).json()
    assert any(d["device_id"] == w.device_id for d in ov["quarantined_devices"])
    assert any(e["kind"] == "device_quarantined" for e in ov["events"])
    assert len(ov["controls"]) >= 8
    assert client.post(f"{url}/reinstate", headers=user, json={"reason": "x"}).status_code == 403
    assert (
        client.post(f"{url}/reinstate", headers=admin, json={"reason": "reviewed"}).status_code
        == 200
    )
    p = col(senv, "device_trust").find_one({"_id": w.device_id})
    assert p["status"] == "probation" and p["memory"] >= 0.5  # heavily audited again
    assert col(senv, "devices").find_one({"_id": w.device_id})["quarantined"] is False


def test_sybil_new_device_inherits_suspicion(senv: Env) -> None:
    client, _, _ = senv
    admin = mk_user(client, "sybadmin@example.com")
    col(senv, "users").update_one({"email": "sybadmin@example.com"}, {"$set": {"roles": ["admin"]}})
    h = mk_user(client, "sybil@example.com")
    (other,) = make_workers(client, "sybother", 1)  # another account: must not inherit anything

    def register(name: str) -> str:
        r = client.post(
            f"{V1}/devices",
            headers=h,
            json={"name": name, "device_type": "laptop", "capabilities": {}},
        )
        assert r.status_code == 201, r.text
        return str(r.json()["device"]["id"])

    first = register("first")
    q = client.post(
        f"{V1}/security/devices/{first}/quarantine", headers=admin, json={"reason": "cheated"}
    )
    assert q.status_code == 200, q.text
    second = register("fresh-identity")  # the same person registers again to shed the record
    t = client.get(f"{V1}/trust/devices/{second}", headers=h).json()
    assert t["memory"] == pytest.approx(0.5) and t["status"] == "probation"
    assert t["audit_probability"] == 1.0  # probation: audited on everything anyway
    assert client.get(f"{V1}/trust/devices/{first}", headers=h).json()["status"] == "quarantined"
    prof = col(senv, "device_trust").find_one({"_id": other.device_id})
    assert prof is None or prof["memory"] == 0.0


def test_trust_and_rewards_endpoints_are_scoped_to_the_user(senv: Env) -> None:
    client, _, _ = senv
    h1 = mk_user(client, "scope1@example.com")
    h2 = mk_user(client, "scope2@example.com")
    ov1 = client.get(f"{V1}/trust/overview", headers=h1)
    assert ov1.status_code == 200 and ov1.json()["devices"] == 0
    assert client.get(f"{V1}/rewards/me", headers=h2).json()["balance"] == {
        "pending": 0.0,
        "confirmed": 0.0,
        "revoked": 0.0,
    }
    assert client.get(f"{V1}/rewards/ledger/check", headers=h1).json()["consistent"] is True
    assert client.get(f"{V1}/trust/overview").status_code == 401
    r = client.get(f"{V1}/trust/devices/dev_nope", headers=h1)
    assert r.status_code == 404


# ---------------------------------------------------------------- simulator
def test_simulator_adaptive_beats_fixed_on_cost_and_none_on_safety(senv: Env) -> None:
    client, _, _ = senv
    h = mk_user(client, "sim@example.com")
    body = {
        "honest": 20,
        "attackers": 5,
        "rounds": 300,
        "policy": "adaptive",
        "seed": 3,
        "cheat_rate": 1.0,
        "attack_strength": 1.0,
    }
    r = client.post(f"{V1}/trust/simulate", headers=h, json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    cmp = out["summary"]["comparison"]
    assert cmp["none"]["corrupt_merged_fraction"] == 1.0  # no audits: every cheat gets merged
    assert cmp["adaptive"]["attackers_caught"] == 5 and cmp["adaptive"]["false_accusations"] == 0
    assert cmp["adaptive"]["audit_cost_fraction"] < cmp["fixed"]["audit_cost_fraction"]
    assert cmp["adaptive"]["corrupt_merged_fraction"] < 0.2
    assert len(out["series"]["audits"]) == 300
    again = client.post(f"{V1}/trust/simulate", headers=h, json=body).json()
    assert again["summary"] == out["summary"]  # reproducible for a given seed
    bad = client.post(f"{V1}/trust/simulate", headers=h, json={**body, "rounds": 10**6})
    assert bad.status_code == 422


def test_rate_limit_on_auth_endpoints(senv: Env) -> None:
    client, _, _ = senv
    seen = set()
    for i in range(80):
        r = client.post(
            f"{V1}/auth/signup",
            json={"email": f"rl{i}@example.com", "password": "password123", "display_name": "x"},
        )
        seen.add(r.status_code)
    assert 429 in seen  # the 40/min budget is exhausted
    rl = [e for e in col(senv, "security_events").find({"kind": "rate_limited"})]
    assert rl and rl[0]["data"]["scope"] in ("signup", "login", "register")
