"""ARCHITECTURE 6.1 eligibility rules, including battery >= 20% OR charging."""

from datetime import timedelta
from typing import Any

import pytest

from proofnet_api.config import Settings
from proofnet_api.db import utcnow
from proofnet_api.scheduling.policies import ineligibility_reasons, is_eligible, mem_budget_bytes

S = Settings(jwt_secret="test-secret-test-secret-test-secret-123")


def device(**over: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "_id": "dev_a",
        "status": "idle",
        "last_seen_at": utcnow(),
        "runtime": {"kind": "pyodide", "bundle": "1"},
        "benchmark": {"score_cells_per_sec": 1e7},
        "capabilities": {"memory_gb_reported": 8.0},
    }
    d.update(over)
    return d


def elig(d: dict[str, Any], rows: int = 80_000, nf: int = 16, **kw: Any) -> bool:
    return is_eligible(d, now=utcnow(), settings=S, rows=rows, n_features=nf, **kw)


def test_baseline_eligible() -> None:
    assert elig(device())


@pytest.mark.parametrize(
    ("caps", "expected"),
    [
        ({"battery": 0.10}, False),  # low and not charging
        ({"battery": 0.10, "charging": False}, False),
        ({"battery": 0.10, "charging": True}, True),  # low but on the charger
        ({"battery": 0.19}, False),
        ({"battery": 0.20}, True),  # boundary: >= 20%
        ({"battery": 0.80}, True),
        ({}, True),  # unknown battery is allowed
    ],
)
def test_battery_or_charging(caps: dict[str, Any], expected: bool) -> None:
    assert elig(device(capabilities={"memory_gb_reported": 8.0, **caps})) is expected


def test_reasons_are_human_readable() -> None:
    why = ineligibility_reasons(
        device(capabilities={"battery": 0.1, "memory_gb_reported": 8.0}),
        now=utcnow(),
        settings=S,
        rows=80_000,
        n_features=16,
    )
    assert why == ["battery 10% and not charging (needs 20% or charging)"]


@pytest.mark.parametrize(
    ("over", "needle"),
    [
        ({"status": "busy"}, "status is busy"),
        ({"status": "offline"}, "status is offline"),
        ({"last_seen_at": utcnow() - timedelta(seconds=25)}, "no recent heartbeat"),
        ({"last_seen_at": None}, "no recent heartbeat"),
        ({"runtime": {"kind": "pyodide", "bundle": "9"}}, "bundle version"),
        ({"benchmark": None}, "no benchmark"),
    ],
)
def test_other_conditions(over: dict[str, Any], needle: str) -> None:
    why = ineligibility_reasons(device(**over), now=utcnow(), settings=S, rows=1000, n_features=4)
    assert any(needle in w for w in why), why


def test_memory_budget_and_exclusion() -> None:
    d = device()
    assert mem_budget_bytes(d) == 256 * 1024 * 1024  # min(256 MB cap, 10% of 8 GB)
    small = device(capabilities={"memory_gb_reported": 1.0})
    assert mem_budget_bytes(small) == int(0.1 * 1.0 * 1e9)  # 10% of 1 GB < cap
    assert not elig(d, rows=300_000, nf=64)  # ~460 MB needed > 256 MB cap
    assert not elig(d, excluded_device_ids=["dev_a"])
    cli = device(runtime={"kind": "cpython", "bundle": "1"})
    assert elig(cli, rows=300_000, nf=64)  # CLI budget is larger


def test_eligibility_policy_hook_can_veto() -> None:
    class Deny:
        def allows(self, device: dict[str, Any], task: dict[str, Any]) -> bool:
            return False

    assert not elig(device(), policy=Deny())
