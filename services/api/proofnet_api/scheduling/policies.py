"""Eligibility and assignment policies (ARCHITECTURE 6.1, 6.3, 17).

`DeviceEligibilityPolicy` and `AssignmentPolicy` are the Part 2 insertion points: trust-aware
implementations replace the MVP ones without touching the scheduler.
"""

from datetime import datetime, timedelta
from typing import Any, Protocol

from ..config import Settings
from ..worker_gateway.runtime import KERNEL_BUNDLE_VERSION

MIN_BATTERY = 0.2
BROWSER_MEM_CAP_BYTES = 256 * 1024 * 1024
CLI_MEM_BUDGET_BYTES = 1024 * 1024 * 1024


def mem_estimate_bytes(rows: int, n_features: int) -> int:
    """ARCHITECTURE 6.2: rows x (d + 1) x 8 bytes x 3 (input + working copies)."""
    return rows * (n_features + 1) * 8 * 3


def mem_budget_bytes(device: dict[str, Any]) -> int:
    runtime_kind = (device.get("runtime") or {}).get("kind")
    if runtime_kind == "cpython":
        return CLI_MEM_BUDGET_BYTES
    reported_gb = (device.get("capabilities") or {}).get("memory_gb_reported")
    if reported_gb:
        return int(min(BROWSER_MEM_CAP_BYTES, 0.10 * reported_gb * 1e9))
    return BROWSER_MEM_CAP_BYTES


class DeviceEligibilityPolicy(Protocol):
    def allows(self, device: dict[str, Any], task: dict[str, Any]) -> bool:
        """Part 2 hook (trust). MVP: always True."""
        ...


class AllowAll:
    def allows(self, device: dict[str, Any], task: dict[str, Any]) -> bool:
        return True


def ineligibility_reasons(
    device: dict[str, Any],
    *,
    now: datetime,
    settings: Settings,
    rows: int,
    n_features: int,
    excluded_device_ids: list[str] | None = None,
    task: dict[str, Any] | None = None,
    policy: DeviceEligibilityPolicy | None = None,
) -> list[str]:
    """Why a device cannot take this work (empty = eligible). ARCHITECTURE 6.1."""
    reasons: list[str] = []
    if device["status"] != "idle":  # 1
        reasons.append(f"status is {device['status']}")
    seen = device.get("last_seen_at")
    if seen is None or now - seen > timedelta(seconds=settings.offline_after_seconds):
        reasons.append("no recent heartbeat")
    runtime = device.get("runtime") or {}
    if runtime.get("bundle") != KERNEL_BUNDLE_VERSION:  # 2
        reasons.append("kernel bundle version mismatch")
    if not (device.get("benchmark") or {}).get("score_cells_per_sec"):  # 3
        reasons.append("no benchmark yet")
    caps = device.get("capabilities") or {}  # 4: battery >= 20% or charging (unknown = allowed)
    battery = caps.get("battery")
    if battery is not None and battery < MIN_BATTERY and not caps.get("charging"):
        reasons.append(f"battery {round(battery * 100)}% and not charging (needs 20% or charging)")
    need, have = mem_estimate_bytes(rows, n_features), mem_budget_bytes(device)  # 5
    if need > have:
        reasons.append(f"needs ~{need // 2**20} MB, device budget {have // 2**20} MB")
    if device.get("quarantined"):  # Part 2: failed verification, no new work until reinstated
        reasons.append("quarantined after failed verification")
    if device["_id"] in (excluded_device_ids or []):  # 6
        reasons.append("already failed this chunk")
    if not (policy or AllowAll()).allows(device, task or {}):  # 7
        reasons.append("excluded by eligibility policy")
    return reasons


def is_eligible(
    device: dict[str, Any],
    *,
    now: datetime,
    settings: Settings,
    rows: int,
    n_features: int,
    excluded_device_ids: list[str] | None = None,
    task: dict[str, Any] | None = None,
    policy: DeviceEligibilityPolicy | None = None,
) -> bool:
    return not ineligibility_reasons(
        device,
        now=now,
        settings=settings,
        rows=rows,
        n_features=n_features,
        excluded_device_ids=excluded_device_ids,
        task=task,
        policy=policy,
    )


class AssignmentPolicy(Protocol):
    def purposes_for(self, chunk: dict[str, Any]) -> list[str]:
        """Assignments to create for a chunk. MVP: one primary. Part 2: replicas/audits."""
        ...


class OnePrimary:
    def purposes_for(self, chunk: dict[str, Any]) -> list[str]:
        return ["primary"]
