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
    """All conditions of ARCHITECTURE 6.1."""
    if device["status"] != "idle":  # 1 (disabled/offline/busy/initializing all fail)
        return False
    seen = device.get("last_seen_at")
    if seen is None or now - seen > timedelta(seconds=settings.offline_after_seconds):
        return False
    runtime = device.get("runtime") or {}
    if runtime.get("bundle") != KERNEL_BUNDLE_VERSION:  # 2
        return False
    if not (device.get("benchmark") or {}).get("score_cells_per_sec"):  # 3
        return False
    battery = (device.get("capabilities") or {}).get("battery")  # 4 (unknown = allowed)
    if battery is not None and battery < MIN_BATTERY:
        return False
    if mem_estimate_bytes(rows, n_features) > mem_budget_bytes(device):  # 5
        return False
    if device["_id"] in (excluded_device_ids or []):  # 6
        return False
    return (policy or AllowAll()).allows(device, task or {})  # 7


class AssignmentPolicy(Protocol):
    def purposes_for(self, chunk: dict[str, Any]) -> list[str]:
        """Assignments to create for a chunk. MVP: one primary. Part 2: replicas/audits."""
        ...


class OnePrimary:
    def purposes_for(self, chunk: dict[str, Any]) -> list[str]:
        return ["primary"]
