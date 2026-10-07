"""Request/response models for the REST API (ARCHITECTURE 13)."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, EmailStr, Field

from .worker import RuntimeInfo


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    display_name: str = Field(min_length=1, max_length=80)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: str
    email: str
    display_name: str
    roles: list[str]


class DeviceType(StrEnum):
    android_phone = "android_phone"
    laptop = "laptop"
    desktop = "desktop"
    other = "other"


class DeviceCapabilities(BaseModel):
    model: str | None = None
    platform_version: str | None = None
    logical_cores: int | None = Field(default=None, ge=1, le=1024)
    memory_gb_reported: float | None = Field(default=None, ge=0, le=1024)
    storage_quota_mb: float | None = Field(default=None, ge=0)
    battery: float | None = Field(default=None, ge=0, le=1)
    network_type: str | None = None


class Benchmark(BaseModel):
    score_cells_per_sec: float = Field(gt=0)
    bench_version: str


class DeviceRegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    device_type: DeviceType = DeviceType.other
    capabilities: DeviceCapabilities = DeviceCapabilities()


class DevicePatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    disabled: bool | None = None


class DeviceOut(BaseModel):
    id: str
    name: str
    device_type: DeviceType
    status: str
    last_seen_at: datetime | None = None
    capabilities: DeviceCapabilities
    runtime: RuntimeInfo | None = None
    benchmark: Benchmark | None = None
    stats: dict[str, int]


class DeviceRegistered(BaseModel):
    device: DeviceOut
    device_token: str = Field(description="Shown once; only its hash is stored.")


class SessionRequest(BaseModel):
    capabilities: DeviceCapabilities = DeviceCapabilities()
    runtime: RuntimeInfo
    benchmark: Benchmark


class SessionResponse(BaseModel):
    session_id: str
    heartbeat_idle_ms: int
    heartbeat_busy_ms: int


class RuntimeManifest(BaseModel):
    pyodide_version: str
    numpy_version: str
    kernel_bundle_version: str
    kernel_bundle_sha256: str
    heartbeat_idle_ms: int
    heartbeat_busy_ms: int
    offline_after_seconds: int


class DatasetOut(BaseModel):
    id: str
    filename: str
    size_bytes: int
    sha256: str
    profile: dict[str, Any]
    created_at: datetime


class ValidationResult(BaseModel):
    ok: bool
    errors: list[str]
    warnings: list[str]
    summary: dict[str, Any]
    plan_preview: dict[str, Any] | None = Field(
        default=None, description="Populated from P5 (device-aware planner)."
    )


class TaskTypeInfo(BaseModel):
    task_type: str
    kernel_version: str
    description: str
    params_schema: dict[str, Any]


class TaskOut(BaseModel):
    id: str
    name: str
    task_type: str
    kernel_version: str
    dataset_id: str
    params: dict[str, Any]
    execution: dict[str, Any]
    validation: dict[str, Any]
    prepared: dict[str, Any]
    plan: dict[str, Any] | None = None
    status: str
    status_history: list[dict[str, Any]]
    result: dict[str, Any] | None = None
    error: str | None = None
    verification_policy: dict[str, Any]
    created_at: datetime


class NetworkDevice(BaseModel):
    id: str
    name: str
    device_type: DeviceType
    status: str
    score_cells_per_sec: float | None = None
    runtime_kind: str | None = None
    last_seen_age_seconds: float | None = None
    current_assignment_id: str | None = None


class NetworkSummary(BaseModel):
    server_time: datetime
    counts: dict[str, int]
    devices: list[NetworkDevice]
    tasks_running: int


class ChunkOut(BaseModel):
    id: str
    index: int
    role: str
    row_start: int
    row_end: int
    n_rows: int
    status: str
    attempt_count: int
    max_attempts: int
    preferred_device_id: str | None = None
    accepted_assignment_id: str | None = None


class AssignmentOut(BaseModel):
    id: str
    chunk_id: str
    device_id: str
    device_name: str | None = None
    attempt_no: int
    purpose: str
    status: str
    assigned_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    deadline_at: datetime
    timings: dict[str, float] | None = None
    runtime_fingerprint: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class ArtifactOut(BaseModel):
    id: str
    task_id: str
    kind: str
    filename: str
    size_bytes: int
    sha256: str
    created_at: datetime


class TaskStatus(BaseModel):
    """Compact live snapshot for the task monitor (real state only)."""

    server_time: datetime
    task: TaskOut
    chunks: list[ChunkOut]
    assignments: list[AssignmentOut]
    artifacts: list[ArtifactOut]


class EventOut(BaseModel):
    id: str
    ts: datetime
    type: str
    task_id: str | None = None
    device_id: str | None = None
    chunk_id: str | None = None
    assignment_id: str | None = None
    data: dict[str, Any]


class ResultAck(BaseModel):
    assignment_id: str
    status: str
    acceptance: str | None = None
