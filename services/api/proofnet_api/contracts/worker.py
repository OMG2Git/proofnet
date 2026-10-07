"""Worker protocol (ARCHITECTURE 8)."""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class DeviceState(StrEnum):
    initializing = "initializing"
    idle = "idle"
    busy = "busy"


class RuntimeInfo(BaseModel):
    kind: Literal["pyodide", "cpython"]
    python: str | None = None
    pyodide: str | None = None
    numpy: str
    bundle: str


class HeartbeatRequest(BaseModel):
    session_id: str
    state: DeviceState
    current_assignment_id: str | None = None
    battery: float | None = Field(default=None, ge=0, le=1)
    charging: bool | None = None


class AssignmentPayload(BaseModel):
    assignment_id: str = Field(pattern=r"^asg_")
    task_id: str = Field(pattern=r"^tsk_")
    chunk_id: str = Field(pattern=r"^chk_")
    kernel: str
    kernel_version: str
    params: dict[str, Any]
    input_url: str
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    n_rows: int = Field(ge=1)
    n_features: int = Field(ge=1)
    deadline_at: datetime


class RunDirective(BaseModel):
    type: Literal["run"] = "run"
    assignment: AssignmentPayload


class CancelDirective(BaseModel):
    type: Literal["cancel"] = "cancel"
    assignment_id: str


class RefreshRuntimeDirective(BaseModel):
    type: Literal["refresh_runtime"] = "refresh_runtime"
    kernel_version: str


class ReregisterDirective(BaseModel):
    type: Literal["reregister"] = "reregister"


Directive = Annotated[
    RunDirective | CancelDirective | RefreshRuntimeDirective | ReregisterDirective,
    Field(discriminator="type"),
]


class HeartbeatResponse(BaseModel):
    server_time: datetime
    next_heartbeat_ms: int = Field(ge=500)
    directives: list[Directive] = []


class ResultTimings(BaseModel):
    download_ms: float = Field(ge=0)
    compute_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)


class PartialResultEnvelope(BaseModel):
    """Worker result: JSON only, never pickle (D9)."""

    model_config = ConfigDict(extra="forbid")

    kernel: str
    kernel_version: str
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    n_rows: int = Field(ge=1)
    payload: dict[str, Any]
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    timings: ResultTimings
    runtime: RuntimeInfo


class FailRequest(BaseModel):
    code: str
    message: str
