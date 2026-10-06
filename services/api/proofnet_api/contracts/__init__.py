"""Contract models (ARCHITECTURE 4.4, 8, 13). FastAPI OpenAPI is the API source of truth."""

from .errors import ApiError, ErrorBody
from .manifest import (
    ExecutionSettings,
    GaussianNBParams,
    LinearRidgeParams,
    MissingValuePolicy,
    StartPolicy,
    TaskManifest,
    TaskType,
)
from .worker import (
    AssignmentPayload,
    CancelDirective,
    DeviceState,
    Directive,
    FailRequest,
    HeartbeatRequest,
    HeartbeatResponse,
    PartialResultEnvelope,
    RefreshRuntimeDirective,
    ReregisterDirective,
    ResultTimings,
    RunDirective,
    RuntimeInfo,
)

__all__ = [
    "ApiError",
    "AssignmentPayload",
    "CancelDirective",
    "DeviceState",
    "Directive",
    "ErrorBody",
    "ExecutionSettings",
    "FailRequest",
    "GaussianNBParams",
    "HeartbeatRequest",
    "HeartbeatResponse",
    "LinearRidgeParams",
    "MissingValuePolicy",
    "PartialResultEnvelope",
    "RefreshRuntimeDirective",
    "ReregisterDirective",
    "ResultTimings",
    "RunDirective",
    "RuntimeInfo",
    "StartPolicy",
    "TaskManifest",
    "TaskType",
]
