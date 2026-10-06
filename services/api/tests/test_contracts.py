import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from proofnet_api.contracts import HeartbeatResponse, TaskManifest
from proofnet_api.main import app


def _manifest(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "name": "t",
        "task_type": "gaussian_nb_train",
        "dataset_id": "ds_1",
        "params": {"target_column": "y", "feature_columns": ["a"]},
    }
    base.update(over)
    return base


def test_manifest_ok() -> None:
    m = TaskManifest.model_validate(_manifest())
    assert m.execution.max_devices == 4


def test_manifest_param_type_mismatch() -> None:
    with pytest.raises(ValidationError):
        TaskManifest.model_validate(
            _manifest(
                task_type="linear_ridge_train",
                params={"target_column": "y", "feature_columns": ["a"], "var_smoothing": 1e-9},
            )
        )


def test_manifest_device_bounds() -> None:
    with pytest.raises(ValidationError):
        TaskManifest.model_validate(_manifest(execution={"min_devices": 5, "max_devices": 2}))


def test_directive_discriminator() -> None:
    r = HeartbeatResponse.model_validate(
        {
            "server_time": "2026-01-01T00:00:00Z",
            "next_heartbeat_ms": 2000,
            "directives": [{"type": "cancel", "assignment_id": "asg_1"}],
        }
    )
    assert r.directives[0].type == "cancel"


def test_openapi_contains_contracts() -> None:
    schemas = TestClient(app).get("/api/v1/openapi.json").json()["components"]["schemas"]
    for name in ("TaskManifest", "HeartbeatResponse", "PartialResultEnvelope", "ApiError"):
        assert name in schemas
