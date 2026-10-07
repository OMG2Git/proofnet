"""Task manifest (ARCHITECTURE 4.4, validation bounds 5.1)."""

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from proofnet_kernels.server.cnn import CnnParams
from proofnet_kernels.server.params import (
    CommonParams,
    GaussianNBParams,
    LinearRidgeParams,
    MissingValuePolicy,
)

__all__ = ["GaussianNBParams", "LinearRidgeParams", "MissingValuePolicy"]


class TaskType(StrEnum):
    gaussian_nb_train = "gaussian_nb_train"
    linear_ridge_train = "linear_ridge_train"


class StartPolicy(StrEnum):
    wait_for_min_devices = "wait_for_min_devices"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExecutionSettings(_Strict):
    min_devices: int = Field(default=1, ge=1, le=8)
    max_devices: int = Field(default=4, ge=1, le=8)
    start_policy: StartPolicy = StartPolicy.wait_for_min_devices

    @model_validator(mode="after")
    def _ordered(self) -> "ExecutionSettings":
        if self.min_devices > self.max_devices:
            raise ValueError("min_devices must be <= max_devices")
        return self


class ImageTaskManifest(_Strict):
    """Image CNN training (separate pipeline from the CSV workloads; ARCHITECTURE 4.5)."""

    name: str = Field(min_length=1, max_length=120)
    task_type: Literal["cnn_image_train"] = "cnn_image_train"
    kernel_version: Literal["1"] = "1"
    dataset_id: str = Field(pattern=r"^img_")
    params: CnnParams = CnnParams()
    execution: ExecutionSettings = ExecutionSettings()


class TaskManifest(_Strict):
    name: str = Field(min_length=1, max_length=120)
    task_type: TaskType
    kernel_version: Literal["1"] = "1"
    dataset_id: str = Field(pattern=r"^ds_")
    params: GaussianNBParams | LinearRidgeParams
    execution: ExecutionSettings = ExecutionSettings()

    @model_validator(mode="before")
    @classmethod
    def _select_params_model(cls, data: Any) -> Any:
        """Pick the params model from task_type so shared-only params are not ambiguous."""
        if isinstance(data, dict) and isinstance(data.get("params"), dict):
            models: dict[str, type[CommonParams]] = {
                "gaussian_nb_train": GaussianNBParams,
                "linear_ridge_train": LinearRidgeParams,
            }
            model = models.get(str(data.get("task_type")))
            if model is not None:
                data = {**data, "params": model.model_validate(data["params"])}
        return data

    @model_validator(mode="after")
    def _params_match_type(self) -> "TaskManifest":
        expected = (
            GaussianNBParams if self.task_type is TaskType.gaussian_nb_train else LinearRidgeParams
        )
        if not isinstance(self.params, expected):
            raise ValueError(f"params do not match task_type {self.task_type}")
        return self
