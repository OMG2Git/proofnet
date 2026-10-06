"""Typed kernel parameters (single source; the API manifest reuses these models)."""

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class MissingValuePolicy(StrEnum):
    drop_rows = "drop_rows"
    reject = "reject"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CommonParams(_Strict):
    target_column: str = Field(min_length=1)
    feature_columns: Annotated[list[str], Field(min_length=1, max_length=64)]
    missing_values: MissingValuePolicy = MissingValuePolicy.drop_rows
    test_fraction: float = Field(default=0.2, ge=0.05, le=0.5)
    split_seed: int = 42


class GaussianNBParams(CommonParams):
    var_smoothing: float = Field(default=1e-9, ge=0)


class LinearRidgeParams(CommonParams):
    alpha: float = Field(default=1.0, ge=0)
