"""Error format (ARCHITECTURE 13)."""

from typing import Any

from pydantic import BaseModel


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[Any] = []


class ApiError(BaseModel):
    error: ErrorBody
