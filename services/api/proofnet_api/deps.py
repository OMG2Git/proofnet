"""FastAPI dependencies: settings, DB handle, user and device authentication."""

from typing import Annotated, Any

from fastapi import Depends, Request

from .auth.security import decode_access_token, hash_device_token
from .config import Settings
from .db import Db
from .errors import ProofNetError


def get_db(request: Request) -> Db:
    db: Db = request.app.state.db
    return db


def settings_dep(request: Request) -> Settings:
    s: Settings = request.app.state.settings
    return s


DbDep = Annotated[Db, Depends(get_db)]
SettingsDep = Annotated[Settings, Depends(settings_dep)]


def _bearer(request: Request) -> str:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise ProofNetError(401, "UNAUTHORIZED", "Missing bearer token")
    return token


async def current_user(request: Request, db: DbDep, settings: SettingsDep) -> dict[str, Any]:
    claims = decode_access_token(_bearer(request), settings.jwt_secret)
    user = await db.col("users").find_one({"_id": claims["sub"]})
    if user is None:
        raise ProofNetError(401, "UNAUTHORIZED", "Unknown user")
    return user


async def current_device(request: Request, db: DbDep) -> dict[str, Any]:
    device = await db.col("devices").find_one({"token_hash": hash_device_token(_bearer(request))})
    if device is None:
        raise ProofNetError(401, "UNAUTHORIZED", "Invalid device token")
    return device


UserDep = Annotated[dict[str, Any], Depends(current_user)]
DeviceDep = Annotated[dict[str, Any], Depends(current_device)]
