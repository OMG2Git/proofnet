from fastapi import APIRouter, Request
from pymongo.errors import DuplicateKeyError

from ..contracts.api import LoginRequest, SignupRequest, TokenResponse, UserOut
from ..db import utcnow
from ..deps import DbDep, SettingsDep, UserDep
from ..errors import ProofNetError
from ..ids import new_id
from ..security.ratelimit import (
    check_lockout,
    client_ip,
    enforce_auth_limit,
    register_failure,
    register_success,
)
from .security import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


def _user_out(u: dict[str, object]) -> UserOut:
    return UserOut(
        id=str(u["_id"]),
        email=str(u["email"]),
        display_name=str(u["display_name"]),
        roles=list(u["roles"]),  # type: ignore[call-overload]
    )


@router.post("/signup", response_model=TokenResponse, status_code=201)
async def signup(
    body: SignupRequest, request: Request, db: DbDep, settings: SettingsDep
) -> TokenResponse:
    await enforce_auth_limit(request, db, settings, "signup")
    user_id = new_id("usr")
    roles = ["user", "contributor"]
    user = {
        "_id": user_id,
        "email": body.email.lower(),
        "password_hash": hash_password(body.password),
        "display_name": body.display_name,
        "roles": roles,
        "created_at": utcnow(),
    }
    try:
        await db.col("users").insert_one(user)
    except DuplicateKeyError:
        raise ProofNetError(
            409, "EMAIL_TAKEN", "An account with this email already exists"
        ) from None
    token = create_access_token(user_id, roles, settings.jwt_secret, settings.jwt_ttl_minutes)
    return TokenResponse(access_token=token)


@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginRequest, request: Request, db: DbDep, settings: SettingsDep
) -> TokenResponse:
    await enforce_auth_limit(request, db, settings, "login")
    email = body.email.lower()
    await check_lockout(db, email)
    user = await db.col("users").find_one({"email": email})
    if user is None or not verify_password(body.password, user["password_hash"]):
        await register_failure(db, settings, email, client_ip(request))
        raise ProofNetError(401, "INVALID_CREDENTIALS", "Wrong email or password")
    await register_success(db, email)
    token = create_access_token(
        user["_id"], user["roles"], settings.jwt_secret, settings.jwt_ttl_minutes
    )
    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserOut)
async def me(user: UserDep) -> UserOut:
    return _user_out(user)
