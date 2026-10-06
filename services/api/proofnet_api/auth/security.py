"""Password hashing (argon2id), user JWTs (HS256) and device-token hashing (SHA-256).

Device tokens are high-entropy random strings, so a fast SHA-256 hash is appropriate
(ARCHITECTURE 3.5: raw token on the device, SHA-256 in the DB).
"""

import hashlib
import hmac
import time
from typing import Any

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from ..errors import ProofNetError

_ph = PasswordHasher()
_ALG = "HS256"


def hash_password(password: str) -> str:
    return _ph.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _ph.verify(password_hash, password)
    except (VerificationError, InvalidHashError):
        return False


def create_access_token(
    user_id: str, roles: list[str], secret: str, ttl_minutes: int, now: float | None = None
) -> str:
    iat = int(now if now is not None else time.time())
    claims = {"sub": user_id, "roles": roles, "iat": iat, "exp": iat + ttl_minutes * 60}
    return jwt.encode(claims, secret, algorithm=_ALG)


def decode_access_token(token: str, secret: str) -> dict[str, Any]:
    try:
        claims: dict[str, Any] = jwt.decode(
            token, secret, algorithms=[_ALG], options={"require": ["sub", "exp"]}
        )
    except jwt.PyJWTError:
        raise ProofNetError(401, "UNAUTHORIZED", "Invalid or expired token") from None
    return claims


def hash_device_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def verify_device_token(token: str, token_hash: str) -> bool:
    return hmac.compare_digest(hash_device_token(token), token_hash)
