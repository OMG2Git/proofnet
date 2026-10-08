"""Prefixed readable IDs (ARCHITECTURE 12). pr_ = partial result; ses_ = worker session."""

import secrets
import time

PREFIXES = {
    "usr",
    "dev",
    "ds",
    "img",
    "tsk",
    "chk",
    "asg",
    "pr",
    "art",
    "evt",
    "ses",
    "vr",
    "rwe",
    "sec",
}
_ALPHABET = "0123456789abcdefghjkmnpqrstvwxyz"  # Crockford base32 (lowercase)


def _b32(n: int, width: int) -> str:
    out = []
    for _ in range(width):
        out.append(_ALPHABET[n & 31])
        n >>= 5
    return "".join(reversed(out))


def new_id(prefix: str) -> str:
    """Time-ordered (ms timestamp) + 10 random chars, e.g. 'tsk_01h8x...'."""
    if prefix not in PREFIXES:
        raise ValueError(f"unknown id prefix '{prefix}'")
    ts = _b32(int(time.time() * 1000), 9)
    rand = _b32(secrets.randbits(50), 10)
    return f"{prefix}_{ts}{rand}"


def new_token() -> str:
    """Opaque secret for device tokens (shown once; only its hash is stored)."""
    return secrets.token_urlsafe(32)
