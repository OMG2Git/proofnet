"""Abuse controls that need no external service: a sliding-window rate limiter (in memory, per
backend instance) and persistent login lockout. Both write to the security event log."""

import time
from collections import defaultdict, deque
from datetime import timedelta

from fastapi import Request

from ..config import Settings
from ..db import Db, utcnow
from ..errors import ProofNetError
from .quarantine import record_event


class RateLimiter:
    """Sliding-window counter per key. Memory is bounded: idle keys are dropped."""

    def __init__(self) -> None:
        self.hits: dict[str, deque[float]] = defaultdict(deque)
        self.last_logged: dict[str, float] = {}

    def check(self, key: str, limit: int, window: float = 60.0, now: float | None = None) -> float:
        """Returns 0.0 if allowed (and counts the hit), else the seconds until a slot frees up."""
        now = time.monotonic() if now is None else now
        q = self.hits[key]
        while q and q[0] <= now - window:
            q.popleft()
        if len(q) >= limit:
            return max(q[0] + window - now, 0.1)
        q.append(now)
        if len(self.hits) > 20000:  # bounded memory under a key-spraying attack
            for k in [k for k, v in self.hits.items() if not v or v[-1] <= now - window][:5000]:
                self.hits.pop(k, None)
        return 0.0


def client_ip(request: Request) -> str:
    """The address the platform's load balancer saw: the LAST X-Forwarded-For entry (earlier ones
    are client-supplied and can be forged)."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


async def enforce_auth_limit(request: Request, db: Db, settings: Settings, scope: str) -> None:
    limiter: RateLimiter = request.app.state.limiter
    ip = client_ip(request)
    wait = limiter.check(f"{scope}:{ip}", settings.rate_limit_auth_per_minute)
    if not wait:
        return
    now = time.monotonic()
    if now - limiter.last_logged.get(f"{scope}:{ip}", -1e9) > 60:
        limiter.last_logged[f"{scope}:{ip}"] = now
        await record_event(db, "rate_limited", severity="warning", data={"scope": scope, "ip": ip})
    raise ProofNetError(429, "RATE_LIMITED", f"Too many requests; retry in {int(wait) + 1} s")


# ------------------------------------------------------------------ login lockout
async def check_lockout(db: Db, email: str) -> None:
    rec = await db.col("login_attempts").find_one({"_id": email})
    if rec and rec.get("locked_until") and rec["locked_until"] > utcnow():
        left = int((rec["locked_until"] - utcnow()).total_seconds()) + 1
        raise ProofNetError(
            429, "ACCOUNT_LOCKED", f"Too many failed sign-ins; try again in {left} s"
        )


async def register_failure(db: Db, settings: Settings, email: str, ip: str) -> None:
    now = utcnow()
    rec = await db.col("login_attempts").find_one_and_update(
        {"_id": email},
        {"$inc": {"fails": 1}, "$set": {"last_fail_at": now, "ip": ip}},
        upsert=True,
        return_document=True,
    )
    if rec is not None and rec["fails"] >= settings.login_max_failures:
        await db.col("login_attempts").update_one(
            {"_id": email},
            {
                "$set": {
                    "locked_until": now + timedelta(seconds=settings.login_lockout_seconds),
                    "fails": 0,
                }
            },
        )
        await record_event(
            db,
            "login_lockout",
            severity="warning",
            data={"ip": ip, "seconds": settings.login_lockout_seconds},
        )


async def register_success(db: Db, email: str) -> None:
    await db.col("login_attempts").delete_one({"_id": email})
