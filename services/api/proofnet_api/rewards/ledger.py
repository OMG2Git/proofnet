"""Reward ledger (ARCHITECTURE 17.5): trust-weighted, verification-gated, revocable.

Credits are an internal accounting unit (no money, no payments). One *entry* per accepted
assignment, one append-only *event* per entry transition, so every balance can be replayed:

  accrue   entry created. Audited-and-correct work is `confirmed` at once; work that was not
           audited stays `pending` until its task completes (the final result is checked
           end-to-end) and is then confirmed.
  revoke   a quarantined device forfeits everything that was never individually verified.

amount = work_units / 1e6 x reward_rate x reward_multiplier(trust). A device that has never been
audited earns only `reward_base` (default 50 %) of the full rate; full trust earns 100 %.
All transitions are conditional atomic updates, so a repeat can never double-credit or double-revoke.
"""

import time
from collections.abc import Iterable
from typing import Any

from pymongo.errors import DuplicateKeyError

from ..config import Settings
from ..db import Db, utcnow
from ..ids import new_id
from ..trust import pwav, store
from ..verification import ACCEPTED_UNVERIFIED, VERIFIED

PENDING, CONFIRMED, REVOKED = "pending", "confirmed", "revoked"
UNITS_PER_CREDIT = 1_000_000.0


_last_seq = 0


def next_seq() -> int:
    """Strictly increasing event number (MongoDB timestamps only have millisecond resolution)."""
    global _last_seq
    _last_seq = max(_last_seq + 1, time.time_ns())
    return _last_seq


def round_amount(x: float) -> float:
    return round(x, 6)


async def _event(
    db: Db, entry: dict[str, Any], kind: str, amount: float, reason: str | None = None
) -> None:
    await db.col("reward_events").insert_one(
        {
            "_id": new_id("rwe"),
            "at": utcnow(),
            "seq": next_seq(),
            "kind": kind,
            "entry_id": entry["_id"],
            "user_id": entry["user_id"],
            "device_id": entry["device_id"],
            "task_id": entry["task_id"],
            "amount": amount,
            "reason": reason,
        }
    )


async def accrue(
    db: Db,
    settings: Settings,
    *,
    task: dict[str, Any],
    chunk: dict[str, Any],
    assignment_id: str,
    device: dict[str, Any],
    acceptance: str,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Create the reward entry for one accepted assignment (idempotent per assignment)."""
    profile = profile or await db.col("device_trust").find_one({"_id": device["_id"]})
    trust = store.trust_of(profile) if profile else 0.0
    mult = pwav.reward_multiplier(trust, settings.reward_base)
    base = chunk["work_units"] / UNITS_PER_CREDIT * settings.reward_rate
    amount = round_amount(base * mult)
    verified = acceptance == VERIFIED
    entry = {
        "_id": assignment_id,
        "user_id": device["owner_user_id"],
        "device_id": device["_id"],
        "task_id": task["_id"],
        "chunk_id": chunk["_id"],
        "work_units": chunk["work_units"],
        "base_amount": round_amount(base),
        "trust": round(trust, 4),
        "multiplier": round(mult, 4),
        "amount": amount,
        "acceptance": acceptance,
        "status": CONFIRMED if verified else PENDING,
        "reason": None,
        "created_at": utcnow(),
        "updated_at": utcnow(),
    }
    try:
        await db.col("reward_entries").insert_one(entry)
    except DuplicateKeyError:
        return None
    await _event(db, entry, "accrue", amount, acceptance)
    if verified:
        await _event(db, entry, "confirm", amount, "audited and correct")
    return entry


async def confirm_task(db: Db, task_id: str, reason: str = "task completed") -> int:
    """Pending entries of a finished task become confirmed (conditional per entry)."""
    n = 0
    async for e in db.col("reward_entries").find({"task_id": task_id, "status": PENDING}):
        res = await db.col("reward_entries").update_one(
            {"_id": e["_id"], "status": PENDING},
            {"$set": {"status": CONFIRMED, "updated_at": utcnow(), "reason": reason}},
        )
        if res.modified_count:
            await _event(db, e, "confirm", e["amount"], reason)
            n += 1
    return n


async def revoke_unverified(db: Db, device_id: str, reason: str) -> dict[str, Any]:
    """Clawback: every entry of this device that was never individually verified is revoked."""
    total, n = 0.0, 0
    async for e in db.col("reward_entries").find(
        {"device_id": device_id, "acceptance": ACCEPTED_UNVERIFIED, "status": {"$ne": REVOKED}}
    ):
        res = await db.col("reward_entries").update_one(
            {"_id": e["_id"], "status": {"$in": [PENDING, CONFIRMED]}},
            {"$set": {"status": REVOKED, "updated_at": utcnow(), "reason": reason}},
        )
        if res.modified_count:
            await _event(db, e, "revoke", e["amount"], reason)
            total += e["amount"]
            n += 1
    return {"entries": n, "amount": round_amount(total)}


async def revoke_entry(db: Db, assignment_id: str, reason: str) -> bool:
    e = await db.col("reward_entries").find_one({"_id": assignment_id})
    if e is None:
        return False
    res = await db.col("reward_entries").update_one(
        {"_id": assignment_id, "status": {"$in": [PENDING, CONFIRMED]}},
        {"$set": {"status": REVOKED, "updated_at": utcnow(), "reason": reason}},
    )
    if res.modified_count:
        await _event(db, e, "revoke", e["amount"], reason)
    return bool(res.modified_count)


# ------------------------------------------------------------------ balances
def replay(events: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Rebuild per-user balances from the event log only (the audit check on the ledger):
    accrue adds to pending; confirm moves pending -> confirmed; revoke removes from whichever
    bucket the entry is in and adds to revoked."""
    state: dict[str, str] = {}  # entry -> bucket
    out: dict[str, dict[str, float]] = {}
    for ev in sorted(events, key=lambda e: e["seq"]):
        b = out.setdefault(ev["user_id"], {PENDING: 0.0, CONFIRMED: 0.0, REVOKED: 0.0})
        amt, eid = ev["amount"], ev["entry_id"]
        if ev["kind"] == "accrue":
            b[PENDING] += amt
            state[eid] = PENDING
        elif ev["kind"] == "confirm" and state.get(eid) == PENDING:
            b[PENDING] -= amt
            b[CONFIRMED] += amt
            state[eid] = CONFIRMED
        elif ev["kind"] == "revoke" and state.get(eid) in (PENDING, CONFIRMED):
            b[state[eid]] -= amt
            b[REVOKED] += amt
            state[eid] = REVOKED
    return {u: {k: round_amount(v) for k, v in b.items()} for u, b in out.items()}


async def balances(db: Db, user_id: str | None = None) -> dict[str, dict[str, float]]:
    """Per-user balances from the current entries."""
    match: dict[str, Any] = {"user_id": user_id} if user_id else {}
    pipeline = [
        {"$match": match},
        {"$group": {"_id": {"u": "$user_id", "s": "$status"}, "amount": {"$sum": "$amount"}}},
    ]
    out: dict[str, dict[str, float]] = {}
    async for r in await db.col("reward_entries").aggregate(pipeline):
        b = out.setdefault(r["_id"]["u"], {PENDING: 0.0, CONFIRMED: 0.0, REVOKED: 0.0})
        b[r["_id"]["s"]] = round_amount(r["amount"])
    return out
