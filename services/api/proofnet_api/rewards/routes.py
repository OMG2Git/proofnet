"""Rewards API: my balance and entries, the network table, and the ledger integrity check."""

from typing import Any

from fastapi import APIRouter

from ..admin.routes import is_admin
from ..contracts.part2 import Balance, LedgerCheck, NetworkRewardRow, RewardEntry, RewardsOut
from ..deps import DbDep, SettingsDep, UserDep
from . import ledger

router = APIRouter(prefix="/rewards", tags=["rewards"])


def _bal(d: dict[str, float] | None) -> Balance:
    d = d or {}
    return Balance(
        pending=d.get("pending", 0.0),
        confirmed=d.get("confirmed", 0.0),
        revoked=d.get("revoked", 0.0),
    )


@router.get("/me", response_model=RewardsOut)
async def my_rewards(db: DbDep, user: UserDep, settings: SettingsDep) -> RewardsOut:
    devices = {
        d["_id"]: d["name"]
        async for d in db.col("devices").find({"owner_user_id": user["_id"]}, {"name": 1})
    }
    per_device: dict[str, Balance] = {}
    pipeline: list[dict[str, Any]] = [
        {"$match": {"user_id": user["_id"]}},
        {"$group": {"_id": {"d": "$device_id", "s": "$status"}, "amount": {"$sum": "$amount"}}},
    ]
    async for r in await db.col("reward_entries").aggregate(pipeline):
        b = per_device.setdefault(r["_id"]["d"], Balance())
        setattr(b, r["_id"]["s"], ledger.round_amount(r["amount"]))
    total = _bal((await ledger.balances(db, user["_id"])).get(user["_id"]))
    entries = [
        RewardEntry(
            id=e["_id"],
            task_id=e["task_id"],
            device_id=e["device_id"],
            device_name=devices.get(e["device_id"]),
            work_units=e["work_units"],
            base_amount=e["base_amount"],
            trust=e["trust"],
            multiplier=e["multiplier"],
            amount=e["amount"],
            acceptance=e["acceptance"],
            status=e["status"],
            reason=e.get("reason"),
            created_at=e["created_at"],
        )
        async for e in db.col("reward_entries")
        .find({"user_id": user["_id"]})
        .sort("created_at", -1)
        .limit(60)
    ]
    return RewardsOut(
        balance=total,
        per_device=per_device,
        device_names=devices,
        entries=entries,
        rate_credits_per_million_units=settings.reward_rate,
        base_multiplier=settings.reward_base,
    )


@router.get("/network", response_model=list[NetworkRewardRow])
async def network_rewards(db: DbDep, user: UserDep) -> list[NetworkRewardRow]:
    """Everyone's balance by display name (a leaderboard for the demo; no e-mails)."""
    bal = await ledger.balances(db)
    users = {
        u["_id"]: u
        async for u in db.col("users").find({"_id": {"$in": list(bal)}}, {"display_name": 1})
    }
    counts: dict[str, int] = {}
    async for d in db.col("devices").find(
        {"owner_user_id": {"$in": list(bal)}}, {"owner_user_id": 1}
    ):
        counts[d["owner_user_id"]] = counts.get(d["owner_user_id"], 0) + 1
    rows = [
        NetworkRewardRow(
            display_name=users.get(u, {}).get("display_name", "unknown"),
            mine=u == user["_id"],
            devices=counts.get(u, 0),
            balance=_bal(b),
        )
        for u, b in bal.items()
    ]
    return sorted(rows, key=lambda r: -r.balance.confirmed)


@router.get("/ledger/check", response_model=LedgerCheck)
async def ledger_check(db: DbDep, user: UserDep, settings: SettingsDep) -> LedgerCheck:
    """Replay the append-only event log and compare it with the current entries. Admins check
    every account, others only their own."""
    admin = is_admin(user, settings)
    q: dict[str, Any] = {} if admin else {"user_id": user["_id"]}
    events = [e async for e in db.col("reward_events").find(q)]
    replayed = ledger.replay(events)
    current = await ledger.balances(db, None if admin else user["_id"])
    mismatches: list[str] = []
    for u in set(replayed) | set(current):
        a, b = replayed.get(u, {}), current.get(u, {})
        for k in ("pending", "confirmed", "revoked"):
            if abs(a.get(k, 0.0) - b.get(k, 0.0)) > 1e-5:
                mismatches.append(f"{u}.{k}: events {a.get(k, 0.0)} vs entries {b.get(k, 0.0)}")
    return LedgerCheck(
        consistent=not mismatches,
        events=len(events),
        entries=await db.col("reward_entries").count_documents(q),
        users_checked=len(set(replayed) | set(current)),
        mismatches=mismatches[:20],
        replayed={u: _bal(b) for u, b in replayed.items()},
    )
