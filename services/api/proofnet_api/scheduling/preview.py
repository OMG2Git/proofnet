"""Plan preview (wizard) and waiting reasons (monitor): the same planner/eligibility rules the
scheduler uses, evaluated read-only against the current device pool."""

from typing import Any

from ..config import Settings
from ..db import Db, utcnow
from .planner import MIN_CHUNK_ROWS
from .policies import ineligibility_reasons
from .scheduler import PLANNER, plan_document, planning_devices


async def device_reasons(
    db: Db, settings: Settings, task_like: dict[str, Any], rows: int, n_features: int
) -> list[str]:
    """Per-device explanations of why a device cannot take work right now."""
    now = utcnow()
    out: list[str] = []
    async for d in db.col("devices").find({"status": {"$ne": "disabled"}}):
        why = ineligibility_reasons(
            d, now=now, settings=settings, rows=rows, n_features=n_features, task=task_like
        )
        if why:
            out.append(f"{d['name']}: " + "; ".join(why))
    return out


def estimate_train_rows(profile: dict[str, Any], params: dict[str, Any]) -> int:
    """Training rows the task will have, from the dataset profile (an estimate: the exact value
    is known after preparation)."""
    cols = {c["name"]: c for c in profile["columns"]}
    used = [params["target_column"], *params["feature_columns"]]
    missing = max((cols[c]["missing"] for c in used if c in cols), default=0)
    n = int(profile["n_rows"]) - int(missing)
    n_test = max(1, round(n * params["test_fraction"]))
    return int(max(0, n - n_test))


async def build_plan_preview(
    db: Db,
    settings: Settings,
    profile: dict[str, Any],
    params: dict[str, Any],
    execution: dict[str, Any],
) -> dict[str, Any]:
    n_train = estimate_train_rows(profile, params)
    n_features = len(params["feature_columns"])
    task_like = {"prepared": {"n_features": n_features}, "execution": execution}
    devices = await planning_devices(db, settings, task_like)
    ready = len(devices) >= execution["min_devices"]
    preview: dict[str, Any] = {
        "estimated": True,
        "note": "Based on the dataset profile and the devices online right now; the final plan is "
        "made when the task starts.",
        "n_train": n_train,
        "eligible_devices": len(devices),
        "min_devices": execution["min_devices"],
        "ready_to_start": ready,
        "min_chunk_rows": MIN_CHUNK_ROWS,
        "shares": [],
        "chunks": [],
        "not_eligible": await device_reasons(db, settings, task_like, MIN_CHUNK_ROWS, n_features),
    }
    if not ready:
        preview["message"] = (
            f"needs {execution['min_devices']} eligible device(s); {len(devices)} available now"
        )
    if devices and n_train > 0:
        plan = PLANNER.plan(n_train, n_features, devices, execution["max_devices"])
        doc = plan_document(plan, devices, n_train)
        preview["shares"] = doc["shares"]
        preview["chunks"] = doc["chunks"]
        preview["explanation"] = doc["explanation"]
    return preview
