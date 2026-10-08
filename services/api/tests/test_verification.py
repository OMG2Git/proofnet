"""Part 2 (P8-P11) on the live stack: audits by recomputation, attack detection, adaptive
sampling, forensics via the end-to-end reference check, quarantine, and the reward ledger.

CLI workers do real work and cheat in structurally valid ways (cli_worker.attacks); the API runs
against real MongoDB. Nothing here is mocked: wrong results are produced by real attack code and
rejected by real recomputation on the backend."""

import secrets
from collections.abc import Iterator
from typing import Any

import pytest
from api_helpers import signup
from e2e_helpers import (
    V1,
    Env,
    clf_csv,
    drive_many,
    make_workers,
    manifest,
    only_these,
    reg_csv,
    set_scores,
    status,
    upload,
)
from fastapi.testclient import TestClient
from pymongo import MongoClient

from cli_worker.worker import Faults, Worker
from proofnet_api.config import Settings
from proofnet_api.main import create_app
from proofnet_api.rewards import ledger

DONE = ("completed", "failed", "cancelled")
ALPHA = 0.2  # lenient lifetime budget so a persistent cheater is accused after ~6 audits


def _env(**over: Any) -> Iterator[Env]:
    s = Settings(
        mongodb_uri=Settings().mongodb_uri,
        mongodb_db=f"proofnet_test_ver_{secrets.token_hex(4)}",
        jwt_secret="test-secret-test-secret-test-secret-123",
        status_cache_seconds=0,
        reconciler_interval_seconds=1,
        exclusion_relax_seconds=300,
        **over,
    )
    mongo: MongoClient[dict[str, Any]] = MongoClient(s.mongodb_uri, tz_aware=True)
    try:
        with TestClient(create_app(s)) as c:
            yield c, mongo, s
    finally:
        mongo.drop_database(s.mongodb_db)
        mongo.close()


@pytest.fixture(scope="module")
def venv() -> Iterator[Env]:
    yield from _env(pwav_alpha=ALPHA)


@pytest.fixture(scope="module")
def noaudit_env() -> Iterator[Env]:
    """After probation nothing is audited (floor 0): isolates the end-to-end safety net."""
    yield from _env(pwav_alpha=ALPHA, audit_floor=0.0, audit_initial=0.0)


def col(env: Env, name: str) -> Any:
    _, mongo, s = env
    return mongo[s.mongodb_db][name]


def new_task(
    env: Env,
    h: dict[str, str],
    kind: str = "gaussian_nb_train",
    n: int = 3000,
    verification: str | None = None,
) -> str:
    client, _, _ = env
    if kind == "gaussian_nb_train":
        m = manifest(
            upload(client, h, clf_csv(n=n, seed=secrets.randbelow(1000))), kind, "label", 4
        )
    else:
        m = manifest(
            upload(client, h, reg_csv(n=n, seed=secrets.randbelow(1000))), kind, "target", 4
        )
    m["execution"] = {"min_devices": 1, "max_devices": 2}
    if verification:
        m["verification"] = verification
    r = client.post(f"{V1}/tasks", headers=h, json=m)
    assert r.status_code == 201, r.text
    return str(r.json()["id"])


def run_task(
    env: Env, h: dict[str, str], workers: list[Worker], tid: str, timeout: float = 60
) -> dict[str, Any]:
    client, _, _ = env
    drive_many(workers, lambda: status(client, h, tid)["task"]["status"] in DONE, timeout=timeout)
    return status(client, h, tid)


def prs(env: Env, tid: str) -> list[dict[str, Any]]:
    return list(col(env, "partial_results").find({"task_id": tid}))


def trust(env: Env, w: Worker) -> dict[str, Any]:
    p = col(env, "device_trust").find_one({"_id": w.device_id})
    assert p is not None
    return dict(p)


def is_quarantined(env: Env, w: Worker) -> bool:
    p = col(env, "device_trust").find_one({"_id": w.device_id})
    return bool(p and p["status"] == "quarantined")


# ---------------------------------------------------------------- honest baseline
def test_honest_devices_are_audited_verified_and_rewarded(venv: Env) -> None:
    client, _, _ = venv
    h = signup(client)
    ws = make_workers(client, "vh", 2)
    only_these(venv, ws)
    tid = new_task(venv, h)
    st = run_task(venv, h, ws, tid)
    assert st["task"]["status"] == "completed", st["task"]["error"]
    assert st["task"]["result"]["reference_check"]["passed"] is True
    rows = prs(venv, tid)
    assert len(rows) == 2 and {r["acceptance"] for r in rows} == {
        "verified"
    }  # probation = audit all
    recs = list(col(venv, "verification_records").find({"task_id": tid}))
    assert len(recs) == 2 and all(r["audited"] and not r["exceeded"] for r in recs)
    assert all(r["discrepancy"] <= r["tolerance"] for r in recs)
    assert all(r["audit_probability"] == 1.0 for r in recs)
    for w in ws:
        p = trust(venv, w)
        assert p["status"] == "probation" and p["audits"] == 1 and p["n_clean"] == 1
        assert p["suspicion"] == 0.0
    # rewards: verified work is confirmed immediately; an unproven device earns the base rate
    entries = list(col(venv, "reward_entries").find({"task_id": tid}))
    assert len(entries) == 2 and {e["status"] for e in entries} == {"confirmed"}
    assert all(0.49 <= e["multiplier"] <= 0.61 for e in entries)  # trust ~ 0 -> base 0.5


def test_report_states_how_much_was_verified(venv: Env) -> None:
    client, _, _ = venv
    h = signup(client)
    ws = make_workers(client, "vr", 2)
    only_these(venv, ws)
    tid = new_task(venv, h, kind="linear_ridge_train")
    st = run_task(venv, h, ws, tid)
    assert st["task"]["status"] == "completed"
    arts = client.get(f"{V1}/tasks/{tid}/artifacts", headers=h).json()
    rep_id = next(a["id"] for a in arts if a["kind"] == "report_json")
    report = client.get(f"{V1}/artifacts/{rep_id}/download", headers=h).json()
    assert report["verification"] == {
        "mode": "adaptive",
        "chunks": 2,
        "chunks_audited": 2,
        "chunks_unaudited": 0,
    }
    assert report["acceptance"].startswith("fully verified")


def test_verification_off_keeps_the_old_behaviour(venv: Env) -> None:
    client, _, _ = venv
    h = signup(client)
    ws = make_workers(client, "vo", 2)
    only_these(venv, ws)
    tid = new_task(venv, h, verification="off")
    st = run_task(venv, h, ws, tid)
    assert st["task"]["status"] == "completed"
    assert {r["acceptance"] for r in prs(venv, tid)} == {"accepted_unverified"}
    assert col(venv, "verification_records").count_documents({"task_id": tid}) == 0
    entries = list(col(venv, "reward_entries").find({"task_id": tid}))
    assert {e["status"] for e in entries} == {"confirmed"}  # pending -> confirmed at completion


# ---------------------------------------------------------------- every attack is caught
@pytest.mark.parametrize(
    "mode", ["subtle", "scale", "bias", "noise", "sign_flip", "zero", "random", "lazy"]
)
def test_attack_is_rejected_and_the_task_still_succeeds(venv: Env, mode: str) -> None:
    client, _, _ = venv
    h = signup(client)
    bad, good = make_workers(client, f"va{mode.replace('_', '')}", 2)
    only_these(venv, [bad, good])
    set_scores(venv, [bad, good], [3e6, 1e6])
    bad.faults = Faults(attack=mode)
    tid = new_task(venv, h)
    st = run_task(venv, h, [bad, good], tid)
    assert st["task"]["status"] == "completed", (mode, st["task"]["error"])
    assert st["task"]["result"]["reference_check"]["passed"] is True  # the merged model is right
    assert bad.attack_log.entries and all(e["attacked"] for e in bad.attack_log.entries)
    mine = list(col(venv, "assignments").find({"task_id": tid, "device_id": bad.device_id}))
    assert mine and all(a["status"] == "rejected" for a in mine), (
        mode,
        [a["status"] for a in mine],
    )
    assert all(a["error"]["code"] == "VERIFICATION_FAILED" for a in mine)
    accepted = [r for r in prs(venv, tid) if r["acceptance"] == "verified"]
    assert accepted and all(r["device_id"] == good.device_id for r in accepted)
    rec = col(venv, "verification_records").find_one(
        {"task_id": tid, "device_id": bad.device_id, "decision": "rejected_verification"}
    )
    assert rec is not None and rec["exceeded"] and rec["discrepancy"] > rec["tolerance"]
    # an attacker earns nothing for rejected work
    assert col(venv, "reward_entries").count_documents({"device_id": bad.device_id}) == 0
    assert trust(venv, bad)["suspicion"] > 0.0 and trust(venv, good)["suspicion"] == 0.0


def test_replay_attack_is_caught_once_it_has_something_to_replay(venv: Env) -> None:
    client, _, _ = venv
    h = signup(client)
    bad, good = make_workers(client, "vrp", 2)
    only_these(venv, [bad, good])
    set_scores(venv, [bad, good], [3e6, 1e6])
    bad.faults = Faults(attack="replay")
    first = run_task(venv, h, [bad, good], new_task(venv, h))
    assert first["task"]["status"] == "completed"
    assert {a["status"] for a in first["assignments"] if a["device_name"] == "vrp-0"} == {
        "succeeded"
    }
    second_id = new_task(venv, h)
    second = run_task(venv, h, [bad, good], second_id)
    assert second["task"]["status"] == "completed"
    assert second["task"]["result"]["reference_check"]["passed"] is True
    rejected = [a for a in second["assignments"] if a["status"] == "rejected"]
    assert len(rejected) == 1 and rejected[0]["device_name"] == "vrp-0"


# ---------------------------------------------------------------- quarantine
def test_persistent_cheater_is_quarantined_and_loses_access(venv: Env) -> None:
    client, _, _ = venv
    h = signup(client)
    bad, good1, good2 = make_workers(client, "vq", 3)
    only_these(venv, [bad, good1, good2])
    set_scores(venv, [bad, good1, good2], [3e6, 1e6, 1e6])
    bad.faults = Faults(attack="zero")
    tasks = 0
    while not is_quarantined(venv, bad) and tasks < 14:
        run_task(venv, h, [bad, good1, good2], new_task(venv, h, n=900))
        tasks += 1
    p = trust(venv, bad)
    assert p["status"] == "quarantined", (tasks, p["evidence"], p["exceedances"])
    assert p["exceedances"] >= 5 and p["quarantine"]["source"] == "pwav"
    # honest devices were never accused
    for g in (good1, good2):
        gp = col(venv, "device_trust").find_one({"_id": g.device_id})
        assert gp is None or (gp["status"] != "quarantined" and gp["exceedances"] == 0)
    dev = col(venv, "devices").find_one({"_id": bad.device_id})
    assert dev["quarantined"] is True
    ev = list(col(venv, "security_events").find({"kind": "device_quarantined"}))
    assert any(e["device_id"] == bad.device_id and e["severity"] == "critical" for e in ev)
    # no further work: the next task is computed by honest devices only
    before = col(venv, "assignments").count_documents({"device_id": bad.device_id})
    st = run_task(venv, h, [bad, good1, good2], new_task(venv, h, n=900))
    assert st["task"]["status"] == "completed"
    assert col(venv, "assignments").count_documents({"device_id": bad.device_id}) == before
    # the owner of the quarantined device keeps nothing from unverified work (there was none)
    assert (
        col(venv, "reward_entries").count_documents(
            {"device_id": bad.device_id, "status": {"$ne": "revoked"}}
        )
        == 0
    )


# ---------------------------------------------------------------- the safety net
def test_unaudited_cheat_is_found_by_the_reference_check_and_healed(noaudit_env: Env) -> None:
    """A trusted device that is (by sampling) not audited cheats. The end-to-end reference check of
    the finished task fails, forensics recomputes every chunk, names the culprit, quarantines it,
    revokes its reward and re-runs the chunk: the final model is correct."""
    env = noaudit_env
    client, _, _ = env
    h = signup(client)
    bad, good = make_workers(client, "vf", 2)
    only_these(env, [bad, good])
    set_scores(env, [bad, good], [3e6, 1e6])
    for w in (bad, good):  # looks like a long-time trusted device -> floor 0 -> never audited
        col(env, "device_trust").update_one(
            {"_id": w.device_id},
            {
                "$set": {
                    "user_id": "u",
                    "status": "trusted",
                    "results_seen": 500,
                    "audits": 400,
                    "n_clean": 400,
                    "exceedances": 0,
                    "rejected": 0,
                    "e_state": {"n": 0, "exceed": 0, "R": []},
                    "evidence": 0.0,
                    "suspicion": 0.0,
                    "memory": 0.0,
                    "history": [],
                }
            },
            upsert=True,
        )
    bad.faults = Faults(attack="scale")
    tid = new_task(env, h)
    st = run_task(env, h, [bad, good], tid)
    assert st["task"]["status"] == "completed", st["task"]["error"]
    assert st["task"]["result"]["reference_check"]["passed"] is True  # healed
    assert trust(env, bad)["status"] == "quarantined"
    assert trust(env, bad)["quarantine"]["source"] == "forensics"
    rows = prs(env, tid)
    assert {r["acceptance"] for r in rows if r["device_id"] == bad.device_id} == {
        "rejected_forensic"
    }
    assert all(
        r["device_id"] == good.device_id for r in rows if r["acceptance"] != "rejected_forensic"
    )
    kinds = {e["kind"] for e in col(env, "security_events").find({"task_id": tid})}
    assert "reference_check_failed" in kinds and "device_quarantined" in kinds
    ent = list(col(env, "reward_entries").find({"device_id": bad.device_id}))
    assert ent and all(e["status"] == "revoked" for e in ent)  # clawback
    good_ent = list(col(env, "reward_entries").find({"device_id": good.device_id}))
    assert good_ent and all(e["status"] == "confirmed" for e in good_ent)


def test_adaptive_sampling_skips_some_audits_for_trusted_devices(venv: Env) -> None:
    client, _, s = venv
    h = signup(client)
    ws = make_workers(client, "vs", 2)
    only_these(venv, ws)
    for w in ws:
        col(venv, "device_trust").update_one(
            {"_id": w.device_id},
            {
                "$set": {
                    "user_id": "u",
                    "status": "trusted",
                    "results_seen": 2000,
                    "audits": 1000,
                    "n_clean": 1000,
                    "exceedances": 0,
                    "rejected": 0,
                    "e_state": {"n": 0, "exceed": 0, "R": []},
                    "evidence": 0.0,
                    "suspicion": 0.0,
                    "memory": 0.0,
                    "history": [],
                }
            },
            upsert=True,
        )
    audited = skipped = 0
    for _ in range(8):
        tid = new_task(venv, h)
        assert run_task(venv, h, ws, tid)["task"]["status"] == "completed"
        for r in col(venv, "verification_records").find({"task_id": tid}):
            audited += bool(r["audited"])
            skipped += not r["audited"]
            assert r["audit_probability"] < 0.12  # trusted: close to the floor ...
            assert r["audit_probability"] >= s.audit_floor  # ... but never below it
    assert skipped >= 6 and audited + skipped == 16  # most results are not re-computed
    # skipped results are accepted_unverified, and their rewards were pending until completion
    unv = list(col(venv, "reward_entries").find({"acceptance": "accepted_unverified"}))
    assert unv and all(e["status"] == "confirmed" for e in unv)


# ---------------------------------------------------------------- the ledger
def test_ledger_replay_equals_the_balances(venv: Env) -> None:
    ev = list(col(venv, "reward_events").find({}))
    assert ev, "earlier tests must have produced ledger events"
    replayed = ledger.replay(ev)
    agg: dict[str, dict[str, float]] = {}
    for e in col(venv, "reward_entries").find({}):
        b = agg.setdefault(e["user_id"], {"pending": 0.0, "confirmed": 0.0, "revoked": 0.0})
        b[e["status"]] += e["amount"]
    for u, b in agg.items():
        for k in b:
            assert replayed[u][k] == pytest.approx(b[k], abs=1e-5), (u, k)
    assert sum(1 for e in ev if e["kind"] == "accrue") == col(
        venv, "reward_entries"
    ).count_documents({})


# ---------------------------------------------------------------- CNN training under attack
def test_cnn_training_survives_a_gradient_poisoning_device(venv: Env) -> None:
    """A device flips the sign of its gradient in every round. Each poisoned slice is rejected and
    recomputed by an honest device; the attacker is quarantined mid-training; the model still
    trains (loss falls) and the sampled centralized gradient checks pass."""
    from test_cnn_training import image_manifest, run_training, upload_images

    client, _, _ = venv
    h = signup(client)
    bad, g1, g2 = make_workers(client, "vcnn", 3)
    only_these(venv, [bad, g1, g2])
    set_scores(venv, [bad, g1, g2], [3e6, 1e6, 1e6])
    bad.faults = Faults(attack="sign_flip")
    ds = upload_images(client, h)
    m = image_manifest(ds["id"], execution={"min_devices": 2, "max_devices": 3})
    st = run_training(venv, [bad, g1, g2], h, m)
    assert st["task"]["status"] == "completed", st["task"]["error"]
    assert st["task"]["result"]["reference_check"]["passed"] is True
    tr = client.get(f"{V1}/tasks/{st['task']['id']}/training", headers=h).json()
    assert tr["round"] == 14 and len(tr["rounds"]) == 14
    assert tr["rounds"][-1]["loss"] < tr["rounds"][0]["loss"]  # learning happened
    prof = trust(venv, bad)
    assert prof["status"] == "quarantined" and prof["exceedances"] >= 5
    rejected = col(venv, "assignments").count_documents(
        {"device_id": bad.device_id, "status": "rejected"}
    )
    assert rejected == prof["exceedances"]
    # no poisoned gradient was ever merged
    merged = col(venv, "partial_results").count_documents(
        {"device_id": bad.device_id, "acceptance": {"$in": ["verified", "accepted_unverified"]}}
    )
    assert merged == 0
