"""Population simulator for the PWAV audit policy (pure Python, no database, no I/O).

It runs the *same* decision functions as the live system (`pwav.update`, `audit_probability`,
`decay_memory`) over a synthetic population, so the policy can be studied at a scale no demo can
reach: hundreds of rounds, many devices, sleeper agents, intermittent cheaters. What is simulated
is only the behaviour of devices (honest numerical noise, cheating); the audit/accusation logic is
the real code. Results are labelled as simulation in the UI and are never mixed with live data.
"""

import random
from dataclasses import dataclass, field
from typing import Any

from . import pwav

HONEST_EXCEED = 0.002  # honest devices exceed the (10x margin) tolerance very rarely


@dataclass
class SimDevice:
    id: int
    attacker: bool
    quarantined_at: int | None = None
    results: int = 0
    audits: int = 0
    n_clean: int = 0
    memory: float = 0.0
    state: dict[str, Any] = field(default_factory=pwav.new_state)
    corrupt_sent: int = 0
    corrupt_accepted: int = 0
    first_cheat_result: int | None = None


def simulate(
    *,
    honest: int,
    attackers: int,
    rounds: int,
    policy: str,
    alpha: float,
    q0: float,
    audit_floor: float,
    attack_strength: float,
    cheat_rate: float,
    sleeper_after: int,
    fixed_rate: float,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    ep = pwav.EParams(q0=q0, alpha=alpha)
    ap = pwav.AuditParams(floor=audit_floor)
    devices = [SimDevice(i, i >= honest) for i in range(honest + attackers)]
    series: dict[str, list[float]] = {
        k: []
        for k in (
            "audits",
            "corrupt_accepted",
            "corrupt_rejected",
            "attackers_quarantined",
            "honest_quarantined",
            "audit_rate",
            "attacker_audit_probability",
            "honest_audit_probability",
        )
    }
    audits = acc = rej = 0
    results_total = 0
    for rnd in range(rounds):
        round_audits = round_results = 0
        for d in devices:
            if d.quarantined_at is not None:
                continue
            if policy == "none":
                p = 0.0
            elif policy == "fixed":
                p = fixed_rate
            else:
                p = pwav.audit_probability(d.results, d.n_clean, d.memory, ap)
            cheats = d.attacker and d.results >= sleeper_after and rng.random() < cheat_rate
            d.results += 1
            results_total += 1
            round_results += 1
            if cheats:
                d.corrupt_sent += 1
                d.first_cheat_result = d.first_cheat_result or d.results
            if rng.random() < p:  # audited
                d.audits += 1
                audits += 1
                round_audits += 1
                detected = cheats and rng.random() < attack_strength
                z = 1 if detected or (not d.attacker and rng.random() < HONEST_EXCEED) else 0
                summ = pwav.update(d.state, z, ep)
                d.memory = pwav.decay_memory(d.memory, summ["suspicion"], ap)
                if not z:
                    d.n_clean += 1
                if z:
                    rej += 1 if cheats else 0
                elif cheats:  # audited but the corruption was below the tolerance
                    acc += 1
                    d.corrupt_accepted += 1
                if summ["accused"]:
                    d.quarantined_at = rnd
            elif cheats:  # not audited: the corrupt result gets merged
                acc += 1
                d.corrupt_accepted += 1
        att = [d for d in devices if d.attacker]
        hon = [d for d in devices if not d.attacker]

        def mean_p(group: list[SimDevice]) -> float:
            live = [d for d in group if d.quarantined_at is None]
            if not live:
                return 0.0
            if policy == "none":
                return 0.0
            if policy == "fixed":
                return fixed_rate
            return sum(
                pwav.audit_probability(d.results, d.n_clean, d.memory, ap) for d in live
            ) / len(live)

        series["audits"].append(audits)
        series["corrupt_accepted"].append(acc)
        series["corrupt_rejected"].append(rej)
        series["attackers_quarantined"].append(sum(d.quarantined_at is not None for d in att))
        series["honest_quarantined"].append(sum(d.quarantined_at is not None for d in hon))
        series["audit_rate"].append(round_audits / round_results if round_results else 0.0)
        series["attacker_audit_probability"].append(mean_p(att))
        series["honest_audit_probability"].append(mean_p(hon))

    att = [d for d in devices if d.attacker]
    caught = [d for d in att if d.quarantined_at is not None]
    delays = sorted(
        (d.quarantined_at + 1) - (d.first_cheat_result or 0)
        for d in caught
        if d.quarantined_at is not None
    )
    detections = [
        {
            "device": d.id,
            "quarantined_round": d.quarantined_at,
            "results_before": d.results,
            "corrupt_accepted": d.corrupt_accepted,
            "corrupt_sent": d.corrupt_sent,
        }
        for d in att
    ]
    corrupt_total = sum(d.corrupt_sent for d in att)
    summary = {
        "policy": policy,
        "results": results_total,
        "audits": audits,
        "audit_cost_fraction": audits / results_total if results_total else 0.0,
        "attackers": attackers,
        "attackers_caught": len(caught),
        "false_accusations": sum(
            1 for d in devices if not d.attacker and d.quarantined_at is not None
        ),
        "corrupt_sent": corrupt_total,
        "corrupt_merged": acc,
        "corrupt_rejected": rej,
        "corrupt_merged_fraction": acc / corrupt_total if corrupt_total else 0.0,
        "median_detection_delay_results": delays[len(delays) // 2] if delays else None,
        "max_detection_delay_results": delays[-1] if delays else None,
    }
    return {"series": series, "summary": summary, "detections": detections}


def simulate_all(**kw: Any) -> dict[str, Any]:
    """Run the requested policy and, for context, the two baselines on the same population."""
    main = simulate(**kw)
    comparison = {}
    for pol in ("none", "fixed", "adaptive"):
        comparison[pol] = (
            main["summary"] if pol == kw["policy"] else simulate(**{**kw, "policy": pol})["summary"]
        )
    main["summary"] = {**main["summary"], "comparison": comparison}
    return main
