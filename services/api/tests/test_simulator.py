"""The population simulator runs the real PWAV decision functions over synthetic devices.
These tests check the claims made in the demo: cost vs safety, sleepers, intermittent cheaters,
lifetime false-accusation control, determinism."""

from typing import Any

from proofnet_api.trust.simulator import simulate, simulate_all

BASE: dict[str, Any] = {
    "honest": 30,
    "attackers": 6,
    "rounds": 400,
    "policy": "adaptive",
    "alpha": 1e-3,
    "q0": 0.03,
    "audit_floor": 0.05,
    "attack_strength": 1.0,
    "cheat_rate": 1.0,
    "sleeper_after": 0,
    "fixed_rate": 0.3,
    "seed": 11,
}


def run(**over: Any) -> dict[str, Any]:
    return simulate(**{**BASE, **over})["summary"]


def test_adaptive_catches_every_always_cheating_device_within_a_few_audits() -> None:
    s = run()
    assert s["attackers_caught"] == 6 and s["false_accusations"] == 0
    assert s["max_detection_delay_results"] <= 40
    assert s["corrupt_merged_fraction"] < 0.15


def test_adaptive_costs_far_less_than_a_fixed_rate_with_similar_safety() -> None:
    both = simulate_all(**BASE)["summary"]["comparison"]
    assert both["adaptive"]["audit_cost_fraction"] < 0.5 * both["fixed"]["audit_cost_fraction"]
    assert both["adaptive"]["attackers_caught"] == both["fixed"]["attackers_caught"] == 6
    assert both["none"]["corrupt_merged_fraction"] == 1.0
    assert both["none"]["attackers_caught"] == 0


def test_audit_cost_falls_towards_the_floor_for_honest_populations() -> None:
    s = run(attackers=0, rounds=1500)
    assert s["false_accusations"] == 0
    assert 0.05 <= s["audit_cost_fraction"] < 0.11  # floor 5 % plus the early, heavier audits


def test_sleeper_agents_are_caught_after_they_start() -> None:
    s = run(attackers=6, sleeper_after=300, rounds=700)
    assert s["attackers_caught"] == 6
    assert s["max_detection_delay_results"] <= 80  # the honest history bought them nothing
    assert s["false_accusations"] == 0


def test_intermittent_cheaters_are_still_caught() -> None:
    s = run(cheat_rate=0.25, rounds=900, attackers=6)
    assert s["attackers_caught"] >= 5
    assert s["false_accusations"] == 0


def test_undetectable_corruption_is_not_flagged_but_is_reported_as_merged() -> None:
    """If a corruption stays below the tolerance (attack_strength 0) the system cannot see it:
    the simulator reports that honestly instead of claiming protection."""
    s = run(attack_strength=0.0, rounds=200)
    assert s["attackers_caught"] == 0 and s["corrupt_merged_fraction"] == 1.0


def test_deterministic_for_a_seed_and_different_across_seeds() -> None:
    assert run(seed=5) == run(seed=5)
    assert run(seed=5) != run(seed=6)


def test_lifetime_false_accusation_budget_holds_at_scale() -> None:
    s = simulate(
        **{**BASE, "honest": 200, "attackers": 0, "rounds": 1200, "alpha": 0.01, "seed": 2}
    )
    assert s["summary"]["false_accusations"] <= 2  # budget 1 % of 200 devices (observed: 0)
