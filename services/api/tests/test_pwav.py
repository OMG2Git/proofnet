"""PWAV math: tolerance limits, betting e-process, audit probability. Pure functions; Monte Carlo
checks of the statistical guarantees (coverage and lifetime false-accusation control)."""

import json
import math
import random

import pytest

from proofnet_api.trust import pwav
from proofnet_api.trust.pwav import (
    AuditParams,
    EParams,
    audit_probability,
    base_audit_probability,
    binom_cdf,
    block_alpha,
    block_of,
    decay_memory,
    min_samples,
    new_state,
    reward_multiplier,
    threshold,
    tolerance_limit,
    tolerance_rank,
    trust_score,
    update,
)


# ---------------------------------------------------------------- tolerance limits
def test_binom_cdf_known_values() -> None:
    assert binom_cdf(5, 10, 0.5) == pytest.approx(0.623046875)
    assert binom_cdf(-1, 10, 0.5) == 0.0
    assert binom_cdf(10, 10, 0.3) == 1.0
    assert binom_cdf(0, 5, 0.0) == pytest.approx(1.0)
    assert binom_cdf(2, 5, 1.0) == pytest.approx(0.0)


def test_min_samples_and_max_is_a_limit_exactly_then() -> None:
    assert min_samples(0.95, 0.95) == 59  # the classic 59-sample rule
    assert tolerance_rank(59, 0.95, 0.95) == 59  # the maximum
    assert tolerance_rank(58, 0.95, 0.95) is None  # too few samples for any rank
    assert min_samples(0.97, 0.95) == 99


def test_rank_is_the_tightest_valid_one() -> None:
    for n, p, g in [(200, 0.95, 0.95), (500, 0.9, 0.99), (1000, 0.97, 0.9)]:
        j = tolerance_rank(n, p, g)
        assert j is not None and j <= n
        assert binom_cdf(j - 1, n, p) >= g
        if j > 1:
            assert binom_cdf(j - 2, n, p) < g  # a smaller rank would not be valid


def test_tolerance_limit_edge_cases() -> None:
    assert tolerance_limit([], 0.95, 0.95) is None
    assert tolerance_limit([1.0] * 10, 0.95, 0.95) is None  # not enough samples
    xs = [float(i) for i in range(100)]
    lim = tolerance_limit(xs, 0.9, 0.9)
    assert lim is not None and lim in xs and 85 <= lim <= 99


def _lognormal_cdf(x: float, mu: float, sigma: float) -> float:
    return 0.5 * (1 + math.erf((math.log(x) - mu) / (sigma * math.sqrt(2))))


def test_tolerance_limit_coverage_monte_carlo() -> None:
    """With probability >= gamma the limit must cover >= p of the honest distribution."""
    rng = random.Random(1)
    mu, sigma, n, p, gamma, trials = -30.0, 2.0, 120, 0.9, 0.9, 1500
    covered = 0
    for _ in range(trials):
        xs = [math.exp(rng.gauss(mu, sigma)) for _ in range(n)]
        lim = tolerance_limit(xs, p, gamma)
        assert lim is not None
        covered += _lognormal_cdf(lim, mu, sigma) >= p
    assert covered / trials >= gamma - 0.03, covered / trials  # allow Monte Carlo noise


# ---------------------------------------------------------------- e-process
def run(zs: list[int], params: EParams | None = None) -> tuple[dict, dict]:  # type: ignore[type-arg]
    params = params or EParams()
    state = new_state()
    out: dict = {}  # type: ignore[type-arg]
    for z in zs:
        out = update(state, z, params)
        if out["accused"]:
            break
    return state, out


def test_params_validation_and_budget_spending() -> None:
    with pytest.raises(ValueError):
        EParams(q0=0.0)
    with pytest.raises(ValueError):
        EParams(alpha=1.5)
    with pytest.raises(ValueError):
        EParams(q0=0.5, lambdas=(3.0,))  # lambda * q0 >= 1 can zero the process
    assert sum(block_alpha(0.01, m) for m in range(200000)) <= 0.01 + 1e-9  # spends <= alpha
    assert block_alpha(0.01, 0) == pytest.approx(0.01 * 6 / math.pi**2)


def test_blocks_and_thresholds() -> None:
    assert [block_of(n) for n in (1, 1000, 1001, 2000, 2001, 4000, 4001)] == [0, 0, 1, 1, 2, 2, 3]
    p = EParams()
    assert threshold(1, p) == threshold(1000, p)  # constant inside a block
    assert threshold(1001, p) > threshold(1000, p)  # a longer horizon needs more evidence
    assert threshold(1, p) == pytest.approx(1000**2 / block_alpha(p.alpha, 0))
    assert threshold(1, EParams(alpha=1e-6)) > threshold(1, EParams(alpha=1e-3))


def test_clean_audits_never_accuse_and_evidence_stays_small() -> None:
    _, out = run([0] * 5000)
    assert out["accused"] is False and out["exceed"] == 0
    assert out["log10_ratio"] < -8  # nowhere near the threshold
    assert out["suspicion"] == 0.0 and out["evidence"] < 15  # honest history leaves no suspicion


def test_persistent_cheater_is_accused_within_a_few_audits() -> None:
    ns = []
    for alpha in (1e-2, 1e-3, 1e-6, 1e-9):
        _, out = run([1] * 60, EParams(alpha=alpha))
        assert out["accused"] is True and out["n"] <= 14, (alpha, out["n"])
        ns.append(out["n"])
    assert ns == sorted(ns)  # a stricter lifetime guarantee needs (slightly) more evidence


def test_a_single_exceedance_is_not_an_accusation_but_raises_evidence() -> None:
    clean = run([0] * 31)[1]
    one = run([0] * 30 + [1])[1]
    assert one["accused"] is False
    assert one["evidence"] > 5 * clean["evidence"] and one["suspicion"] > clean["suspicion"] + 0.05
    two = run([0] * 30 + [1, 1])[1]
    assert two["suspicion"] > one["suspicion"]


def test_intermittent_cheater_is_eventually_accused() -> None:
    detected, lat = 0, []
    for seed in range(200):
        rng = random.Random(seed)
        zs = [1 if rng.random() < 0.25 else 0 for _ in range(600)]
        _, out = run(zs)
        if out["accused"]:
            detected += 1
            lat.append(out["n"])
    assert detected >= 190, detected  # cheats in 1 of 4 audits: caught almost always
    assert sorted(lat)[len(lat) // 2] < 120


def test_sleeper_agent_cannot_hide_behind_a_long_honest_history() -> None:
    """3000 clean audits, then it cheats on every audit: the detector restarts at each audit, so
    the delay does not grow with the length of the honest history."""
    p = EParams()
    delays = []
    for history in (0, 50, 500, 3000, 20000):
        state = new_state()
        for _ in range(history):
            update(state, 0, p)
        extra, out = 0, {"accused": False}
        while not out["accused"] and extra < 60:
            out = update(state, 1, p)
            extra += 1
        assert out["accused"] is True, history
        delays.append(extra)
    assert max(delays) <= 14 and max(delays) - min(delays) <= 4, delays


def test_false_accusation_rate_is_controlled_even_at_the_honest_boundary() -> None:
    """Honest devices exceed with probability exactly q0 (the worst case allowed): the fraction
    ever accused must stay <= alpha. (The proof is conservative, so the observed rate is far lower.)"""
    alpha = 0.05
    p = EParams(q0=0.03, alpha=alpha)
    rng = random.Random(7)
    devices, audits, accused = 3000, 1500, 0
    for _ in range(devices):
        st = new_state()
        for _ in range(audits):
            z = 1 if rng.random() < p.q0 else 0
            if update(st, z, p)["accused"]:
                accused += 1
                break
    assert accused / devices <= alpha, accused / devices


def test_honest_with_a_few_isolated_exceedances_is_not_accused() -> None:
    """An honest-but-noisy device (exceedance rate below q0) stays safe for a very long time."""
    p = EParams()
    st = new_state()
    rng = random.Random(3)
    for _ in range(3000):
        out = update(st, 1 if rng.random() < 0.01 else 0, p)
        assert not out["accused"]
    back = json.loads(json.dumps(st))  # the state round-trips through the database format
    assert pwav.summary(back, p)["evidence"] == pwav.summary(st, p)["evidence"]


# ---------------------------------------------------------------- audit probability / memory
def test_audit_probability_floor_probation_and_monotonicity() -> None:
    ap = AuditParams()
    assert audit_probability(0, 0, 0.0, ap) == 1.0  # probation: audit everything
    assert audit_probability(ap.probation_results - 1, 0, 0.0, ap) == 1.0
    probs = [audit_probability(100 + n, n, 0.0, ap) for n in (0, 10, 50, 200, 1000, 10**6)]
    assert probs == sorted(probs, reverse=True)  # clean history lowers the audit cost
    assert probs[-1] >= ap.floor and probs[-1] == pytest.approx(ap.floor, abs=1e-3)
    for n_clean in (0, 5, 500, 10**7):  # the floor is never undercut, for any history
        assert audit_probability(10**9, n_clean, 0.0, ap) >= ap.floor
    sus = [audit_probability(500, 200, s / 10, ap) for s in range(11)]
    assert sus == sorted(sus) and sus[-1] == 1.0  # suspicion raises it up to certainty
    assert audit_probability(500, 200, -3, ap) == audit_probability(500, 200, 0, ap)  # clamped
    assert base_audit_probability(0, ap) == pytest.approx(ap.initial)


def test_suspicion_memory_persists_then_fades_slowly() -> None:
    ap = AuditParams()
    mem = decay_memory(0.0, 0.9, ap)
    assert mem == 0.9  # evidence raises it immediately
    for _ in range(50):
        mem = decay_memory(mem, 0.0, ap)
    assert 0.3 < mem < 0.9  # still elevated after 50 clean audits...
    for _ in range(400):
        mem = decay_memory(mem, 0.0, ap)
    assert mem < 0.001  # ...and eventually forgotten
    assert decay_memory(0.5, 0.8, ap) == 0.8  # new evidence overrides


def test_trust_and_reward_multiplier_bounds() -> None:
    assert trust_score(0, 0.0) == 0.0  # unproven
    assert trust_score(90, 0.0) == pytest.approx(0.9)
    assert trust_score(1000, 1.0) == 0.0  # fully suspected
    assert trust_score(100, 0.5) < trust_score(100, 0.0)
    assert trust_score(10, 0.0) < trust_score(1000, 0.0) <= 1.0
    assert reward_multiplier(0.0) == 0.5 and reward_multiplier(1.0) == 1.0
    assert reward_multiplier(2.0) == 1.0 and reward_multiplier(-1.0) == 0.5
