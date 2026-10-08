"""PWAV building blocks: pure, dependency-free functions (stdlib only) so they can be tested
exhaustively and reused by the live system and the simulator.

Evidence-Adaptive Auditing of untrusted compute nodes, as implemented in ProofNet:

1. Per-class order-statistic tolerance limits. For one "class" (kernel x version x runtime kind) the
   honest discrepancies between a device result and the backend's recomputation are collected. The
   j-th smallest of n samples is a distribution-free upper (p, gamma) tolerance limit: with
   confidence gamma, at least a fraction p of future honest discrepancies lie below it. A device
   result whose discrepancy exceeds the limit is an *exceedance*; an honest device exceeds with
   probability at most q0 = 1 - p.

2. Betting e-detector per device with lifetime false-accusation control. For every audit, Z = 1 if
   the discrepancy exceeded the limit. Under honesty E[Z | past] <= q0, so for a bet lambda in
   (0, 1/q0) the factor f = 1 + lambda (Z - q0) has conditional mean <= 1. The Shiryaev-Roberts
   statistic  R_t = (R_{t-1} + 1) f_t  (kept per lambda and averaged) restarts at every audit, so a
   long honest history cannot hide later cheating ("sleeper" resistance), and cheating makes R grow
   geometrically. R_t is a sum over start times s of products P_s(t) = prod_{i=s..t} f_i, each a
   non-negative supermartingale started at 1. If R_t >= h at some t <= N, one of the <= N terms is
   >= h/N, and by Ville's inequality that has probability <= N/h; a union bound over the starts
   gives P(false alarm within N audits) <= N^2 / h. We therefore accuse when R_t >= h_m with
   h_m = N_m^2 / alpha_m for the audit block m (N_m = T0 * 2^m, alpha_m = alpha * 6 / (pi^2 (m+1)^2),
   sum alpha_m <= alpha): the probability that an honest device is EVER accused is <= alpha. (This
   holds conditionally on the tolerance limit being valid, i.e. with probability >= gamma.)

3. Adaptive audit probability with a minimum floor: new devices are audited on every result, the
   base rate decays with clean audits towards the floor, and suspicion (evidence, with persistent
   memory across sessions) pushes the probability towards 1. It never drops below the floor.
"""

import math
from dataclasses import dataclass, field
from typing import Any

# ----------------------------------------------------------------------------- tolerance limits


def _log_binom_pmf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    total = sum(math.exp(_log_binom_pmf(i, n, p)) for i in range(k + 1))
    return min(1.0, total)


def min_samples(p: float, gamma: float) -> int:
    """Smallest n for which the sample maximum is an upper (p, gamma) tolerance limit:
    1 - p^n >= gamma."""
    return math.ceil(math.log(1.0 - gamma) / math.log(p))


def tolerance_rank(n: int, p: float, gamma: float) -> int | None:
    """Smallest rank j (1-based, ascending) such that X_(j) is an upper (p, gamma) tolerance limit:
    P(X_(j) >= q_p) = P(Binomial(n, p) <= j - 1) >= gamma. None if n is too small even for j = n."""
    for j in range(1, n + 1):
        if binom_cdf(j - 1, n, p) >= gamma:
            return j
    return None


def tolerance_limit(samples: list[float], p: float, gamma: float) -> float | None:
    """Distribution-free upper (p, gamma) tolerance limit from honest discrepancy samples."""
    if not samples:
        return None
    j = tolerance_rank(len(samples), p, gamma)
    if j is None:
        return None
    return sorted(samples)[j - 1]


# ----------------------------------------------------------------------------- e-detector
LAMBDAS = (1.0, 2.0, 4.0, 8.0, 16.0, 30.0)  # bets; must satisfy lambda * q0 < 1
BLOCK0 = 1000  # first audit block length T0


@dataclass(frozen=True)
class EParams:
    q0: float = 0.03  # honest exceedance probability bound (= 1 - p)
    alpha: float = 1e-3  # lifetime false-accusation budget per device
    lambdas: tuple[float, ...] = LAMBDAS

    def __post_init__(self) -> None:
        if not 0 < self.q0 < 1 or not 0 < self.alpha < 1:
            raise ValueError("q0 and alpha must be in (0, 1)")
        if any(lam * self.q0 >= 1 or lam <= 0 for lam in self.lambdas):
            raise ValueError("each bet needs 0 < lambda < 1 / q0")


def block_alpha(alpha: float, m: int) -> float:
    """Budget of audit block m: alpha * 6 / (pi^2 (m+1)^2); the budgets sum to alpha."""
    return alpha * 6.0 / (math.pi**2 * (m + 1) ** 2)


def block_of(n: int) -> int:
    """Audit block of the n-th audit: N_m = BLOCK0 * 2^m is the smallest block end >= n."""
    return 0 if n <= BLOCK0 else math.ceil(math.log2(n / BLOCK0))


def threshold(n: int, params: EParams) -> float:
    """Accusation threshold h at audit n: N_m^2 / alpha_m."""
    m = block_of(n)
    return float((BLOCK0 * 2**m) ** 2 / block_alpha(params.alpha, m))


def new_state() -> dict[str, Any]:
    return {"n": 0, "exceed": 0, "R": []}  # R: Shiryaev-Roberts statistic per lambda


def update(state: dict[str, Any], z: int, params: EParams) -> dict[str, Any]:
    """Apply one audit outcome (z = 1: the discrepancy exceeded the limit)."""
    if not state["R"]:
        state["R"] = [0.0] * len(params.lambdas)
    state["n"] += 1
    state["exceed"] += int(z)
    state["R"] = [
        min((r + 1.0) * (1.0 + lam * (z - params.q0)), 1e300)
        for r, lam in zip(state["R"], params.lambdas, strict=True)
    ]
    return summary(state, params)


def clean_baseline(params: EParams) -> float:
    """Steady state of the statistic for an all-clean history: R* = f / (1 - f), f = 1 - lambda q0."""
    return sum((1.0 - lam * params.q0) / (lam * params.q0) for lam in params.lambdas) / len(
        params.lambdas
    )


def summary(state: dict[str, Any], params: EParams | None = None) -> dict[str, Any]:
    params = params or EParams()
    n = max(state["n"], 1)
    e = sum(state["R"]) / len(state["R"]) if state["R"] else 0.0
    h = threshold(n, params)
    base = clean_baseline(params)
    excess = max(e / base, 1.0)  # evidence beyond what an honest history produces anyway
    return {
        "n": state["n"],
        "exceed": state["exceed"],
        "evidence": e,  # E_t: mean Shiryaev-Roberts statistic
        "threshold": h,
        "log10_ratio": math.log10(max(e, 1e-300)) - math.log10(h),  # accuse when >= 0
        "suspicion": max(0.0, min(1.0, math.log10(excess) / math.log10(h / base))),
        "accused": e >= h,
    }


# ----------------------------------------------------------------------------- audit policy
@dataclass(frozen=True)
class AuditParams:
    floor: float = 0.05  # minimum audit probability, always
    initial: float = 0.30  # base probability right after probation
    probation_results: int = 5  # a new device is audited on every one of its first results
    tau: float = 50.0  # clean audits for the base rate to close half the gap to the floor
    memory_decay: float = 0.98  # per audit decay of the persistent suspicion memory


def base_audit_probability(n_clean: int, params: AuditParams) -> float:
    """Decreases with the device's clean-audit history, towards (never below) the floor."""
    return params.floor + (params.initial - params.floor) / (1.0 + n_clean / params.tau)


def audit_probability(
    results_seen: int, n_clean: int, suspicion: float, params: AuditParams
) -> float:
    """Adaptive audit probability in [floor, 1]."""
    if results_seen < params.probation_results:
        return 1.0
    base = base_audit_probability(n_clean, params)
    s = max(0.0, min(1.0, suspicion))
    return max(params.floor, min(1.0, base + (1.0 - base) * s))


def decay_memory(memory: float, suspicion_now: float, params: AuditParams) -> float:
    """Persistent suspicion memory: it rises with evidence and fades slowly, so a device that
    cheated once stays audited more for a long time even if it behaves afterwards."""
    return max(memory * params.memory_decay, suspicion_now)


# ----------------------------------------------------------------------------- trust / rewards
def trust_score(n_clean: int, suspicion: float, confidence_k: float = 10.0) -> float:
    """Trust in [0, 1]: grows with clean audits (confidence) and is cut by suspicion."""
    confidence = n_clean / (n_clean + confidence_k)
    return max(0.0, min(1.0, (1.0 - suspicion) * confidence))


def reward_multiplier(trust: float, base: float = 0.5) -> float:
    """Trust adjustment of the reward: base for an unproven device, 1.0 for a fully trusted one."""
    return base + (1.0 - base) * max(0.0, min(1.0, trust))


@dataclass
class Evidence:
    """Convenience bundle used by the simulator and tests."""

    state: dict[str, Any] = field(default_factory=new_state)
    memory: float = 0.0
    n_clean: int = 0
    results_seen: int = 0
