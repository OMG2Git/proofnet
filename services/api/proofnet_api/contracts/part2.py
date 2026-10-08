"""Part 2 API models: trust profiles, verification records, rewards, security (ARCHITECTURE 17)."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TrustEvent(BaseModel):
    at: datetime
    event: str
    discrepancy: float | None = None
    tolerance: float | None = None
    suspicion: float | None = None
    task_id: str | None = None
    reason: str | None = None


class DeviceTrust(BaseModel):
    device_id: str
    device_name: str
    owner_display_name: str | None = None
    runtime_kind: str | None = None
    status: str  # probation | trusted | watch | quarantined
    results_seen: int
    audits: int
    n_clean: int
    exceedances: int
    evidence: float  # mean Shiryaev-Roberts statistic
    threshold: float  # accusation threshold h at the current audit count
    log10_ratio: float  # log10(evidence / threshold); accuse when >= 0
    suspicion: float = Field(ge=0, le=1)
    memory: float = Field(ge=0, le=1)  # persistent suspicion memory
    trust: float = Field(ge=0, le=1)
    audit_probability: float = Field(ge=0, le=1)  # probability the next result is audited
    reward_multiplier: float
    quarantine: dict[str, Any] | None = None
    history: list[TrustEvent] = []


class VerificationRecord(BaseModel):
    id: str
    at: datetime
    task_id: str
    device_id: str
    device_name: str | None = None
    mode: str
    audit_probability: float | None = None
    draw: float | None = None
    audited: bool
    decision: str
    discrepancy: float | None = None
    tolerance: float | None = None
    tolerance_source: str | None = None
    exceeded: bool | None = None
    evidence: float | None = None
    suspicion: float | None = None
    class_key: str | None = None


class CalibrationClass(BaseModel):
    class_key: str
    n: int
    limit: float | None
    min_samples: int
    p: float
    gamma: float
    margin: float
    tolerance_floor: float | None = None
    tolerance_hard: float | None = None
    tolerance: float | None = None  # what an audit uses now
    source: str  # kernel_default | calibrated
    sample_min: float | None = None
    sample_median: float | None = None
    sample_max: float | None = None


class TrustOverview(BaseModel):
    devices: int
    probation: int
    trusted: int
    watch: int
    quarantined: int
    results_seen: int
    audits: int
    exceedances: int
    audit_rate: float  # audits / results
    parameters: dict[str, float]
    devices_detail: list[DeviceTrust]
    calibration: list[CalibrationClass]
    recent_records: list[VerificationRecord]


class RewardEntry(BaseModel):
    id: str
    task_id: str
    device_id: str
    device_name: str | None = None
    work_units: int
    base_amount: float
    trust: float
    multiplier: float
    amount: float
    acceptance: str
    status: str  # pending | confirmed | revoked
    reason: str | None = None
    created_at: datetime


class Balance(BaseModel):
    pending: float = 0.0
    confirmed: float = 0.0
    revoked: float = 0.0


class RewardsOut(BaseModel):
    balance: Balance
    per_device: dict[str, Balance]
    device_names: dict[str, str]
    entries: list[RewardEntry]
    rate_credits_per_million_units: float
    base_multiplier: float


class NetworkRewardRow(BaseModel):
    display_name: str
    mine: bool
    devices: int
    balance: Balance


class LedgerCheck(BaseModel):
    consistent: bool
    events: int
    entries: int
    users_checked: int
    mismatches: list[str]
    replayed: dict[str, Balance]


class SecurityEvent(BaseModel):
    id: str
    ts: datetime
    kind: str
    severity: str
    device_id: str | None = None
    task_id: str | None = None
    data: dict[str, Any] = {}


class SecurityOverview(BaseModel):
    quarantined_devices: list[DeviceTrust]
    events: list[SecurityEvent]
    counts: dict[str, int]  # events by kind in the last 24 h
    controls: list[dict[str, str]]  # what protects the system, for the dashboard


class ReasonBody(BaseModel):
    reason: str = Field(min_length=1, max_length=300)


class SimulateRequest(BaseModel):
    honest: int = Field(default=20, ge=1, le=200)
    attackers: int = Field(default=5, ge=0, le=100)
    attack_strength: float = Field(default=1.0, ge=0, le=1.0)  # probability a cheat is detectable
    cheat_rate: float = Field(default=1.0, gt=0, le=1.0)  # share of results an attacker corrupts
    sleeper_after: int = Field(default=0, ge=0, le=5000)
    rounds: int = Field(default=400, ge=10, le=2000)
    alpha: float = Field(default=1e-3, gt=0, lt=1)
    q0: float = Field(default=0.03, gt=0, lt=0.5)
    audit_floor: float = Field(default=0.05, ge=0, le=1)
    seed: int = 1
    policy: str = Field(default="adaptive", pattern="^(adaptive|fixed|none)$")
    fixed_rate: float = Field(default=0.3, ge=0, le=1)


class SimulationOut(BaseModel):
    request: SimulateRequest
    series: dict[str, list[float]]  # per round: audits, corrupt accepted, quarantined ...
    summary: dict[str, Any]
    detections: list[dict[str, Any]]
