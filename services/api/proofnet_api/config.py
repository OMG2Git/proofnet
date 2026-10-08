"""Configuration: environment variables only (hosting-agnostic, ARCHITECTURE 16.1).

Retry/timeout constants (ARCHITECTURE 6, 8, 11) live here too so they are in one place.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "proofnet_dev"
    jwt_secret: str = Field(default="change-me", min_length=8)
    cors_origins: str = "http://localhost:3000"
    max_upload_mb: int = 25
    admin_emails: str = ""  # comma-separated; these accounts may call /admin/demo/reset
    # --- Part 2: verification / trust (ARCHITECTURE 17) ---
    verification_default_mode: str = "adaptive"  # off | adaptive | full
    pwav_q0: float = 0.03  # honest exceedance probability (tolerance limit covers p = 1 - q0)
    pwav_gamma: float = 0.95  # confidence of the tolerance limit
    pwav_alpha: float = 1e-3  # lifetime false-accusation budget per device
    pwav_margin: float = 10.0  # safety factor on the calibrated tolerance limit
    calibration_cap: int = 5000  # honest discrepancy samples kept per class
    audit_floor: float = 0.05  # minimum audit probability, always
    audit_initial: float = 0.30
    audit_probation_results: int = 5  # a new device is audited on every one of its first results
    audit_tau: float = 10.0
    audit_memory_decay: float = 0.98
    rate_limit_auth_per_minute: int = 60  # sign-up / sign-in / device registration per address
    max_devices_per_user: int = 25
    login_max_failures: int = 5  # consecutive wrong passwords before the account locks
    login_lockout_seconds: int = 300
    reward_rate: float = 1.0  # credits per million work units at full trust
    reward_base: float = 0.5  # reward multiplier of an unproven device (1.0 = fully trusted)
    # Free Atlas tier is 512 MB and blocks writes at the limit: release old data above this budget.
    storage_budget_mb: int = 380
    storage_check_seconds: int = 120
    public_base_url: str = "http://localhost:8000"

    jwt_ttl_minutes: int = 60 * 12
    status_cache_seconds: float = 1.0  # live snapshot cache (0 disables)
    # Worker protocol / failure handling constants
    heartbeat_idle_ms: int = 2000
    heartbeat_busy_ms: int = 5000
    offline_after_seconds: int = 20
    reconciler_interval_seconds: int = 2
    max_attempts: int = 3
    preferred_wait_seconds: int = 10
    queue_timeout_seconds: int = 600
    task_timeout_seconds: int = 900
    # An excluded device (it already failed/lost this chunk) may retry it after this long
    # if nobody else is eligible; attempts stay bounded by max_attempts.
    exclusion_relax_seconds: int = 15
    # Slowest transfer a phone is expected to sustain; used to size assignment deadlines.
    min_bandwidth_bytes_per_s: int = 50_000

    @property
    def cors_origin_list(self) -> list[str]:
        # Browsers send origins without a trailing slash; tolerate one in the setting.
        return [o.strip().rstrip("/") for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
