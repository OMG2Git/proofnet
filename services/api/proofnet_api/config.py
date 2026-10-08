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
