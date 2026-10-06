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
    public_base_url: str = "http://localhost:8000"

    jwt_ttl_minutes: int = 60 * 12
    # Worker protocol / failure handling constants
    heartbeat_idle_ms: int = 2000
    heartbeat_busy_ms: int = 5000
    offline_after_seconds: int = 20
    reconciler_interval_seconds: int = 2
    max_attempts: int = 3
    preferred_wait_seconds: int = 10
    queue_timeout_seconds: int = 600
    task_timeout_seconds: int = 900

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
