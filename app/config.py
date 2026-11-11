"""Service settings, read once from the environment."""

from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Shared with the Express API, which signs every request with it
    agent_shared_secret: SecretStr = Field(default=SecretStr(""))
    allowed_clock_skew_seconds: int = Field(default=300, ge=30, le=900)
    # Vercel refuses request bodies over 4.5 MB, so anything bigger is refused here first with a clear message
    max_body_bytes: int = Field(default=4_000_000, ge=10_000, le=4_400_000)
    # Local testing only. Never set on Vercel.
    agent_allow_fake_provider: bool = False

    @field_validator("agent_shared_secret")
    @classmethod
    def _long_enough(cls, value: SecretStr) -> SecretStr:
        secret = value.get_secret_value()
        if secret and len(secret) < 32:
            raise ValueError("AGENT_SHARED_SECRET must be at least 32 characters")
        return value

    @property
    def configured(self) -> bool:
        return bool(self.agent_shared_secret.get_secret_value())


@lru_cache
def get_settings() -> Settings:
    return Settings()
