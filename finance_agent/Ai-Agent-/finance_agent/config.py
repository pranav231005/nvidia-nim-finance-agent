from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "finance-ai-agent"
    environment: str = "production"
    timezone: str = "Asia/Kolkata"
    run_hour: int = 0
    run_minute: int = 0
    data_lookback_hours: int = 24
    news_limit_per_ticker: int = 8
    macro_news_limit: int = 10
    company_news_language: str = "en"
    local_storage_path: Path = Path("./artifacts")
    log_level: str = "INFO"
    request_timeout_seconds: int = 20
    retry_attempts: int = 3
    retry_min_seconds: int = 2
    retry_max_seconds: int = 8

    tickers: list[str] = Field(default_factory=list)

    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"

    newsapi_api_key: str | None = None
    finnhub_api_key: str | None = None
    alpha_vantage_api_key: str | None = None

    email_enabled: bool = False
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: str | None = None
    email_from: str | None = None
    email_to: list[str] = Field(default_factory=list)
    email_subject_prefix: str = "[Finance AI]"

    whatsapp_enabled: bool = False
    twilio_account_sid: str | None = None
    twilio_auth_token: str | None = None
    twilio_whatsapp_from: str | None = None
    twilio_whatsapp_to: list[str] = Field(default_factory=list)

    s3_bucket: str | None = None
    s3_prefix: str = "finance-agent"
    s3_public_base_url: str | None = None
    aws_region: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    report_retention_days: int = 30

    @field_validator("tickers", mode="before")
    @classmethod
    def parse_tickers(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip().upper() for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            return [str(item).strip().upper() for item in value if str(item).strip()]
        raise TypeError("TICKERS must be a comma-separated string or list")

    @field_validator("email_to", "twilio_whatsapp_to", mode="before")
    @classmethod
    def parse_recipients(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        raise TypeError("Recipient settings must be a comma-separated string or list")

    @property
    def reports_dir(self) -> Path:
        return self.local_storage_path / "reports"

    @property
    def runs_dir(self) -> Path:
        return self.local_storage_path / "runs"

    @property
    def logs_dir(self) -> Path:
        return self.local_storage_path / "logs"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
