"""
Core configuration for the autonomous AI agent system.
Uses pydantic-settings for type-safe env-based configuration.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    app_name: str = "autonomous-ai-agent"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    timezone: str = "UTC"
    local_storage_path: Path = Path("./artifacts")
    log_level: str = "INFO"

    # --- LLM ---
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o"
    openai_fast_model: str = "gpt-4o-mini"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-6"
    google_api_key: str | None = None
    google_model: str = "gemini-2.5-pro"
    nim_api_key: str | None = None
    nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    nim_model: str = "deepseek-ai/deepseek-v4-pro"
    default_llm_provider: Literal["openai", "anthropic", "google", "nim"] = "openai"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 4096
    llm_request_timeout: int = 120
    llm_max_retries: int = 3

    # --- Memory ---
    chroma_persist_directory: str = "./artifacts/chromadb"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    short_term_memory_limit: int = 100
    episodic_memory_limit: int = 500
    vector_search_top_k: int = 5
    memory_similarity_threshold: float = 0.7

    # --- Task Queue ---
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    task_timeout_seconds: int = 600
    max_concurrent_tasks: int = 5

    # --- Database ---
    postgres_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ai_agent"
    database_pool_size: int = 10
    database_max_overflow: int = 5

    # --- API Server ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    websocket_ping_interval: int = 30
    websocket_ping_timeout: int = 10

    # --- Execution ---
    max_autonomous_steps: int = 50
    max_retries_per_task: int = 3
    execution_timeout_seconds: int = 3600
    max_tool_calls_per_step: int = 5
    recursion_depth_limit: int = 10
    interruption_recovery_enabled: bool = True

    # --- Safety ---
    require_human_approval: bool = False
    human_approval_for: list[str] = Field(default_factory=lambda: [
        "file_delete",
        "git_push",
        "shell_destructive",
        "code_exec",
    ])
    allowed_domains: list[str] = Field(default_factory=list)
    blocked_domains: list[str] = Field(default_factory=list)
    max_cost_per_session_usd: float = 50.0
    max_tokens_per_session: int = 1_000_000
    sandbox_enabled: bool = True
    sandbox_timeout_seconds: int = 30

    # --- Finance APIs ---
    tickers: list[str] = Field(default_factory=list)
    yahoo_finance_enabled: bool = True
    newsapi_api_key: str | None = None
    finnhub_api_key: str | None = None
    alpha_vantage_api_key: str | None = None
    polygon_api_key: str | None = None
    sec_api_key: str | None = None

    # --- GitHub ---
    github_token: str | None = None
    github_default_org: str | None = None

    # --- Email ---
    email_enabled: bool = False
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: str | None = None
    email_from: str | None = None
    email_to: list[str] = Field(default_factory=list)

    # --- Observability ---
    telemetry_enabled: bool = True
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    sentry_dsn: str | None = None
    metrics_port: int = 9090

    # --- AWS ---
    s3_bucket: str | None = None
    aws_region: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    # --- Validators ---
    @field_validator("tickers", mode="before")
    @classmethod
    def parse_tickers(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [t.strip().upper() for t in value.split(",") if t.strip()]
        if isinstance(value, list):
            return [str(t).strip().upper() for t in value if str(t).strip()]
        raise TypeError("TICKERS must be a comma-separated string or list")

    @field_validator("email_to", mode="before")
    @classmethod
    def parse_recipients(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [r.strip() for r in value.split(",") if r.strip()]
        return list(value) if isinstance(value, list) else []

    # --- Derived paths ---
    @property
    def artifacts_dir(self) -> Path:
        return self.local_storage_path

    @property
    def reports_dir(self) -> Path:
        return self.local_storage_path / "reports"

    @property
    def runs_dir(self) -> Path:
        return self.local_storage_path / "runs"

    @property
    def logs_dir(self) -> Path:
        return self.local_storage_path / "logs"

    @property
    def tool_logs_dir(self) -> Path:
        return self.local_storage_path / "tool_logs"

    @property
    def memory_dir(self) -> Path:
        return self.local_storage_path / "memory"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()