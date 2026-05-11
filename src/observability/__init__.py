"""
Observability: structured logging, telemetry, tracing, execution timelines,
token tracking, and agent monitoring.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from src.config import Settings
from src.utils.helpers import ensure_directory
from src.models import utc_now_iso

LOGGER = logging.getLogger(__name__)


def configure_observability(settings: Settings) -> None:
    """Set up structured logging with console and rotating file handlers."""
    ensure_directory(settings.logs_dir)

    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        settings.logs_dir / "agent.log",
        maxBytes=5_000_000, backupCount=10, encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root_logger.setLevel(level)
    root_logger.addHandler(console)
    root_logger.addHandler(file_handler)

    # Optionally configure Langfuse
    if settings.telemetry_enabled and settings.langfuse_public_key:
        try:
            import langfuse
            langfuse_client = langfuse.Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            LOGGER.info("Langfuse telemetry configured")
        except ImportError:
            LOGGER.warning("langfuse package not installed; telemetry disabled")
        except Exception:
            pass


class Telemetry:
    """Lightweight telemetry and metrics collector."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._events: list[dict[str, Any]] = []
        self._metrics: dict[str, list[float]] = defaultdict(list)
        self._span_stack: list[dict[str, Any]] = []
        self.session_id: str = utc_now_iso().replace(":", "-")[:19]
        self.session_start = time.perf_counter()
        self.total_tokens: int = 0
        self.total_cost: float = 0.0
        self.tool_calls: int = 0
        self.errors: int = 0

    def record_event(self, event: str, data: dict[str, Any] | None = None) -> None:
        entry = {
            "event": event,
            "timestamp": utc_now_iso(),
            "data": data or {},
            "session_elapsed_s": round(time.perf_counter() - self.session_start, 2),
        }
        self._events.append(entry)

    def record_metric(self, name: str, value: float) -> None:
        self._metrics[name].append(value)

    def record_tool_call(self, tool_name: str, success: bool, duration_ms: int) -> None:
        self.tool_calls += 1
        self.record_event("tool_call", {"tool": tool_name, "success": success, "duration_ms": duration_ms})
        self.record_metric(f"tool_{tool_name}_duration_ms", duration_ms)

    def record_tokens(self, prompt_tokens: int, completion_tokens: int, model: str) -> None:
        self.total_tokens += prompt_tokens + completion_tokens
        self.record_metric("prompt_tokens", prompt_tokens)
        self.record_metric("completion_tokens", completion_tokens)

    def record_error(self, error_type: str, message: str) -> None:
        self.errors += 1
        self.record_event("error", {"type": error_type, "message": message})

    def start_span(self, name: str) -> None:
        self._span_stack.append({"name": name, "start": time.perf_counter()})

    def end_span(self) -> dict[str, Any] | None:
        if not self._span_stack:
            return None
        span = self._span_stack.pop()
        span["duration_ms"] = (time.perf_counter() - span["start"]) * 1000
        self.record_event("span", span)
        return span

    def get_summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "uptime_s": round(time.perf_counter() - self.session_start, 0),
            "total_events": len(self._events),
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "tool_calls": self.tool_calls,
            "errors": self.errors,
            "avg_tool_duration_ms": self._avg_metric("tool_.*_duration_ms"),
        }

    def _avg_metric(self, pattern: str) -> float:
        import re
        values = []
        for name, vals in self._metrics.items():
            if re.match(pattern, name):
                values.extend(vals)
        return round(sum(values) / len(values), 2) if values else 0.0

    def recent_events(self, n: int = 50) -> list[dict[str, Any]]:
        return self._events[-n:]