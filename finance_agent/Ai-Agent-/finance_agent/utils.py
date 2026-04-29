from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import Settings


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def lookback_timestamp(hours: int) -> datetime:
    return utc_now() - timedelta(hours=hours)


def isoformat(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def truncate_text(value: str, limit: int = 1400) -> str:
    clean = " ".join(value.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def build_retry(settings: Settings):
    return retry(
        reraise=True,
        stop=stop_after_attempt(settings.retry_attempts),
        wait=wait_exponential(
            multiplier=1,
            min=settings.retry_min_seconds,
            max=settings.retry_max_seconds,
        ),
        retry=retry_if_exception_type(Exception),
    )


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, indent=2)


def parse_json_response(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    return json.loads(text)
