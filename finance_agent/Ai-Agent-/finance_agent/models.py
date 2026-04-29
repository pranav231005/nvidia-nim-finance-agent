from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class NewsItem:
    title: str
    source: str
    published_at: str
    url: str
    summary: str = ""


@dataclass
class FilingItem:
    title: str
    filing_date: str
    form_type: str
    source: str
    url: str = ""
    summary: str = ""


@dataclass
class MarketSnapshot:
    currency: str
    current_price: float | None
    previous_close: float | None
    percent_change_1d: float | None
    market_cap: float | None
    fifty_two_week_high: float | None
    fifty_two_week_low: float | None


@dataclass
class MacroIndicator:
    name: str
    value: str
    date: str
    source: str
    interpretation: str


@dataclass
class MacroContext:
    news: list[NewsItem] = field(default_factory=list)
    indicators: list[MacroIndicator] = field(default_factory=list)
    global_summary: str = ""


@dataclass
class StockContext:
    ticker: str
    company_name: str
    sector: str
    industry: str
    market_snapshot: MarketSnapshot
    news: list[NewsItem] = field(default_factory=list)
    filings: list[FilingItem] = field(default_factory=list)


@dataclass
class StockAnalysis:
    ticker: str
    company_name: str
    executive_summary: str
    sentiment: str
    sentiment_score: int
    key_events: list[str]
    price_impact: str
    macro_correlation: str
    short_term_outlook: str
    long_term_outlook: str
    risks: list[str]
    opportunities: list[str]
    confidence: str
    cited_sources: list[str]


@dataclass
class ConclusionSummary:
    overview: str
    top_risks: list[str]
    top_opportunities: list[str]
    watch_items: list[str]


@dataclass
class RunArtifacts:
    run_id: str
    generated_at: str
    tickers: list[str]
    report_path: str
    report_url: str | None = None
    metadata_path: str | None = None


def to_dict(data: Any) -> Any:
    if hasattr(data, "__dataclass_fields__"):
        return asdict(data)
    if isinstance(data, list):
        return [to_dict(item) for item in data]
    if isinstance(data, dict):
        return {key: to_dict(value) for key, value in data.items()}
    return data


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
