from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import requests
import yfinance as yf

from .config import Settings
from .models import FilingItem, MacroContext, MacroIndicator, MarketSnapshot, NewsItem, StockContext
from .utils import build_retry, isoformat, lookback_timestamp, safe_float, truncate_text

LOGGER = logging.getLogger(__name__)


class DataFetcher:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "finance-ai-agent/1.0",
                "Accept": "application/json",
            }
        )
        self.retry = build_retry(settings)

    def fetch_macro_context(self) -> MacroContext:
        news = self._fetch_macro_news()
        indicators = self._fetch_macro_indicators()
        return MacroContext(news=news, indicators=indicators)

    def fetch_stock_context(self, ticker: str) -> StockContext:
        ticker_obj = yf.Ticker(ticker)
        info = self._safe_ticker_info(ticker_obj)
        company_name = (
            info.get("longName")
            or info.get("shortName")
            or info.get("displayName")
            or ticker
        )
        sector = info.get("sector") or "Unknown"
        industry = info.get("industry") or "Unknown"
        snapshot = self._fetch_market_snapshot(ticker_obj, info)
        news = self._merge_news_sources(ticker=ticker, company_name=company_name, ticker_obj=ticker_obj)
        filings = self._fetch_filings(ticker)
        return StockContext(
            ticker=ticker,
            company_name=company_name,
            sector=sector,
            industry=industry,
            market_snapshot=snapshot,
            news=news[: self.settings.news_limit_per_ticker],
            filings=filings,
        )

    def _request_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        @self.retry
        def _do_request() -> dict[str, Any]:
            response = self.session.get(
                url,
                params=params,
                timeout=self.settings.request_timeout_seconds,
            )
            response.raise_for_status()
            return response.json()

        return _do_request()

    def _safe_ticker_info(self, ticker_obj: yf.Ticker) -> dict[str, Any]:
        try:
            return ticker_obj.info or {}
        except Exception as exc:
            LOGGER.warning("Failed to load yfinance info for %s: %s", ticker_obj.ticker, exc)
            return {}

    def _fetch_market_snapshot(self, ticker_obj: yf.Ticker, info: dict[str, Any]) -> MarketSnapshot:
        try:
            history = ticker_obj.history(period="5d", interval="1d", auto_adjust=False)
        except Exception as exc:
            LOGGER.warning("Failed to fetch price history for %s: %s", ticker_obj.ticker, exc)
            history = None

        current_price = None
        previous_close = safe_float(info.get("previousClose"))
        percent_change = None

        if history is not None and not history.empty:
            closes = history["Close"].dropna().tolist()
            if closes:
                current_price = safe_float(closes[-1])
                if len(closes) >= 2:
                    previous_close = safe_float(closes[-2])
        if current_price is not None and previous_close not in (None, 0):
            percent_change = round(((current_price - previous_close) / previous_close) * 100, 2)

        return MarketSnapshot(
            currency=info.get("currency") or "USD",
            current_price=current_price,
            previous_close=previous_close,
            percent_change_1d=percent_change,
            market_cap=safe_float(info.get("marketCap")),
            fifty_two_week_high=safe_float(info.get("fiftyTwoWeekHigh")),
            fifty_two_week_low=safe_float(info.get("fiftyTwoWeekLow")),
        )

    def _merge_news_sources(self, ticker: str, company_name: str, ticker_obj: yf.Ticker) -> list[NewsItem]:
        items: list[NewsItem] = []
        items.extend(self._fetch_yfinance_news(ticker_obj))
        items.extend(self._fetch_newsapi_company_news(company_name, ticker))

        deduped: dict[str, NewsItem] = {}
        for item in sorted(items, key=lambda news: news.published_at, reverse=True):
            key = item.url or item.title
            if key and key not in deduped:
                deduped[key] = item
        return list(deduped.values())

    def _fetch_yfinance_news(self, ticker_obj: yf.Ticker) -> list[NewsItem]:
        lookback = lookback_timestamp(self.settings.data_lookback_hours)
        results: list[NewsItem] = []
        try:
            raw_news = ticker_obj.news or []
        except Exception as exc:
            LOGGER.warning("Failed to fetch yfinance news for %s: %s", ticker_obj.ticker, exc)
            return results

        for item in raw_news:
            published_epoch = item.get("providerPublishTime")
            published_dt = None
            if published_epoch:
                published_dt = datetime.fromtimestamp(published_epoch, tz=timezone.utc)
            if published_dt and published_dt < lookback:
                continue

            title = item.get("title") or "Untitled"
            summary = item.get("summary") or item.get("snippet") or ""
            link = item.get("link") or item.get("canonicalUrl", {}).get("url") or ""
            source = item.get("publisher") or "Yahoo Finance"
            results.append(
                NewsItem(
                    title=title,
                    source=source,
                    published_at=isoformat(published_dt or datetime.now(timezone.utc)),
                    url=link,
                    summary=truncate_text(summary, 500),
                )
            )
        return results

    def _fetch_newsapi_company_news(self, company_name: str, ticker: str) -> list[NewsItem]:
        if not self.settings.newsapi_api_key:
            return []

        lookback = lookback_timestamp(self.settings.data_lookback_hours)
        query = f'"{company_name}" OR "{ticker}"'
        payload = self._request_json(
            "https://newsapi.org/v2/everything",
            {
                "apiKey": self.settings.newsapi_api_key,
                "q": query,
                "sortBy": "publishedAt",
                "language": self.settings.company_news_language,
                "from": lookback.isoformat(),
                "pageSize": self.settings.news_limit_per_ticker,
            },
        )
        articles = payload.get("articles", [])
        return [
            NewsItem(
                title=article.get("title") or "Untitled",
                source=(article.get("source") or {}).get("name") or "NewsAPI",
                published_at=article.get("publishedAt") or "",
                url=article.get("url") or "",
                summary=truncate_text(article.get("description") or article.get("content") or "", 500),
            )
            for article in articles
        ]

    def _fetch_filings(self, ticker: str) -> list[FilingItem]:
        if not self.settings.finnhub_api_key:
            return []

        lookback = lookback_timestamp(self.settings.data_lookback_hours).date().isoformat()
        today = datetime.now(timezone.utc).date().isoformat()
        payload = self._request_json(
            "https://finnhub.io/api/v1/stock/filings",
            {
                "symbol": ticker,
                "from": lookback,
                "to": today,
                "token": self.settings.finnhub_api_key,
            },
        )
        filings: list[FilingItem] = []
        for item in payload if isinstance(payload, list) else []:
            filings.append(
                FilingItem(
                    title=item.get("title") or item.get("form") or "Regulatory filing",
                    filing_date=item.get("filingDate") or "",
                    form_type=item.get("form") or "N/A",
                    source="Finnhub",
                    url=item.get("reportUrl") or item.get("reportLink") or "",
                    summary=truncate_text(item.get("description") or "", 400),
                )
            )
        return filings

    def _fetch_macro_news(self) -> list[NewsItem]:
        if not self.settings.newsapi_api_key:
            return []

        lookback = lookback_timestamp(self.settings.data_lookback_hours)
        query = (
            '"Federal Reserve" OR inflation OR recession OR "oil prices" OR '
            '"central bank" OR tariff OR geopolitics OR "global markets"'
        )
        payload = self._request_json(
            "https://newsapi.org/v2/everything",
            {
                "apiKey": self.settings.newsapi_api_key,
                "q": query,
                "sortBy": "publishedAt",
                "language": "en",
                "from": lookback.isoformat(),
                "pageSize": self.settings.macro_news_limit,
            },
        )
        articles = payload.get("articles", [])
        return [
            NewsItem(
                title=article.get("title") or "Untitled",
                source=(article.get("source") or {}).get("name") or "NewsAPI",
                published_at=article.get("publishedAt") or "",
                url=article.get("url") or "",
                summary=truncate_text(article.get("description") or article.get("content") or "", 500),
            )
            for article in articles
        ]

    def _fetch_macro_indicators(self) -> list[MacroIndicator]:
        if not self.settings.alpha_vantage_api_key:
            return []

        indicators: list[MacroIndicator] = []
        indicators.extend(self._fetch_alpha_vantage_indicator("FEDERAL_FUNDS_RATE", "Federal Funds Rate"))
        indicators.extend(self._fetch_alpha_vantage_indicator("INFLATION", "Inflation"))
        return indicators

    def _fetch_alpha_vantage_indicator(self, function_name: str, label: str) -> list[MacroIndicator]:
        payload = self._request_json(
            "https://www.alphavantage.co/query",
            {
                "function": function_name,
                "apikey": self.settings.alpha_vantage_api_key,
                "interval": "monthly",
            },
        )
        datapoints = payload.get("data") or []
        if not datapoints:
            return []

        latest = datapoints[0]
        previous = datapoints[1] if len(datapoints) > 1 else None
        latest_value = latest.get("value", "N/A")
        interpretation = f"{label} latest reading is {latest_value}."
        if previous and latest.get("value") and previous.get("value"):
            try:
                delta = float(latest["value"]) - float(previous["value"])
                if delta > 0:
                    interpretation += " The indicator increased versus the previous reading."
                elif delta < 0:
                    interpretation += " The indicator decreased versus the previous reading."
                else:
                    interpretation += " The indicator was unchanged versus the previous reading."
            except ValueError:
                pass

        return [
            MacroIndicator(
                name=label,
                value=str(latest_value),
                date=latest.get("date") or "",
                source="Alpha Vantage",
                interpretation=interpretation,
            )
        ]
