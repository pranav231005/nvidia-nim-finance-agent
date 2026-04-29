from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI

from .config import Settings
from .models import ConclusionSummary, MacroContext, StockAnalysis, StockContext, to_dict
from .utils import json_dumps, parse_json_response

LOGGER = logging.getLogger(__name__)


class Analyzer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def summarize_macro(self, macro_context: MacroContext) -> str:
        if not macro_context.news and not macro_context.indicators:
            return "No macroeconomic updates were collected in the configured lookback window."

        prompt = (
            "You are a macro strategist writing a concise research note.\n"
            "Return JSON only with this schema:\n"
            '{"summary":"string"}\n\n'
            f"Macro context:\n{json_dumps(to_dict(macro_context))}"
        )
        fallback = self._fallback_macro_summary(macro_context)
        result = self._run_json_prompt(prompt, fallback={"summary": fallback})
        return str(result.get("summary") or fallback)

    def analyze_stock(self, stock_context: StockContext, macro_context: MacroContext) -> StockAnalysis:
        fallback = self._fallback_stock_analysis(stock_context, macro_context)
        prompt = (
            "You are a senior equity research analyst.\n"
            "Return valid JSON only.\n"
            "Schema:\n"
            "{"
            '"executive_summary":"string",'
            '"sentiment":"positive|negative|neutral",'
            '"sentiment_score":0,'
            '"key_events":["string"],'
            '"price_impact":"string",'
            '"macro_correlation":"string",'
            '"short_term_outlook":"string",'
            '"long_term_outlook":"string",'
            '"risks":["string"],'
            '"opportunities":["string"],'
            '"confidence":"high|medium|low",'
            '"cited_sources":["string"]'
            "}\n\n"
            "Use only the provided inputs. Be balanced, specific, and professional.\n\n"
            f"Stock context:\n{json_dumps(to_dict(stock_context))}\n\n"
            f"Macro context:\n{json_dumps(to_dict(macro_context))}"
        )
        result = self._run_json_prompt(prompt, fallback=to_dict(fallback))

        return StockAnalysis(
            ticker=stock_context.ticker,
            company_name=stock_context.company_name,
            executive_summary=str(result.get("executive_summary") or fallback.executive_summary),
            sentiment=str(result.get("sentiment") or fallback.sentiment).lower(),
            sentiment_score=int(result.get("sentiment_score") or fallback.sentiment_score),
            key_events=self._coerce_list(result.get("key_events"), fallback.key_events),
            price_impact=str(result.get("price_impact") or fallback.price_impact),
            macro_correlation=str(result.get("macro_correlation") or fallback.macro_correlation),
            short_term_outlook=str(result.get("short_term_outlook") or fallback.short_term_outlook),
            long_term_outlook=str(result.get("long_term_outlook") or fallback.long_term_outlook),
            risks=self._coerce_list(result.get("risks"), fallback.risks),
            opportunities=self._coerce_list(result.get("opportunities"), fallback.opportunities),
            confidence=str(result.get("confidence") or fallback.confidence).lower(),
            cited_sources=self._coerce_list(result.get("cited_sources"), fallback.cited_sources),
        )

    def build_conclusion(
        self,
        analyses: list[StockAnalysis],
        macro_summary: str,
    ) -> ConclusionSummary:
        fallback = self._fallback_conclusion(analyses, macro_summary)
        prompt = (
            "You are compiling the conclusion section of an equity research packet.\n"
            "Return JSON only with schema:\n"
            '{"overview":"string","top_risks":["string"],"top_opportunities":["string"],"watch_items":["string"]}\n\n'
            f"Macro summary:\n{macro_summary}\n\n"
            f"Stock analyses:\n{json_dumps(to_dict(analyses))}"
        )
        result = self._run_json_prompt(prompt, fallback=to_dict(fallback))
        return ConclusionSummary(
            overview=str(result.get("overview") or fallback.overview),
            top_risks=self._coerce_list(result.get("top_risks"), fallback.top_risks),
            top_opportunities=self._coerce_list(result.get("top_opportunities"), fallback.top_opportunities),
            watch_items=self._coerce_list(result.get("watch_items"), fallback.watch_items),
        )

    def _run_json_prompt(self, prompt: str, fallback: dict[str, Any]) -> dict[str, Any]:
        if not self.client:
            LOGGER.warning("OPENAI_API_KEY not configured; using heuristic analysis fallback.")
            return fallback

        try:
            response = self.client.responses.create(
                model=self.settings.openai_model,
                temperature=0.2,
                max_output_tokens=1400,
                input=[
                    {
                        "role": "system",
                        "content": "You are a disciplined financial analysis assistant. Return JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return parse_json_response(response.output_text)
        except Exception as exc:
            LOGGER.exception("OpenAI analysis failed; using fallback. Error: %s", exc)
            return fallback

    def _fallback_macro_summary(self, macro_context: MacroContext) -> str:
        fragments: list[str] = []
        if macro_context.indicators:
            fragments.extend(indicator.interpretation for indicator in macro_context.indicators[:3])
        if macro_context.news:
            titles = "; ".join(news.title for news in macro_context.news[:3])
            fragments.append(f"Key global finance headlines in the last 24 hours included: {titles}.")
        return " ".join(fragments) or "Macro updates were limited during the lookback window."

    def _fallback_stock_analysis(self, stock_context: StockContext, macro_context: MacroContext) -> StockAnalysis:
        pct_change = stock_context.market_snapshot.percent_change_1d
        if pct_change is None:
            sentiment = "neutral"
            score = 50
        elif pct_change > 1:
            sentiment = "positive"
            score = 68
        elif pct_change < -1:
            sentiment = "negative"
            score = 32
        else:
            sentiment = "neutral"
            score = 50

        news_titles = [item.title for item in stock_context.news[:3]]
        key_events = news_titles or ["No material company-specific headlines were found in the last 24 hours."]
        price_sentence = (
            f"The stock moved {pct_change}% over the latest trading session."
            if pct_change is not None
            else "Recent price performance could not be computed from the available market data."
        )

        macro_reference = (
            macro_context.indicators[0].interpretation
            if macro_context.indicators
            else "Macro linkage is driven primarily by general market sentiment and sector-level news."
        )

        return StockAnalysis(
            ticker=stock_context.ticker,
            company_name=stock_context.company_name,
            executive_summary=(
                f"{stock_context.company_name} remains influenced by its latest company headlines, "
                f"near-term price action, and the broader macro backdrop."
            ),
            sentiment=sentiment,
            sentiment_score=score,
            key_events=key_events,
            price_impact=price_sentence,
            macro_correlation=macro_reference,
            short_term_outlook=(
                "Short-term performance will likely depend on whether recent news flow translates into "
                "earnings momentum, guidance changes, or sector rerating."
            ),
            long_term_outlook=(
                "Long-term positioning should be assessed against competitive strength, balance sheet durability, "
                "capital allocation, and sensitivity to macro cycles."
            ),
            risks=[
                "Unexpected macro tightening or risk-off market moves.",
                "Earnings disappointment or adverse guidance revisions.",
            ],
            opportunities=[
                "Positive earnings revisions or improved operating momentum.",
                "Valuation rerating if macro conditions stabilize.",
            ],
            confidence="medium",
            cited_sources=[item.url for item in stock_context.news[:5] if item.url],
        )

    def _fallback_conclusion(self, analyses: list[StockAnalysis], macro_summary: str) -> ConclusionSummary:
        positive = [analysis.ticker for analysis in analyses if analysis.sentiment == "positive"]
        negative = [analysis.ticker for analysis in analyses if analysis.sentiment == "negative"]
        overview_parts = [macro_summary]
        if positive:
            overview_parts.append(f"Relatively constructive tone was observed in {', '.join(positive)}.")
        if negative:
            overview_parts.append(f"Near-term pressure was more visible in {', '.join(negative)}.")
        overview = " ".join(overview_parts).strip()
        return ConclusionSummary(
            overview=overview or "Portfolio-level signals were mixed across the tracked stocks.",
            top_risks=[
                "Macro volatility spilling into equity multiples.",
                "Company-specific execution or earnings miss risk.",
            ],
            top_opportunities=[
                "Selective upside from positive revisions and resilient demand trends.",
                "Improved sentiment if inflation and rates move in a market-friendly direction.",
            ],
            watch_items=["Upcoming earnings, central bank commentary, and abrupt shifts in market breadth."],
        )

    def _coerce_list(self, value: Any, fallback: list[str]) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        return fallback
