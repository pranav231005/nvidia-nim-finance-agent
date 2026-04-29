from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from .analyzer import Analyzer
from .config import Settings, get_settings
from .data_fetcher import DataFetcher
from .logging_config import configure_logging
from .models import StockAnalysis, to_dict, utc_timestamp
from .notifier import Notifier
from .report_generator import ReportGenerator
from .scheduler import FinanceAgentScheduler
from .storage import StorageManager

LOGGER = logging.getLogger(__name__)


class FinanceAgentService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.fetcher = DataFetcher(settings)
        self.analyzer = Analyzer(settings)
        self.report_generator = ReportGenerator()
        self.storage = StorageManager(settings)
        self.notifier = Notifier(settings)

    def run_once(self, tickers: list[str] | None = None) -> dict:
        active_tickers = tickers or self.settings.tickers
        if not active_tickers:
            raise ValueError("No tickers configured. Set TICKERS in .env or pass --tickers.")

        run_id = datetime.now(tz=ZoneInfo(self.settings.timezone)).strftime("%Y%m%d-%H%M%S")
        generated_at = utc_timestamp()
        LOGGER.info("Starting finance agent run %s for tickers: %s", run_id, ", ".join(active_tickers))

        macro_context = self.fetcher.fetch_macro_context()
        macro_summary = self.analyzer.summarize_macro(macro_context)

        analyses: list[StockAnalysis] = []
        stock_errors: list[dict[str, str]] = []
        for ticker in active_tickers:
            try:
                context = self.fetcher.fetch_stock_context(ticker)
                analysis = self.analyzer.analyze_stock(context, macro_context)
                analyses.append(analysis)
            except Exception as exc:
                LOGGER.exception("Stock analysis failed for %s: %s", ticker, exc)
                stock_errors.append({"ticker": ticker, "error": str(exc)})

        if not analyses:
            raise RuntimeError("All stock analyses failed; report generation aborted.")

        conclusion = self.analyzer.build_conclusion(analyses, macro_summary)

        report_path = self.storage.build_report_path(run_id)
        report_date = datetime.now(tz=ZoneInfo(self.settings.timezone)).strftime("%d %b %Y %H:%M %Z")
        self.report_generator.generate(
            output_path=report_path,
            report_date=report_date,
            tickers=active_tickers,
            macro_context=macro_context,
            macro_summary=macro_summary,
            analyses=analyses,
            conclusion=conclusion,
        )

        report_url = self.storage.upload_report(report_path)
        metadata = {
            "run_id": run_id,
            "generated_at": generated_at,
            "tickers": active_tickers,
            "macro_context": to_dict(macro_context),
            "macro_summary": macro_summary,
            "analyses": to_dict(analyses),
            "conclusion": to_dict(conclusion),
            "errors": stock_errors,
            "report_path": str(report_path.resolve()),
            "report_url": report_url,
        }
        metadata_path = self.storage.save_run_metadata(run_id, metadata)
        artifacts = self.storage.finalize_artifacts(
            run_id=run_id,
            generated_at=generated_at,
            tickers=active_tickers,
            report_path=report_path,
            metadata_path=metadata_path,
            report_url=report_url,
        )

        notification_summary = self._notification_summary(analyses, stock_errors)
        self.notifier.send_notifications(artifacts, notification_summary)
        LOGGER.info("Finance agent run %s completed successfully.", run_id)
        return metadata

    def _notification_summary(self, analyses: list[StockAnalysis], errors: list[dict[str, str]]) -> str:
        sentiments = ", ".join(f"{item.ticker}: {item.sentiment.title()}" for item in analyses)
        if errors:
            return f"Daily finance report generated. Sentiment snapshot: {sentiments}. Partial errors: {len(errors)}."
        return f"Daily finance report generated. Sentiment snapshot: {sentiments}."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous finance AI agent")
    parser.add_argument(
        "mode",
        nargs="?",
        default="run-once",
        choices=["run-once", "schedule"],
        help="Run the agent once or keep the scheduler running.",
    )
    parser.add_argument(
        "--tickers",
        help="Optional comma-separated tickers overriding TICKERS from the environment.",
    )
    return parser.parse_args()


def build_service(settings: Settings) -> FinanceAgentService:
    return FinanceAgentService(settings)


def main() -> int:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings)
    service = build_service(settings)

    try:
        tickers_override = (
            [item.strip().upper() for item in args.tickers.split(",") if item.strip()]
            if args.tickers
            else None
        )
        if args.mode == "schedule":
            scheduler = FinanceAgentScheduler(settings, job=lambda: service.run_once(tickers_override))
            scheduler.start()
            return 0

        service.run_once(tickers_override)
        return 0
    except Exception as exc:
        LOGGER.exception("Finance agent execution failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
