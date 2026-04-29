from __future__ import annotations

import logging
import time
from collections.abc import Callable
from zoneinfo import ZoneInfo

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from .config import Settings

LOGGER = logging.getLogger(__name__)


class FinanceAgentScheduler:
    def __init__(self, settings: Settings, job: Callable[[], None]) -> None:
        self.settings = settings
        self.job = job
        self.scheduler = BlockingScheduler(timezone=ZoneInfo(settings.timezone))

    def start(self) -> None:
        trigger = CronTrigger(
            hour=self.settings.run_hour,
            minute=self.settings.run_minute,
            timezone=ZoneInfo(self.settings.timezone),
        )
        self.scheduler.add_job(
            self._wrapped_job,
            trigger=trigger,
            id="daily-finance-agent",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=3600,
            replace_existing=True,
        )
        LOGGER.info(
            "Scheduler started. Daily run is set for %02d:%02d %s",
            self.settings.run_hour,
            self.settings.run_minute,
            self.settings.timezone,
        )
        self.scheduler.start()

    def _wrapped_job(self) -> None:
        LOGGER.info("Starting scheduled finance agent run.")
        start = time.perf_counter()
        self.job()
        elapsed = time.perf_counter() - start
        LOGGER.info("Scheduled finance agent run finished in %.2f seconds.", elapsed)
