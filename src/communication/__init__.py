"""
Multi-agent communication system: Event Bus for pub/sub and Task Queue for async work.

Supports:
- Structured JSON message passing
- Shared memory access
- Publish/subscribe event bus
- Priority-based task queue
- Parallel execution
- Asynchronous workflows
- Distributed agent communication
"""

from __future__ import annotations

import asyncio
import heapq
import logging
from collections import defaultdict
from typing import Any, Callable

from src.config import Settings
from src.models import AgentMessage, MessageType, utc_now_iso

LOGGER = logging.getLogger(__name__)


class EventBus:
    """Publish/subscribe event bus for inter-agent communication."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._history: list[AgentMessage] = []

    def subscribe(self, event_type: str, callback: Callable) -> None:
        self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(callback)

    async def publish(self, message: AgentMessage) -> None:
        self._history.append(message)
        event_type = message.message_type.value
        subscribers = self._subscribers.get(event_type, []) + \
                      self._subscribers.get("*", [])

        for callback in subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)
            except Exception as exc:
                LOGGER.error("Event handler failed for %s: %s", event_type, exc)

    def recent_messages(self, n: int = 20) -> list[AgentMessage]:
        return self._history[-n:]

    def clear_history(self) -> None:
        self._history.clear()


class PriorityQueue:
    """Min-heap priority queue for task scheduling."""

    def __init__(self) -> None:
        self._heap: list[tuple[int, int, Any]] = []
        self._counter = 0

    def push(self, item: Any, priority: int = 3) -> None:
        heapq.heappush(self._heap, (priority, self._counter, item))
        self._counter += 1

    def pop(self) -> Any | None:
        if self._heap:
            return heapq.heappop(self._heap)[2]
        return None

    def peek(self) -> Any | None:
        if self._heap:
            return self._heap[0][2]
        return None

    @property
    def size(self) -> int:
        return len(self._heap)

    def is_empty(self) -> bool:
        return len(self._heap) == 0


class TaskQueue:
    """Async task queue backed by Redis/Celery or in-memory for development."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._queue = PriorityQueue()
        self._results: dict[str, Any] = {}
        self._use_redis = bool(settings.redis_url)
        self._celery_app = None

        if self._use_redis:
            try:
                from celery import Celery
                self._celery_app = Celery("ai_agent")
                self._celery_app.conf.update(
                    broker_url=settings.celery_broker_url,
                    result_backend=settings.celery_result_backend,
                    task_serializer="json",
                    accept_content=["json"],
                    result_serializer="json",
                    timezone="UTC",
                    enable_utc=True,
                )
            except ImportError:
                self._use_redis = False

    async def enqueue(self, func_name: str, args: tuple = (), kwargs: dict | None = None,
                       priority: int = 3) -> str:
        task_id = f"{func_name}_{utc_now_iso()}"
        if self._use_redis and self._celery_app:
            self._celery_app.send_task(func_name, args=args, kwargs=kwargs or {},
                                       queue=f"priority_{priority}")
        else:
            self._queue.push({"id": task_id, "func": func_name, "args": args, "kwargs": kwargs or {}}, priority)
        return task_id

    async def dequeue(self) -> dict[str, Any] | None:
        return self._queue.pop()

    async def store_result(self, task_id: str, result: Any) -> None:
        self._results[task_id] = result

    async def get_result(self, task_id: str) -> Any:
        return self._results.get(task_id)

    def pending_count(self) -> int:
        return self._queue.size