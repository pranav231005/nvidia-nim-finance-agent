#!/usr/bin/env python3
"""Main entry point for the Autonomous AI Agent System."""

from __future__ import annotations

import argparse
import asyncio
import sys

from src.config import get_settings
from src.models import Priority
from src.observability import configure_observability
from src.orchestration.engine import AgentOrchestrator


async def run_autonomous(objective: str, context: str = "", priority: str = "medium") -> None:
    settings = get_settings()
    configure_observability(settings)
    orchestrator = AgentOrchestrator(settings)

    prio = Priority[priority.upper()] if priority.upper() in Priority.__members__ else Priority.MEDIUM

    task = await orchestrator.execute(objective=objective, context=context, priority=prio)

    print("\n" + "=" * 60)
    print(f"TASK COMPLETE: {task.status.value}")
    print(f"Task ID: {task.id}")
    print(f"Subtasks: {task.completed_subtasks}/{task.total_subtasks}")
    print(f"Tokens: {task.total_tokens_used} | Cost: ${task.total_cost_usd:.4f}")
    print(f"Result: {task.result_summary[:1000]}")
    if task.final_output:
        print(f"\nFinal Report:\n{str(task.final_output)[:2000]}")
    if task.errors:
        print(f"\nErrors: {len(task.errors)}")
        for e in task.errors[:5]:
            print(f"  - {e.get('error', '')}")
    print("=" * 60)


def run_api() -> None:
    from src.api.server import run
    run()


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous AI Agent System")
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Execute an autonomous objective")
    run_parser.add_argument("objective", help="The objective to accomplish")
    run_parser.add_argument("--context", default="", help="Additional context")
    run_parser.add_argument("--priority", default="medium", choices=["critical", "high", "medium", "low"])

    sub.add_parser("api", help="Start the API server")
    sub.add_parser("status", help="Show system status")

    args = parser.parse_args()

    if args.command == "run":
        asyncio.run(run_autonomous(args.objective, args.context, args.priority))
    elif args.command == "api":
        run_api()
    elif args.command == "status":
        settings = get_settings()
        print(f"App: {settings.app_name}")
        print(f"Environment: {settings.environment}")
        print(f"Model: {settings.openai_model}")
        print(f"Memory: {settings.chroma_persist_directory}")
        print(f"Tools enabled: file_read, file_write, shell_exec, python_exec, web_search, web_fetch, github, finance")
    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())