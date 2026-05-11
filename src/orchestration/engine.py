"""
Autonomous Agent Orchestrator — the central nervous system.

Implements the full autonomous execution loop:
1. Receive objective
2. Analyze & plan (PlannerAgent)
3. Break into subtasks
4. Assign to specialized agents
5. Execute with tool usage
6. Evaluate results (ReflectorAgent)
7. Retry on failure
8. Continue until completion
9. Produce final report

Supports: recursive planning, dynamic task creation, interruption recovery,
context persistence, parallel execution, and multi-agent collaboration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.agents.base import (
    BaseAgent, CoderAgent, DebuggerAgent, FinanceAnalystAgent,
    PlannerAgent, ReflectorAgent, ReportGeneratorAgent, ResearcherAgent,
)
from src.communication import EventBus, TaskQueue
from src.config import Settings
from src.llm import LLMProvider
from src.memory.manager import MemoryManager
from src.models import (
    AgentMessage, AgentRole, AgentTask, ExecutionContext, MessageType,
    Priority, ReflectionResult, SubTask, TaskStatus, ToolCall, to_dict, utc_now_iso,
)
from src.observability import Telemetry
from src.reasoning.engine import ReasoningEngine
from src.safety import Guardrails
from src.tools.registry import ToolRegistry
from src.utils.helpers import ensure_directory, json_dumps, json_loads

LOGGER = logging.getLogger(__name__)


class AgentOrchestrator:
    """Central orchestrator managing the full autonomous agent lifecycle."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm = LLMProvider(settings)
        self.memory = MemoryManager(settings, self.llm)
        self.tools = ToolRegistry.create_default_registry(settings)
        self.reasoning = ReasoningEngine(settings, self.llm)
        self.event_bus = EventBus()
        self.task_queue = TaskQueue(settings)
        self.guardrails = Guardrails(settings)
        self.telemetry = Telemetry(settings)

        self._agents: dict[AgentRole, BaseAgent] = {}
        self._init_agents()

        self._active_tasks: dict[str, ExecutionContext] = {}
        self._task_history: list[AgentTask] = []
        self._checkpoint_file = settings.artifacts_dir / "checkpoints.json"

    def _init_agents(self) -> None:
        agent_classes: dict[AgentRole, type[BaseAgent]] = {
            AgentRole.PLANNER: PlannerAgent,
            AgentRole.RESEARCHER: ResearcherAgent,
            AgentRole.FINANCE_ANALYST: FinanceAnalystAgent,
            AgentRole.CODER: CoderAgent,
            AgentRole.DEBUGGER: DebuggerAgent,
            AgentRole.REFLECTOR: ReflectorAgent,
            AgentRole.REPORT_GENERATOR: ReportGeneratorAgent,
        }
        for role, cls in agent_classes.items():
            self._agents[role] = cls(
                role=role, settings=self.settings, llm=self.llm,
                memory=self.memory, tools=self.tools, reasoning=self.reasoning,
            )

    def get_agent(self, role: AgentRole) -> BaseAgent | None:
        return self._agents.get(role)

    # --- Autonomous Execution Loop ---

    async def execute(self, objective: str, context: str = "",
                       priority: Priority = Priority.MEDIUM,
                       max_steps: int | None = None) -> AgentTask:
        """
        Run the full autonomous execution loop for a given objective.
        This is the main entry point — it thinks, plans, executes, and iterates.
        """
        max_steps = max_steps or self.settings.max_autonomous_steps
        task = AgentTask(
            objective=objective, description=context[:5000], context=context,
            priority=priority, created_at=utc_now_iso(),
        )
        ctx = ExecutionContext(task=task, total_steps=max_steps)

        LOGGER.info("=" * 60)
        LOGGER.info("ORCHESTRATOR: Starting autonomous execution")
        LOGGER.info("Objective: %s", objective[:200])
        LOGGER.info("Max Steps: %d | Priority: %s", max_steps, priority.name)
        LOGGER.info("=" * 60)

        # Phase 1: Analyze & Plan
        await self._phase_plan(task, ctx)
        if task.status == TaskStatus.FAILED:
            return task

        # Phase 2: Autonomous Loop — Execute, Evaluate, Iterate
        while ctx.step_count < max_steps and task.status not in (
            TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED,
        ):
            ctx.step_count += 1
            ctx.recursion_depth += 1

            LOGGER.info("--- Step %d/%d ---", ctx.step_count, max_steps)

            # Check cost/token limits
            if ctx.cost_usd >= self.settings.max_cost_per_session_usd:
                LOGGER.warning("Cost limit reached: $%.2f", ctx.cost_usd)
                await self._handle_limit_reached(task, "cost")
                break
            if ctx.token_count >= self.settings.max_tokens_per_session:
                LOGGER.warning("Token limit reached: %d", ctx.token_count)
                await self._handle_limit_reached(task, "token")
                break

            # Save checkpoint
            await self._save_checkpoint(task, ctx)

            # Execute pending subtasks
            await self._phase_execute(task, ctx)

            # Evaluate results
            await self._phase_evaluate(task, ctx)

            # Reflect on failures
            if task.errors:
                await self._phase_reflect(task, ctx)

            # Check if done
            if self._all_subtasks_resolved(task):
                task.status = TaskStatus.COMPLETED
                break

            # Dynamic replanning if stuck
            if self._is_stuck(task, ctx):
                await self._phase_replan(task, ctx)

        # Phase 3: Finalize
        await self._phase_finalize(task, ctx)
        self._task_history.append(task)
        await self.memory.store_episodic(task.id, to_dict(task))

        LOGGER.info("=" * 60)
        LOGGER.info("ORCHESTRATOR: Execution complete — Status: %s", task.status.value)
        LOGGER.info("Steps: %d | Tokens: %d | Cost: $%.4f", ctx.step_count, ctx.token_count, ctx.cost_usd)
        LOGGER.info("=" * 60)

        return task

    async def _phase_plan(self, task: AgentTask, ctx: ExecutionContext) -> None:
        """Phase 1: Analyze objective and create execution plan."""
        LOGGER.info("PHASE 1: Planning")
        task.status = TaskStatus.PLANNING

        planner = self.get_agent(AgentRole.PLANNER)
        if not planner:
            task.status = TaskStatus.FAILED
            task.errors.append({"error": "Planner agent not available"})
            return

        await planner.execute(task)

        # Assign subtasks to appropriate agents
        for subtask in task.subtasks:
            assigned = self._assign_agent(subtask)
            subtask.assigned_agent = assigned

        # Add to memory
        await self.memory.store_short_term("plan_created", {
            "task_id": task.id, "subtask_count": task.total_subtasks,
            "assignments": {st.id: st.assigned_agent.value if st.assigned_agent else None for st in task.subtasks},
        })

        self.telemetry.record_event("plan_phase_complete", {"subtask_count": task.total_subtasks})

    async def _phase_execute(self, task: AgentTask, ctx: ExecutionContext) -> None:
        """Phase 2: Execute pending subtasks."""
        task.status = TaskStatus.IN_PROGRESS

        pending = [st for st in task.subtasks if st.status in (
            TaskStatus.PENDING, TaskStatus.RETRYING,
        ) and self._dependencies_satisfied(st, task)]

        if not pending:
            return

        # Execute subtasks (with parallel support for independent tasks)
        independent = [st for st in pending if not st.dependencies]
        dependent = [st for st in pending if st.dependencies]

        # Fire parallel execution for independent subtasks
        if independent:
            results = await asyncio.gather(
                *[self._execute_subtask(st, task, ctx) for st in independent],
                return_exceptions=True,
            )
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    independent[i].status = TaskStatus.FAILED
                    independent[i].error = str(result)

        # Execute dependent subtasks sequentially
        for st in dependent:
            if self._dependencies_satisfied(st, task):
                await self._execute_subtask(st, task, ctx)

    async def _execute_subtask(self, subtask: SubTask, task: AgentTask, ctx: ExecutionContext) -> None:
        """Execute a single subtask — agent loop with tool calls."""
        subtask.status = TaskStatus.IN_PROGRESS
        subtask.started_at = utc_now_iso()
        subtask.attempts += 1

        agent = self.get_agent(subtask.assigned_agent) if subtask.assigned_agent else None
        if not agent:
            agent = self.get_agent(AgentRole.RESEARCHER)

        LOGGER.info("Executing subtask: %s [%s] (attempt %d/%d)",
                     subtask.id[:8], subtask.assigned_agent.value if subtask.assigned_agent else "N/A",
                     subtask.attempts, subtask.max_attempts)

        try:
            # Safety check
            if not self.guardrails.check_tool_permission(subtask.objective, self.settings):
                subtask.status = TaskStatus.WAITING
                task.human_approvals.append({"subtask_id": subtask.id, "objective": subtask.objective})
                LOGGER.info("Subtask %s requires human approval", subtask.id[:8])
                return

            # Agent handles the subtask with tool access
            result = await agent.process_subtask(subtask)

            # Track tokens/cost
            ctx.token_count += result.get("tokens", 0) if isinstance(result, dict) else 0
            ctx.cost_usd += self._estimate_cost(ctx.token_count)

            # Verify result
            if subtask.status == TaskStatus.COMPLETED:
                verified = await self._verify_subtask_result(subtask)
                if not verified:
                    subtask.status = TaskStatus.FAILED
                    subtask.error = "Result verification failed"

            task.completed_subtasks = sum(1 for st in task.subtasks if st.status == TaskStatus.COMPLETED)

        except Exception as exc:
            subtask.status = TaskStatus.FAILED
            subtask.error = str(exc)
            task.errors.append({"subtask_id": subtask.id, "error": str(exc)})
            await self.memory.store_error("subtask_execution", str(exc), {"subtask_id": subtask.id})
            LOGGER.error("Subtask %s failed: %s", subtask.id[:8], exc)

        ctx.recursion_depth += 1

    async def _verify_subtask_result(self, subtask: SubTask) -> bool:
        """Verify subtask output quality using the critic."""
        if not subtask.result:
            return False

        critic = self.get_agent(AgentRole.REFLECTOR)

        try:
            result_str = str(subtask.result)[:2000]
            prompt = (
                f"Evaluate if this subtask was successfully completed:\n"
                f"Objective: {subtask.objective}\n"
                f"Result: {result_str}\n\n"
                'Return JSON: {"completed": true/false, "quality": 0-100, "reason": "..."}'
            )
            verdict = await self.llm.generate_json(prompt)
            return verdict.get("completed", True)
        except Exception:
            return subtask.result is not None

    async def _phase_evaluate(self, task: AgentTask, ctx: ExecutionContext) -> None:
        """Phase 3: Evaluate overall progress."""
        completed = sum(1 for st in task.subtasks if st.status == TaskStatus.COMPLETED)
        failed = sum(1 for st in task.subtasks if st.status == TaskStatus.FAILED)
        pending = sum(1 for st in task.subtasks if st.status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS))

        LOGGER.info("Progress: %d done | %d failed | %d pending", completed, failed, pending)

        task.completed_subtasks = completed

        if completed == task.total_subtasks:
            task.status = TaskStatus.COMPLETED
        elif completed + failed >= task.total_subtasks and pending == 0:
            if completed > 0:
                task.status = TaskStatus.COMPLETED
            else:
                task.status = TaskStatus.FAILED

        self.telemetry.record_event("evaluation", {
            "task_id": task.id, "completed": completed, "failed": failed, "pending": pending,
        })

    async def _phase_reflect(self, task: AgentTask, ctx: ExecutionContext) -> None:
        """Phase 4: Reflect on failures and retry if appropriate."""
        task.status = TaskStatus.REFLECTING

        reflector = self.get_agent(AgentRole.REFLECTOR)
        if not reflector:
            return

        reflection = await self.reasoning.reflect(
            task=task,
            execution_result={"completed": task.completed_subtasks, "total": task.total_subtasks},
            errors=[e.get("error", "") for e in task.errors],
        )

        await self.memory.store_long_term(
            content=f"Reflection: {reflection.analysis}", entry_type="reflection",
            metadata=to_dict(reflection),
        )

        if reflection.should_retry:
            for subtask in task.subtasks:
                if subtask.status == TaskStatus.FAILED and subtask.attempts < subtask.max_attempts:
                    subtask.status = TaskStatus.RETRYING
                    LOGGER.info("Retrying subtask %s with strategy: %s", subtask.id[:8], reflection.retry_strategy)

    async def _phase_replan(self, task: AgentTask, ctx: ExecutionContext) -> None:
        """Phase 5: Dynamic replanning when execution is stuck."""
        if ctx.recursion_depth > self.settings.recursion_depth_limit:
            LOGGER.warning("Recursion depth limit reached, forcing completion")
            task.status = TaskStatus.COMPLETED
            return

        LOGGER.info("Replanning — current approach may be stuck")
        planner = self.get_agent(AgentRole.PLANNER)
        if planner:
            await planner.execute(task)
            for st in task.subtasks:
                if not st.assigned_agent:
                    st.assigned_agent = self._assign_agent(st)

    async def _phase_finalize(self, task: AgentTask, ctx: ExecutionContext) -> None:
        """Phase 6: Generate final report and cleanup."""
        task.completed_at = utc_now_iso()
        task.total_tokens_used = ctx.token_count
        task.total_cost_usd = ctx.cost_usd

        reporter = self.get_agent(AgentRole.REPORT_GENERATOR)
        if reporter:
            await reporter.execute(task)

        # Consolidate memory
        await self.memory.consolidate()

        self.telemetry.record_event("task_complete", {
            "task_id": task.id, "status": task.status.value,
            "steps": ctx.step_count, "cost": ctx.cost_usd,
        })

    # --- Helpers ---

    def _assign_agent(self, subtask: SubTask) -> AgentRole:
        """Route subtask to the most appropriate agent based on content."""
        obj = subtask.objective.lower()
        keywords = {
            AgentRole.CODER: ["code", "implement", "refactor", "test", "build", "program", "develop"],
            AgentRole.RESEARCHER: ["search", "research", "find", "lookup", "fetch", "analyze web"],
            AgentRole.FINANCE_ANALYST: ["stock", "finance", "market", "portfolio", "ticker", "earnings", "risk"],
            AgentRole.DEBUGGER: ["debug", "fix", "error", "bug", "issue", "trace", "crash"],
            AgentRole.PLANNER: ["plan", "strategy", "roadmap", "architecture", "design"],
            AgentRole.REPORT_GENERATOR: ["report", "summary", "document", "present"],
        }
        scores: dict[AgentRole, int] = {}
        for role, kws in keywords.items():
            scores[role] = sum(1 for kw in kws if kw in obj)
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else AgentRole.RESEARCHER

    def _dependencies_satisfied(self, subtask: SubTask, task: AgentTask) -> bool:
        if not subtask.dependencies:
            return True
        for dep_id in subtask.dependencies:
            dep = next((st for st in task.subtasks if st.id == dep_id), None)
            if dep is None or dep.status != TaskStatus.COMPLETED:
                return False
        return True

    def _all_subtasks_resolved(self, task: AgentTask) -> bool:
        if not task.subtasks:
            return task.completed_subtasks > 0
        return all(st.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
                   for st in task.subtasks)

    def _is_stuck(self, task: AgentTask, ctx: ExecutionContext) -> bool:
        """Detect if execution is stuck in a loop."""
        if ctx.step_count < 3:
            return False
        recent_failures = sum(1 for st in task.subtasks[-5:] if st.status == TaskStatus.FAILED) if len(task.subtasks) >= 5 else 0
        return recent_failures >= 3 and ctx.recursion_depth > 3

    async def _handle_limit_reached(self, task: AgentTask, limit_type: str) -> None:
        task.status = TaskStatus.COMPLETED if task.completed_subtasks > 0 else TaskStatus.FAILED
        task.errors.append({"error": f"{limit_type} limit reached"})

    async def _save_checkpoint(self, task: AgentTask, ctx: ExecutionContext) -> None:
        if not self.settings.interruption_recovery_enabled:
            return
        try:
            data = {
                "task": to_dict(task),
                "step_count": ctx.step_count,
                "token_count": ctx.token_count,
                "cost_usd": ctx.cost_usd,
                "saved_at": utc_now_iso(),
            }
            ensure_directory(self._checkpoint_file.parent)
            self._checkpoint_file.write_text(json_dumps(data))
        except Exception as exc:
            LOGGER.warning("Failed to save checkpoint: %s", exc)

    async def recover_from_checkpoint(self) -> AgentTask | None:
        """Recover from a saved checkpoint after interruption."""
        if not self._checkpoint_file.exists():
            return None
        try:
            data = json_loads(self._checkpoint_file.read_text())
            LOGGER.info("Recovered checkpoint from %s", data.get("saved_at"))
            from dataclasses import asdict as dc_asdict
            task_data = data["task"]
            return task_data  # Simplified; full recovery would reconstruct objects
        except Exception as exc:
            LOGGER.error("Failed to recover checkpoint: %s", exc)
            return None

    def _estimate_cost(self, tokens: int) -> float:
        # Rough cost estimate based on GPT-4 pricing
        return (tokens / 1_000_000) * 10  # ~$10/1M tokens

    def get_status(self) -> dict[str, Any]:
        return {
            "active_tasks": len(self._active_tasks),
            "completed_tasks": len(self._task_history),
            "agents": {
                role.value: {
                    "status": agent.state.status,
                    "tasks_completed": agent.state.tasks_completed,
                    "errors": agent.state.error_count,
                }
                for role, agent in self._agents.items()
            },
            "memory": {
                "short_term_size": self.memory.short_term.size,
                "long_term_entries": self.memory.vector.collection_count("long_term"),
            },
            "session_cost_usd": sum(t.total_cost_usd for t in self._task_history),
        }