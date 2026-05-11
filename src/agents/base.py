"""
Base agent class and specialized agent implementations.

Each agent:
- has a clear role with system prompts
- can use tools autonomously
- communicates with other agents
- returns structured outputs
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

from src.config import Settings
from src.llm import LLMProvider
from src.memory.manager import MemoryManager
from src.models import (
    AgentMessage, AgentRole, AgentState, AgentTask, MessageType,
    ReflectionResult, SubTask, TaskStatus, ToolCall, to_dict, utc_now_iso,
)
from src.reasoning.engine import ReasoningEngine
from src.tools.registry import ToolRegistry, ToolResult
from src.utils.helpers import json_dumps, parse_json_response, truncate_text

LOGGER = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base for all specialized agents."""

    def __init__(
        self,
        role: AgentRole,
        settings: Settings,
        llm: LLMProvider,
        memory: MemoryManager,
        tools: ToolRegistry,
        reasoning: ReasoningEngine,
    ) -> None:
        self.role = role
        self.settings = settings
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.reasoning = reasoning
        self.state = AgentState(role=role, tool_count=len(tools.list_tools()))
        self._message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._running = False

    @property
    @abstractmethod
    def system_prompt(self) -> str: ...

    @property
    @abstractmethod
    def capabilities(self) -> list[str]: ...

    async def start(self) -> None:
        self._running = True
        self.state.status = "running"
        self.state.last_active = utc_now_iso()
        LOGGER.info("Agent %s started", self.role.value)
        asyncio.create_task(self._message_loop())

    async def stop(self) -> None:
        self._running = False
        self.state.status = "stopped"

    async def _message_loop(self) -> None:
        while self._running:
            try:
                msg = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                await self._handle_message(msg)
            except asyncio.TimeoutError:
                continue

    async def _handle_message(self, msg: AgentMessage) -> None:
        LOGGER.debug("Agent %s received message from %s: %s", self.role.value, msg.from_agent, msg.message_type.value)

    async def send_message(self, to: AgentRole, msg_type: MessageType,
                            payload: dict[str, Any], task_id: str | None = None) -> None:
        msg = AgentMessage(
            from_agent=self.role, to_agent=to, message_type=msg_type,
            payload=payload, task_id=task_id, timestamp=utc_now_iso(),
        )
        await self.memory.store_conversation(self.role.value, to.value, str(payload)[:500])

    async def think(self, prompt: str, context: str = "", use_tools: bool = True) -> str:
        memory_context = await self.memory.retrieve_context_for_task(prompt)
        full_context = f"{context}\n{memory_context}" if memory_context else context

        tools_schema = self.tools.get_all_schemas() if use_tools else None
        system = self.system_prompt
        if tools_schema and use_tools:
            system += f"\n\nYou have access to these tools:\n{self.tools.get_tool_descriptions()}"

        result = await self.llm.generate(
            prompt=prompt, system_prompt=system, tools=tools_schema,
        )
        return result["content"]

    async def think_json(self, prompt: str) -> dict[str, Any]:
        return await self.llm.generate_json(prompt, system_prompt=self.system_prompt)

    async def use_tool(self, tool_name: str, **kwargs: Any) -> ToolResult:
        return await self.tools.execute(tool_name, **kwargs)

    async def process_subtask(self, subtask: SubTask) -> dict[str, Any]:
        """Core method: process a single subtask end-to-end."""
        self.state.current_task_id = subtask.id
        subtask.status = TaskStatus.IN_PROGRESS
        subtask.started_at = utc_now_iso()

        try:
            context = await self.memory.retrieve_context_for_task(subtask.objective)
            prompt = (
                f"Task: {subtask.objective}\nDescription: {subtask.description}\n"
                f"Available Tools: {self.tools.get_tool_descriptions()}\n"
                f"Relevant Context: {context}\n\n"
                "Complete this task. You may use tools. When done, output your result."
            )
            result = await self.think(prompt, context)
            subtask.result = result
            subtask.status = TaskStatus.COMPLETED
            subtask.completed_at = utc_now_iso()
            self.state.tasks_completed += 1
        except Exception as exc:
            subtask.status = TaskStatus.FAILED
            subtask.error = str(exc)
            self.state.error_count += 1
            LOGGER.error("Agent %s failed subtask %s: %s", self.role.value, subtask.id, exc)

        return to_dict(subtask)

    @abstractmethod
    async def execute(self, task: AgentTask) -> dict[str, Any]: ...


# --- Planner Agent ---

class PlannerAgent(BaseAgent):
    role = AgentRole.PLANNER
    capabilities = ["goal_decomposition", "task_prioritization", "dependency_analysis", "plan_optimization"]

    @property
    def system_prompt(self) -> str:
        return (
            "You are an expert planning agent. Your role is to:\n"
            "1. Break complex objectives into concrete, executable subtasks\n"
            "2. Identify dependencies between subtasks\n"
            "3. Prioritize tasks based on importance and urgency\n"
            "4. Assign appropriate agents/tools to each subtask\n"
            "5. Validate plans for completeness and feasibility\n\n"
            "Always return structured JSON. Be thorough and realistic."
        )

    async def execute(self, task: AgentTask) -> dict[str, Any]:
        plan = await self.reasoning.plan_execute(task.objective, task.context, max_subtasks=15)
        subtasks = []
        for item in plan:
            st = SubTask(
                id=item.get("id", ""), objective=item.get("description", ""),
                description=item.get("success_criteria", ""),
                dependencies=item.get("dependencies", []),
                parent_task_id=task.id, created_at=utc_now_iso(),
            )
            subtasks.append(st)

        task.subtasks = subtasks
        task.total_subtasks = len(subtasks)
        task.status = TaskStatus.PLANNING

        await self.memory.store_episodic(task.id, {"phase": "planning", "subtask_count": len(subtasks)})
        await self.send_message(AgentRole.ORCHESTRATOR, MessageType.RESULT,
                                {"phase": "planning", "subtask_count": len(subtasks)}, task.id)
        return to_dict(task)


# --- Researcher Agent ---

class ResearcherAgent(BaseAgent):
    role = AgentRole.RESEARCHER
    capabilities = ["web_search", "web_fetch", "information_synthesis", "fact_verification", "source_citation"]

    @property
    def system_prompt(self) -> str:
        return (
            "You are an expert research agent. Your role is to:\n"
            "1. Search the web for relevant, up-to-date information\n"
            "2. Fetch and analyze web content\n"
            "3. Synthesize findings into clear, structured summaries\n"
            "4. Verify facts across multiple sources\n"
            "5. Cite all sources properly\n\n"
            "Always distinguish between facts and opinions. Flag unverified claims."
        )

    async def execute(self, task: AgentTask) -> dict[str, Any]:
        findings = []
        for subtask in task.subtasks:
            if subtask.assigned_agent == AgentRole.RESEARCHER:
                search_result = await self.use_tool("web_search", query=subtask.objective, num_results=5)
                if search_result.success:
                    findings.append({"subtask": subtask.id, "results": search_result.output})
                subtask.result = search_result.output if search_result.success else search_result.error

        task.result_summary = json_dumps(findings)
        return to_dict(task)


# --- Finance Analyst Agent ---

class FinanceAnalystAgent(BaseAgent):
    role = AgentRole.FINANCE_ANALYST
    capabilities = ["stock_analysis", "news_sentiment", "market_research", "portfolio_insight",
                    "risk_scoring", "earnings_analysis", "sec_filings"]

    @property
    def system_prompt(self) -> str:
        return (
            "You are an expert financial analyst. Your role is to:\n"
            "1. Analyze stock performance and fundamentals\n"
            "2. Assess market sentiment from news\n"
            "3. Evaluate risk factors and opportunities\n"
            "4. Generate investment insights\n"
            "5. Create professional financial reports\n\n"
            "Be data-driven, balanced, and professional. Always cite data sources."
        )

    async def execute(self, task: AgentTask) -> dict[str, Any]:
        analyses = []
        for subtask in task.subtasks:
            if subtask.assigned_agent == AgentRole.FINANCE_ANALYST:
                tool_name = "finance"
                for ticker in self.settings.tickers:
                    result = await self.use_tool(tool_name, action="stock_info", symbol=ticker)
                    if result.success:
                        analyses.append({"ticker": ticker, "info": result.output})

                    news_result = await self.use_tool(tool_name, action="news", symbol=ticker)
                    if news_result.success:
                        analyses.append({"ticker": ticker, "news": news_result.output})

        task.result_summary = json_dumps(analyses)
        return to_dict(task)


# --- Coding Agent ---

class CoderAgent(BaseAgent):
    role = AgentRole.CODER
    capabilities = ["code_generation", "code_refactoring", "debugging", "testing",
                    "code_review", "repository_analysis", "pr_creation"]

    @property
    def system_prompt(self) -> str:
        return (
            "You are an expert software engineer and coding agent. Your role is to:\n"
            "1. Write clean, modular, production-ready code\n"
            "2. Refactor existing code for quality and performance\n"
            "3. Debug and fix issues systematically\n"
            "4. Write and run tests\n"
            "5. Analyze repositories and suggest improvements\n\n"
            "Follow SOLID principles. Use type hints. Write self-documenting code. "
            "Never generate placeholder implementations."
        )

    async def execute(self, task: AgentTask) -> dict[str, Any]:
        for subtask in task.subtasks:
            if subtask.assigned_agent == AgentRole.CODER:
                code = await self.reasoning.chain_of_thought(subtask.objective, task.context)
                subtask.result = {"plan": code}

                if "generate" in subtask.objective.lower() or "write" in subtask.objective.lower():
                    prompt = (
                        f"Generate production-ready Python code for: {subtask.objective}\n"
                        f"Context: {task.context}\n\n"
                        "Return the complete code. Use type hints, docstrings, and proper error handling."
                    )
                    result = await self.llm.generate(prompt, self.system_prompt)
                    code_blocks = "\n".join([
                        r for r in [result["content"]] if r
                    ])
                    subtask.result = {"code": code_blocks}

        return to_dict(task)


# --- Debugger Agent ---

class DebuggerAgent(BaseAgent):
    role = AgentRole.DEBUGGER
    capabilities = ["error_analysis", "root_cause_identification", "fix_generation",
                    "log_analysis", "stack_trace_parsing"]

    @property
    def system_prompt(self) -> str:
        return (
            "You are an expert debugging agent. Your role is to:\n"
            "1. Analyze error messages and stack traces\n"
            "2. Identify root causes systematically\n"
            "3. Generate targeted fixes\n"
            "4. Verify fixes don't introduce regressions\n"
            "5. Log analysis for patterns\n\n"
            "Be methodical. Eliminate one cause at a time. Verify before declaring fixed."
        )

    async def execute(self, task: AgentTask) -> dict[str, Any]:
        for subtask in task.subtasks:
            if subtask.assigned_agent == AgentRole.DEBUGGER:
                similar_errors = await self.memory.retrieve_similar_errors(subtask.objective)
                context = task.context + f"\nSimilar past errors: {json_dumps(similar_errors)}"
                analysis = await self.reasoning.chain_of_thought(
                    f"Debug this issue: {subtask.objective}", context
                )
                subtask.result = {"debug_analysis": analysis, "similar_errors": similar_errors}

        return to_dict(task)


# --- Reflection / Critic Agent ---

class ReflectorAgent(BaseAgent):
    role = AgentRole.REFLECTOR
    capabilities = ["self_reflection", "execution_evaluation", "improvement_suggestions",
                    "error_pattern_analysis", "strategy_optimization"]

    @property
    def system_prompt(self) -> str:
        return (
            "You are an expert reflection and critique agent. Your role is to:\n"
            "1. Evaluate task execution objectively\n"
            "2. Identify what worked and what didn't\n"
            "3. Detect patterns in failures\n"
            "4. Suggest concrete improvements\n"
            "5. Recommend retry strategies\n\n"
            "Be brutally honest. Sugarcoating helps no one. But be constructive."
        )

    async def execute(self, task: AgentTask) -> dict[str, Any]:
        reflection = await self.reasoning.reflect(
            task=task,
            execution_result={"result_summary": task.result_summary, "subtasks": to_dict(task.subtasks)},
            errors=[e.get("error", "") for e in task.errors],
        )
        await self.memory.store_long_term(
            content=f"Reflection for {task.id}: {reflection.analysis}",
            entry_type="reflection",
            metadata=to_dict(reflection),
        )
        return to_dict(reflection)


# --- Report Generator Agent ---

class ReportGeneratorAgent(BaseAgent):
    role = AgentRole.REPORT_GENERATOR
    capabilities = ["report_generation", "data_visualization", "summary_synthesis", "formatting"]

    @property
    def system_prompt(self) -> str:
        return (
            "You are an expert report generation agent. Your role is to:\n"
            "1. Synthesize findings from all agents into cohesive reports\n"
            "2. Create structured, professional reports with clear sections\n"
            "3. Include data visualizations and key metrics\n"
            "4. Write actionable executive summaries\n"
            "5. Format output for intended audience\n\n"
            "Reports should be comprehensive yet concise. Use markdown formatting."
        )

    async def execute(self, task: AgentTask) -> dict[str, Any]:
        report = (
            f"# {task.objective}\n\n"
            f"## Executive Summary\n{task.result_summary}\n\n"
            f"## Task Details\n"
            f"- Total Subtasks: {task.total_subtasks}\n"
            f"- Completed: {task.completed_subtasks}\n"
            f"- Status: {task.status.value}\n\n"
            f"## Subtask Results\n"
        )
        for st in task.subtasks:
            report += f"### {st.objective}\n- Status: {st.status.value}\n- Result: {str(st.result)[:500]}\n\n"

        if task.errors:
            report += f"## Errors Encountered\n"
            for e in task.errors:
                report += f"- {e.get('error', 'Unknown error')}\n"

        task.final_output = report
        return to_dict(task)