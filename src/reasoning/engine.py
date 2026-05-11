"""
Advanced reasoning frameworks for autonomous agent decision-making.

Supports:
- Chain of Thought (CoT)
- ReAct (Reasoning + Acting)
- Tree of Thoughts (ToT)
- Reflexion (Self-reflection and improvement)
- Plan-Execute-Verify
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import Settings
from src.llm import LLMProvider
from src.models import AgentTask, ReflectionResult, SubTask, TaskStatus, to_dict
from src.utils.helpers import json_dumps, parse_json_response
from src.models import utc_now_iso

LOGGER = logging.getLogger(__name__)


class ReasoningEngine:
    """Provides multiple reasoning strategies for autonomous agents."""

    def __init__(self, settings: Settings, llm: LLMProvider) -> None:
        self.settings = settings
        self.llm = llm
        self.reasoning_history: list[dict[str, Any]] = []

    async def chain_of_thought(self, question: str, context: str = "") -> list[str]:
        prompt = (
            "You are an analytical reasoning engine. Think step by step.\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            "Break down your reasoning into distinct steps. Return your answer as a JSON object "
            'with a "steps" array of strings, each representing one reasoning step.\n'
            'Format: {"steps": ["Step 1: ...", "Step 2: ...", ...]}'
        )
        result = await self.llm.generate_json(prompt, system_prompt="Think step by step.")
        return result.get("steps", [])

    async def react(
        self, task: str, available_tools: list[str], context: str = "", max_cycles: int = 10,
    ) -> list[dict[str, Any]]:
        prompt = (
            "You are a ReAct (Reasoning + Acting) agent. For the given task, produce a "
            "sequence of Thought-Action-Observation cycles.\n\n"
            f"Available Tools: {', '.join(available_tools)}\n"
            f"Context: {context}\n"
            f"Task: {task}\n\n"
            "Return your plan as JSON with a 'cycles' array of objects:\n"
            '{"cycles": [{"thought": "...", "action": "tool_name", "action_input": {...}, '
            '"expected_observation": "..."}]}\n'
            f"Limit to {max_cycles} cycles."
        )
        result = await self.llm.generate_json(prompt, system_prompt="Use ReAct reasoning.")
        return result.get("cycles", [])

    async def tree_of_thoughts(
        self, problem: str, context: str = "", num_branches: int = 3, max_depth: int = 3,
    ) -> dict[str, Any]:
        prompt = (
            "You are a Tree-of-Thoughts reasoning engine. For the given problem, explore "
            f"{num_branches} different reasoning paths up to depth {max_depth}.\n\n"
            f"Context: {context}\n"
            f"Problem: {problem}\n\n"
            "Return JSON with a 'tree' object containing 'root' with 'branches' array. "
            "Each branch has 'thought', 'evaluation' (0-10), and optional 'children' array."
        )
        result = await self.llm.generate_json(
            prompt, system_prompt="Explore multiple reasoning paths simultaneously."
        )
        return result

    async def reflect(
        self, task: AgentTask, execution_result: dict[str, Any], errors: list[str],
    ) -> ReflectionResult:
        prompt = (
            "You are a self-reflection engine. Analyze the execution of this task "
            "and determine what went wrong or could be improved.\n\n"
            f"Objective: {task.objective}\n"
            f"Status: {task.status.value}\n"
            f"Result: {json_dumps(execution_result)}\n"
            f"Errors: {json_dumps(errors)}\n\n"
            "Return JSON with these fields: success (bool), analysis (string), "
            "root_cause (string), lessons_learned (string array), "
            "improvement_suggestions (string array), should_retry (bool), "
            'retry_strategy (string), confidence (0.0-1.0)'
        )
        result = await self.llm.generate_json(
            prompt, system_prompt="Be honest and analytical in your self-reflection."
        )
        return ReflectionResult(
            success=result.get("success", True),
            analysis=result.get("analysis", ""),
            root_cause=result.get("root_cause", ""),
            lessons_learned=result.get("lessons_learned", []),
            improvement_suggestions=result.get("improvement_suggestions", []),
            should_retry=result.get("should_retry", False),
            retry_strategy=result.get("retry_strategy", ""),
            confidence=result.get("confidence", 1.0),
        )

    async def plan_execute(self, objective: str, context: str = "", max_subtasks: int = 10) -> list[dict[str, Any]]:
        prompt = (
            "You are a hierarchical planning engine. Break down the given objective into "
            "concrete, executable subtasks.\n\n"
            f"Context: {context}\n"
            f"Objective: {objective}\n\n"
            "Return JSON with a 'plan' array of subtask objects:\n"
            '{"plan": [{"id": "1", "description": "...", "dependencies": [], '
            '"estimated_tool": "tool_name", "success_criteria": "..."}]}\n'
            f"Create at most {max_subtasks} subtasks."
        )
        result = await self.llm.generate_json(
            prompt, system_prompt="Create actionable, dependency-aware plans."
        )
        return result.get("plan", [])

    async def self_critique(self, plan_or_output: str, criteria: list[str]) -> dict[str, Any]:
        prompt = (
            "You are a self-critique engine. Evaluate the following content "
            "against the given criteria and identify issues.\n\n"
            f"Content: {plan_or_output}\n"
            f"Criteria: {json_dumps(criteria)}\n\n"
            "Return JSON with: score (0-100), issues (string array), "
            "hallucination_risk (high/medium/low), factual_errors (string array), "
            'improvements (string array)'
        )
        return await self.llm.generate_json(
            prompt, system_prompt="Be critical and thorough. Flag every concern."
        )

    async def decompose_objective(self, objective: str) -> list[SubTask]:
        plan_items = await self.plan_execute(objective)
        subtasks = []
        for i, item in enumerate(plan_items):
            subtask = SubTask(
                id=item.get("id", str(i + 1)),
                objective=item.get("description", ""),
                description=item.get("success_criteria", ""),
                dependencies=item.get("dependencies", []),
                created_at=utc_now_iso(),
            )
            subtasks.append(subtask)
        return subtasks