"""Core domain models for the autonomous agent system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class TaskStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFLECTING = "reflecting"
    RETRYING = "retrying"


class AgentRole(str, Enum):
    ORCHESTRATOR = "orchestrator"
    PLANNER = "planner"
    RESEARCHER = "researcher"
    FINANCE_ANALYST = "finance_analyst"
    CODER = "coder"
    DEBUGGER = "debugger"
    SEARCHER = "searcher"
    REFLECTOR = "reflector"
    EXECUTOR = "executor"
    REPORT_GENERATOR = "report_generator"
    MEMORY_KEEPER = "memory_keeper"
    CRITIC = "critic"


class Priority(int, Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class MessageType(str, Enum):
    TASK = "task"
    RESULT = "result"
    QUERY = "query"
    RESPONSE = "response"
    EVENT = "event"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


@dataclass
class ToolCall:
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    duration_ms: int = 0
    timestamp: str = ""
    retry_count: int = 0


@dataclass
class SubTask:
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    parent_task_id: str | None = None
    objective: str = ""
    description: str = ""
    assigned_agent: AgentRole | None = None
    status: TaskStatus = TaskStatus.PENDING
    priority: Priority = Priority.MEDIUM
    dependencies: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    result: Any = None
    error: str | None = None
    attempts: int = 0
    max_attempts: int = 3
    created_at: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTask:
    id: str = field(default_factory=lambda: uuid4().hex[:16])
    objective: str = ""
    description: str = ""
    context: str = ""
    subtasks: list[SubTask] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    priority: Priority = Priority.MEDIUM
    assigned_agents: list[AgentRole] = field(default_factory=list)
    completed_subtasks: int = 0
    total_subtasks: int = 0
    result_summary: str = ""
    final_output: Any = None
    errors: list[dict[str, str]] = field(default_factory=list)
    execution_trace: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    human_approvals: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMessage:
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    from_agent: AgentRole | None = None
    to_agent: AgentRole | None = None
    message_type: MessageType = MessageType.TASK
    payload: dict[str, Any] = field(default_factory=dict)
    task_id: str | None = None
    correlation_id: str | None = None
    timestamp: str = ""
    reply_to: str | None = None


@dataclass
class ExecutionContext:
    session_id: str = field(default_factory=lambda: uuid4().hex[:16])
    task: AgentTask | None = None
    active_tool_calls: list[ToolCall] = field(default_factory=list)
    step_count: int = 0
    total_steps: int = 50
    recursion_depth: int = 0
    token_count: int = 0
    cost_usd: float = 0.0
    interrupted: bool = False
    interruption_checkpoint: dict[str, Any] | None = None


@dataclass
class MemoryEntry:
    id: str = field(default_factory=lambda: uuid4().hex[:16])
    content: str = ""
    entry_type: str = "general"  # short_term, long_term, episodic, task, conversation
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    accessed_at: str = ""
    access_count: int = 0
    importance: float = 0.5


@dataclass
class ReflectionResult:
    success: bool = True
    analysis: str = ""
    root_cause: str = ""
    lessons_learned: list[str] = field(default_factory=list)
    improvement_suggestions: list[str] = field(default_factory=list)
    should_retry: bool = False
    retry_strategy: str = ""
    confidence: float = 1.0


@dataclass
class AgentState:
    agent_id: str = field(default_factory=lambda: uuid4().hex[:8])
    role: AgentRole | None = None
    current_task_id: str | None = None
    status: str = "idle"
    last_active: str = ""
    capabilities: list[str] = field(default_factory=list)
    tool_count: int = 0
    tasks_completed: int = 0
    error_count: int = 0


def to_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(obj)
    if isinstance(obj, list):
        return [to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {key: to_dict(value) for key, value in obj.items()}
    if isinstance(obj, Enum):
        return obj.value
    return obj


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"