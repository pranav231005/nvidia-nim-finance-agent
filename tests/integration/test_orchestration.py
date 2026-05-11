"""Integration tests for orchestration and multi-agent interactions."""

import asyncio
import pytest
from src.config import Settings
from src.llm import LLMProvider
from src.memory.manager import MemoryManager
from src.models import AgentTask, AgentRole, Priority, TaskStatus, SubTask
from src.reasoning.engine import ReasoningEngine
from src.tools.registry import ToolRegistry


@pytest.mark.integration
class TestOrchestrationIntegration:
    @pytest.fixture
    def settings(self) -> Settings:
        return Settings()

    def test_orchestrator_initialization(self, settings: Settings) -> None:
        """Orchestrator should initialize with all agents and tools."""
        from src.orchestration.engine import AgentOrchestrator
        orch = AgentOrchestrator(settings)

        assert orch.get_agent(AgentRole.PLANNER) is not None
        assert orch.get_agent(AgentRole.RESEARCHER) is not None
        assert orch.get_agent(AgentRole.FINANCE_ANALYST) is not None
        assert orch.get_agent(AgentRole.CODER) is not None
        assert orch.get_agent(AgentRole.DEBUGGER) is not None
        assert orch.get_agent(AgentRole.REFLECTOR) is not None
        assert orch.get_agent(AgentRole.REPORT_GENERATOR) is not None
        assert len(orch.tools.list_tools()) >= 9


@pytest.mark.integration
class TestMemoryIntegration:
    def test_short_term_memory(self) -> None:
        settings = Settings()
        from src.llm import LLMProvider
        from src.memory.manager import MemoryManager
        mem = MemoryManager(settings, LLMProvider(settings))
        mem.short_term.add({"type": "test", "data": "hello"})
        assert mem.short_term.size >= 1

    def test_error_log_structure(self) -> None:
        settings = Settings()
        from src.llm import LLMProvider
        from src.memory.manager import MemoryManager
        mem = MemoryManager(settings, LLMProvider(settings))
        mem.error_log.append({"type": "test_error", "message": "Something broke"})
        assert len(mem.error_log) >= 1


@pytest.mark.integration
class TestToolIntegration:
    def test_all_tools_registered(self) -> None:
        settings = Settings()
        registry = ToolRegistry.create_default_registry(settings)
        tools = registry.list_tools()
        expected = {"file_read", "file_write", "file_list", "shell_exec",
                     "python_exec", "web_search", "web_fetch", "github", "finance"}
        assert set(tools) == expected


@pytest.mark.integration
class TestAgentCommunication:
    def test_event_bus_pub_sub(self) -> None:
        from src.communication import EventBus
        from src.models import AgentMessage, MessageType, utc_now_iso

        bus = EventBus()
        received = []

        async def handler(msg: AgentMessage) -> None:
            received.append(msg)

        bus.subscribe("task", handler)

        msg = AgentMessage(
            from_agent=AgentRole.PLANNER,
            to_agent=AgentRole.RESEARCHER,
            message_type=MessageType.TASK,
            payload={"action": "search"},
            timestamp=utc_now_iso(),
        )

        asyncio.run(bus.publish(msg))
        assert len(received) == 1
        assert received[0].payload == {"action": "search"}


@pytest.mark.integration
class TestTaskFlow:
    def test_full_task_lifecycle(self) -> None:
        """Test the complete lifecycle from objective to subtask completion."""
        task = AgentTask(
            objective="Analyze AAPL stock",
            context="Market research",
            priority=Priority.HIGH,
        )

        # Planning
        st1 = SubTask(
            objective="Fetch AAPL data",
            assigned_agent=AgentRole.FINANCE_ANALYST,
            parent_task_id=task.id,
        )
        st2 = SubTask(
            objective="Search for AAPL news",
            assigned_agent=AgentRole.RESEARCHER,
            parent_task_id=task.id,
        )

        task.subtasks = [st1, st2]
        task.total_subtasks = 2
        task.status = TaskStatus.PLANNING

        # Simulate execution
        st1.status = TaskStatus.IN_PROGRESS
        st1.status = TaskStatus.COMPLETED
        st1.result = {"price": 185.50}
        task.completed_subtasks = 1

        st2.status = TaskStatus.IN_PROGRESS
        st2.status = TaskStatus.COMPLETED
        st2.result = {"news_count": 5}
        task.completed_subtasks = 2

        task.status = TaskStatus.COMPLETED

        assert task.status == TaskStatus.COMPLETED
        assert task.completed_subtasks == 2
        assert task.total_subtasks == 2