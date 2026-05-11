"""Unit tests for core domain models."""

import pytest
from src.models import (
    AgentTask, Priority, ReflectionResult, SubTask, TaskStatus, to_dict, utc_now_iso,
)


class TestSubTask:
    def test_creation(self) -> None:
        st = SubTask(objective="Test task", description="A test")
        assert st.objective == "Test task"
        assert st.status == TaskStatus.PENDING
        assert st.attempts == 0

    def test_status_transitions(self) -> None:
        st = SubTask(objective="Test")
        st.status = TaskStatus.IN_PROGRESS
        assert st.status == TaskStatus.IN_PROGRESS
        st.status = TaskStatus.COMPLETED
        assert st.status == TaskStatus.COMPLETED


class TestAgentTask:
    def test_creation(self) -> None:
        task = AgentTask(objective="Build a dashboard", description="Context here")
        assert task.objective == "Build a dashboard"
        assert task.status == TaskStatus.PENDING
        assert task.total_subtasks == 0

    def test_with_subtasks(self) -> None:
        task = AgentTask(objective="Analyze stocks")
        st = SubTask(objective="Fetch AAPL data", parent_task_id=task.id)
        task.subtasks.append(st)
        task.total_subtasks = 1
        assert len(task.subtasks) == 1
        assert task.total_subtasks == 1


class TestToDict:
    def test_converts_dataclass(self) -> None:
        st = SubTask(objective="Hello")
        result = to_dict(st)
        assert isinstance(result, dict)
        assert result["objective"] == "Hello"

    def test_converts_list(self) -> None:
        items = [SubTask(objective="A"), SubTask(objective="B")]
        result = to_dict(items)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["objective"] == "A"


class TestUtcNowIso:
    def test_format(self) -> None:
        ts = utc_now_iso()
        assert ts.endswith("Z")
        assert "T" in ts


class TestReflectionResult:
    def test_defaults(self) -> None:
        rr = ReflectionResult()
        assert rr.success is True
        assert rr.should_retry is False
        assert rr.confidence == 1.0