"""Unit tests for tool registry and tools."""

import pytest
from src.config import Settings
from src.tools.registry import FileReadTool, ToolRegistry, FileListTool


class TestToolRegistry:
    @pytest.fixture
    def registry(self) -> ToolRegistry:
        settings = Settings()
        return ToolRegistry(settings)

    def test_registry_empty(self, registry: ToolRegistry) -> None:
        assert len(registry.list_tools()) == 0

    def test_register_tool(self, registry: ToolRegistry) -> None:
        settings = Settings()
        tool = FileReadTool(settings)
        registry.register(tool)
        assert "file_read" in registry.list_tools()
        assert registry.get("file_read") is tool

    def test_get_all_schemas(self, registry: ToolRegistry) -> None:
        settings = Settings()
        registry.register(FileReadTool(settings))
        schemas = registry.get_all_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "file_read"

    def test_create_default_registry(self) -> None:
        settings = Settings()
        registry = ToolRegistry.create_default_registry(settings)
        tools = registry.list_tools()
        assert "file_read" in tools
        assert "web_search" in tools
        assert "shell_exec" in tools
        assert "python_exec" in tools
        assert "github" in tools
        assert "finance" in tools


class TestFileReadTool:
    @pytest.fixture
    def tool(self) -> FileReadTool:
        return FileReadTool(Settings())

    @pytest.mark.asyncio
    async def test_read_existing_file(self, tool: FileReadTool, tmp_path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        result = await tool.execute(file_path=str(f))
        assert result.success
        assert "hello world" in result.output

    @pytest.mark.asyncio
    async def test_read_missing_file(self, tool: FileReadTool) -> None:
        result = await tool.execute(file_path="/nonexistent/file.txt")
        assert not result.success
        assert "File not found" in result.error


class TestFileListTool:
    @pytest.fixture
    def tool(self) -> FileListTool:
        return FileListTool(Settings())

    @pytest.mark.asyncio
    async def test_list_directory(self, tool: FileListTool, tmp_path) -> None:
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        result = await tool.execute(directory=str(tmp_path))
        assert result.success
        assert "a.txt" in result.output
        assert "b.txt" in result.output