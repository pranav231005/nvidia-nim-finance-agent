"""
Comprehensive tool library with autonomous capability.
All tools implement a unified interface with retry, logging, and safety checks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

from src.config import Settings
from src.utils.helpers import build_retry, extract_code_blocks, json_dumps, truncate_text
from src.models import utc_now_iso

LOGGER = logging.getLogger(__name__)


@dataclass
class ToolResult:
    success: bool = True
    output: Any = None
    error: str | None = None
    duration_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """Base class for all tools with logging, retry, and safety hooks."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.call_history: list[dict[str, Any]] = []

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]: ...

    async def execute(self, **kwargs: Any) -> ToolResult:
        start = time.perf_counter()
        self.call_history.append({"args": kwargs, "timestamp": utc_now_iso()})
        try:
            result = await self._run(**kwargs)
            elapsed = int((time.perf_counter() - start) * 1000)
            result.duration_ms = elapsed
            self.call_history[-1]["result"] = str(result.output)[:500]
            LOGGER.info("Tool %s completed in %dms", self.name, elapsed)
            return result
        except Exception as exc:
            elapsed = int((time.perf_counter() - start) * 1000)
            LOGGER.error("Tool %s failed in %dms: %s", self.name, elapsed, exc)
            return ToolResult(success=False, error=str(exc), duration_ms=elapsed)

    @abstractmethod
    async def _run(self, **kwargs: Any) -> ToolResult: ...

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# --- File System Tools ---

class FileReadTool(BaseTool):
    name = "file_read"
    description = "Read the contents of a file from the filesystem."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path to the file"},
            "offset": {"type": "integer", "description": "Line number to start reading from"},
            "limit": {"type": "integer", "description": "Maximum number of lines to read"},
        },
        "required": ["file_path"],
    }

    async def _run(self, file_path: str, offset: int = 0, limit: int = 2000) -> ToolResult:
        path = Path(file_path)
        if not path.exists():
            return ToolResult(success=False, error=f"File not found: {file_path}")
        content = path.read_text(encoding="utf-8")
        lines = content.split("\n")
        if offset or limit < len(lines):
            lines = lines[offset : offset + limit]
        return ToolResult(output="\n".join(lines), metadata={"total_lines": len(content.split('\n'))})


class FileWriteTool(BaseTool):
    name = "file_write"
    description = "Write content to a file, creating or overwriting it."
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path to the file"},
            "content": {"type": "string", "description": "Content to write"},
        },
        "required": ["file_path", "content"],
    }

    async def _run(self, file_path: str, content: str) -> ToolResult:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ToolResult(output=f"Written {len(content)} bytes to {file_path}")


class FileListTool(BaseTool):
    name = "file_list"
    description = "List files and directories at a given path."
    parameters = {
        "type": "object",
        "properties": {
            "directory": {"type": "string", "description": "Directory path to list"},
            "pattern": {"type": "string", "description": "Optional glob pattern to filter"},
        },
        "required": ["directory"],
    }

    async def _run(self, directory: str, pattern: str = "*") -> ToolResult:
        path = Path(directory)
        if not path.exists():
            return ToolResult(success=False, error=f"Directory not found: {directory}")
        files = list(path.glob(pattern))
        output = "\n".join(str(f.relative_to(path)) for f in files[:500])
        return ToolResult(output=output, metadata={"count": len(files)})


# --- Shell / Terminal ---

class ShellTool(BaseTool):
    name = "shell_exec"
    description = "Execute a shell command and return its output."
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "working_dir": {"type": "string", "description": "Working directory"},
            "timeout_seconds": {"type": "integer", "description": "Timeout in seconds"},
        },
        "required": ["command"],
    }

    async def _run(self, command: str, working_dir: str = ".", timeout_seconds: int = 30) -> ToolResult:
        timeout_seconds = min(timeout_seconds, self.settings.sandbox_timeout_seconds)
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_dir,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            proc.kill()
            return ToolResult(success=False, error=f"Command timed out after {timeout_seconds}s")
        output = stdout.decode("utf-8", errors="replace")
        if stderr:
            output += "\n[stderr]\n" + stderr.decode("utf-8", errors="replace")
        return ToolResult(output=truncate_text(output, 5000), metadata={"exit_code": proc.returncode})


# --- Python Execution ---

class PythonExecTool(BaseTool):
    name = "python_exec"
    description = "Execute Python code in a sandboxed environment and return the output."
    parameters = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"},
            "timeout_seconds": {"type": "integer", "description": "Max execution time"},
        },
        "required": ["code"],
    }

    async def _run(self, code: str, timeout_seconds: int = 30) -> ToolResult:
        timeout_seconds = min(timeout_seconds, self.settings.sandbox_timeout_seconds)
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
            output = stdout.decode("utf-8", errors="replace").strip()
            if stderr:
                output += "\n" + stderr.decode("utf-8", errors="replace").strip()
            return ToolResult(output=truncate_text(output, 5000) or "(no output)", metadata={"exit_code": proc.returncode})
        except asyncio.TimeoutError:
            proc.kill()
            return ToolResult(success=False, error=f"Execution timed out after {timeout_seconds}s")


# --- Web Search ---

class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web using DuckDuckGo and return results."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "num_results": {"type": "integer", "description": "Number of results (max 10)"},
        },
        "required": ["query"],
    }

    async def _run(self, query: str, num_results: int = 5) -> ToolResult:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=min(num_results, 10)))
            output = json_dumps([
                {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
                for r in results
            ])
            return ToolResult(output=output, metadata={"result_count": len(results)})
        except ImportError:
            return ToolResult(success=False, error="duckduckgo_search package not installed")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


# --- Web Fetch ---

class WebFetchTool(BaseTool):
    name = "web_fetch"
    description = "Fetch and parse content from a URL, returning the text content."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "max_length": {"type": "integer", "description": "Max characters to return"},
        },
        "required": ["url"],
    }

    async def _run(self, url: str, max_length: int = 5000) -> ToolResult:
        try:
            headers = {"User-Agent": "Autonomous-AI-Agent/1.0"}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            text = re.sub(r"\n{3,}", "\n\n", text)
            return ToolResult(output=truncate_text(text, max_length))
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


# --- GitHub API ---

class GitHubTool(BaseTool):
    name = "github"
    description = "Interact with GitHub: list repos, view files, create issues, search code."
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "description": "Action: search_repos, get_file, create_issue, list_issues, search_code"},
            "repo": {"type": "string", "description": "Repository in owner/repo format"},
            "query": {"type": "string", "description": "Search query or file path"},
            "title": {"type": "string", "description": "Issue title (for create_issue)"},
            "body": {"type": "string", "description": "Issue body (for create_issue)"},
        },
        "required": ["action"],
    }

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/vnd.github+json"}
        if self.settings.github_token:
            headers["Authorization"] = f"Bearer {self.settings.github_token}"
        return headers

    async def _run(self, action: str, repo: str = "", query: str = "",
                    title: str = "", body: str = "") -> ToolResult:
        base = "https://api.github.com"
        try:
            if action == "search_repos":
                resp = requests.get(f"{base}/search/repositories", headers=self._headers(),
                                    params={"q": query, "per_page": 5}, timeout=15)
                data = resp.json()
                items = [{"name": i["full_name"], "stars": i["stargazers_count"],
                          "description": i.get("description", "")} for i in data.get("items", [])]
                return ToolResult(output=json_dumps(items))
            elif action == "get_file":
                resp = requests.get(f"{base}/repos/{repo}/contents/{query}", headers=self._headers(), timeout=15)
                data = resp.json()
                if "content" in data:
                    import base64
                    content = base64.b64decode(data["content"]).decode("utf-8")
                    return ToolResult(output=truncate_text(content, 5000))
                return ToolResult(output=json_dumps(data))
            elif action == "create_issue":
                resp = requests.post(f"{base}/repos/{repo}/issues", headers=self._headers(),
                                     json={"title": title, "body": body}, timeout=15)
                data = resp.json()
                return ToolResult(output=f"Issue created: {data.get('html_url', '')}")
            elif action == "search_code":
                resp = requests.get(f"{base}/search/code", headers=self._headers(),
                                    params={"q": query, "per_page": 5}, timeout=15)
                items = [{"repo": i["repository"]["full_name"], "path": i["path"]}
                         for i in resp.json().get("items", [])]
                return ToolResult(output=json_dumps(items))
            return ToolResult(success=False, error=f"Unknown action: {action}")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


# --- Finance Tools ---

class FinanceTool(BaseTool):
    name = "finance"
    description = "Fetch stock data, financial metrics, news, and analysis."
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "description": "Action: stock_info, history, news, search"},
            "symbol": {"type": "string", "description": "Stock ticker symbol"},
            "period": {"type": "string", "description": "History period: 1d, 5d, 1mo, 3mo, 6mo, 1y"},
        },
        "required": ["action", "symbol"],
    }

    async def _run(self, action: str, symbol: str, period: str = "5d") -> ToolResult:
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            if action == "stock_info":
                info = ticker.info
                key_fields = {
                    "name": info.get("longName"), "sector": info.get("sector"),
                    "industry": info.get("industry"), "marketCap": info.get("marketCap"),
                    "currentPrice": info.get("currentPrice"), "peRatio": info.get("trailingPE"),
                    "dividendYield": info.get("dividendYield"), "52WeekHigh": info.get("fiftyTwoWeekHigh"),
                    "52WeekLow": info.get("fiftyTwoWeekLow"), "currency": info.get("currency"),
                }
                return ToolResult(output=json_dumps(key_fields))
            elif action == "history":
                hist = ticker.history(period=period)
                data = hist.reset_index().to_dict(orient="records")
                return ToolResult(output=json_dumps(data[-5:], default=str))
            elif action == "news":
                news = ticker.news or []
                items = [{"title": n.get("title"), "publisher": n.get("publisher"),
                          "link": n.get("link")} for n in news[:5]]
                return ToolResult(output=json_dumps(items))
            elif action == "search":
                import yfinance as yf_search
                results = yf_search.Search(symbol)
                return ToolResult(output=json_dumps({"quotes": str(results.quotes)[:2000]}))
            return ToolResult(success=False, error=f"Unknown action: {action}")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


# --- Tool Registry ---

class ToolRegistry:
    """Central registry for all tools with schema generation and execution logging."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
        LOGGER.info("Registered tool: %s", tool.name)

    def register_all(self, tools: list[BaseTool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def get_all_schemas(self) -> list[dict[str, Any]]:
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def get_tool_descriptions(self) -> str:
        return "\n".join(
            f"- {name}: {tool.description}" for name, tool in self._tools.items()
        )

    async def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(success=False, error=f"Unknown tool: {tool_name}")
        LOGGER.info("Executing tool %s with args: %s", tool_name, str(kwargs)[:200])
        return await tool.execute(**kwargs)

    def create_default_registry(settings: Settings) -> "ToolRegistry":
        registry = ToolRegistry(settings)
        registry.register_all([
            FileReadTool(settings),
            FileWriteTool(settings),
            FileListTool(settings),
            ShellTool(settings),
            PythonExecTool(settings),
            WebSearchTool(settings),
            WebFetchTool(settings),
            GitHubTool(settings),
            FinanceTool(settings),
        ])
        return registry