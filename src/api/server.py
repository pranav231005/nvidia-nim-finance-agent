"""
FastAPI server with WebSocket support for agent interaction and monitoring.
Provides REST endpoints and real-time agent communication.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.models import Priority, TaskStatus, utc_now_iso
from src.observability import configure_observability
from src.orchestration.engine import AgentOrchestrator
from src.utils.helpers import json_dumps

LOGGER = logging.getLogger(__name__)

settings: Settings | None = None
orchestrator: AgentOrchestrator | None = None
ws_connections: dict[str, WebSocket] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global settings, orchestrator
    settings = get_settings()
    configure_observability(settings)
    orchestrator = AgentOrchestrator(settings)
    LOGGER.info("Autonomous AI Agent System started")
    yield
    LOGGER.info("Autonomous AI Agent System shutting down")


app = FastAPI(
    title="Autonomous AI Agent",
    description="Production-grade multi-agent autonomous system with planning, tool use, and reflection.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response Models ---

class ExecuteRequest(BaseModel):
    objective: str = Field(..., description="The objective to accomplish autonomously")
    context: str = Field(default="", description="Additional context for the task")
    priority: str = Field(default="medium", description="Task priority: critical, high, medium, low")
    max_steps: int | None = Field(default=None, description="Max autonomous execution steps")


class ExecuteResponse(BaseModel):
    task_id: str
    status: str
    result_summary: str
    final_output: str = ""
    subtasks_completed: int = 0
    total_subtasks: int = 0
    errors: list[dict[str, str]] = []
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    execution_trace: list[dict[str, Any]] = []


class StatusResponse(BaseModel):
    active_tasks: int
    completed_tasks: int
    agents: dict[str, Any]
    memory: dict[str, Any]
    session_cost_usd: float


# --- Routes ---

@app.get("/")
async def root() -> dict[str, str]:
    return {
        "name": "Autonomous AI Agent System",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "timestamp": utc_now_iso()}


@app.get("/status", response_model=StatusResponse)
async def get_status() -> dict[str, Any]:
    if not orchestrator:
        raise HTTPException(503, "Orchestrator not initialized")
    return orchestrator.get_status()


@app.post("/execute", response_model=ExecuteResponse)
async def execute_task(req: ExecuteRequest) -> dict[str, Any]:
    """Submit an objective for autonomous execution."""
    if not orchestrator:
        raise HTTPException(503, "Orchestrator not initialized")

    priority = Priority[req.priority.upper()] if req.priority.upper() in Priority.__members__ else Priority.MEDIUM

    task = await orchestrator.execute(
        objective=req.objective,
        context=req.context,
        priority=priority,
        max_steps=req.max_steps,
    )

    return {
        "task_id": task.id,
        "status": task.status.value,
        "result_summary": task.result_summary[:1000],
        "final_output": str(task.final_output)[:5000],
        "subtasks_completed": task.completed_subtasks,
        "total_subtasks": task.total_subtasks,
        "errors": task.errors[:20],
        "total_tokens_used": task.total_tokens_used,
        "total_cost_usd": task.total_cost_usd,
        "execution_trace": task.execution_trace[-50:],
    }


@app.get("/tasks/{task_id}")
async def get_task(task_id: str) -> dict[str, Any]:
    """Retrieve a specific task by ID."""
    if not orchestrator:
        raise HTTPException(503)
    for task in orchestrator._task_history:
        if task.id == task_id:
            from src.models import to_dict
            return to_dict(task)
    raise HTTPException(404, f"Task {task_id} not found")


@app.get("/tasks")
async def list_tasks(limit: int = 20) -> list[dict[str, Any]]:
    """List recent tasks."""
    if not orchestrator:
        raise HTTPException(503)
    from src.models import to_dict
    return [to_dict(t) for t in orchestrator._task_history[-limit:]]


@app.get("/memory/search")
async def search_memory(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Search vector memory for relevant context."""
    if not orchestrator:
        raise HTTPException(503)
    return await orchestrator.memory.retrieve(query, top_k=top_k, search_episodic=True)


@app.get("/tools")
async def list_tools() -> list[dict[str, Any]]:
    """List available tools and their schemas."""
    if not orchestrator:
        raise HTTPException(503)
    return orchestrator.tools.get_all_schemas()


@app.post("/tools/{tool_name}")
async def invoke_tool(tool_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Manually invoke a tool."""
    if not orchestrator:
        raise HTTPException(503)
    result = await orchestrator.tools.execute(tool_name, **kwargs)
    return {
        "success": result.success,
        "output": str(result.output)[:10000] if result.output else None,
        "error": result.error,
        "duration_ms": result.duration_ms,
    }


@app.get("/observability/summary")
async def observability_summary() -> dict[str, Any]:
    if not orchestrator:
        raise HTTPException(503)
    return orchestrator.telemetry.get_summary()


@app.get("/observability/events")
async def observability_events(limit: int = 50) -> list[dict[str, Any]]:
    if not orchestrator:
        raise HTTPException(503)
    return orchestrator.telemetry.recent_events(limit)


# --- WebSocket for real-time agent monitoring ---

class ConnectionManager:
    def __init__(self) -> None:
        self.active: dict[str, WebSocket] = {}

    async def connect(self, ws: WebSocket) -> str:
        await ws.accept()
        conn_id = uuid.uuid4().hex[:12]
        self.active[conn_id] = ws
        return conn_id

    def disconnect(self, conn_id: str) -> None:
        self.active.pop(conn_id, None)

    async def broadcast(self, message: dict[str, Any]) -> None:
        dead = []
        for conn_id, ws in self.active.items():
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(conn_id)
        for conn_id in dead:
            self.disconnect(conn_id)

    async def send(self, conn_id: str, message: dict[str, Any]) -> None:
        ws = self.active.get(conn_id)
        if ws:
            await ws.send_json(message)


ws_manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    conn_id = await ws_manager.connect(ws)
    LOGGER.info("WebSocket connected: %s", conn_id)
    try:
        # Send initial status
        if orchestrator:
            await ws_manager.send(conn_id, {
                "type": "status", "data": orchestrator.get_status(),
            })

        while True:
            data = await ws.receive_text()
            message = json.loads(data)

            if message.get("type") == "execute":
                if orchestrator:
                    task = await orchestrator.execute(
                        objective=message["objective"],
                        context=message.get("context", ""),
                    )
                    await ws_manager.send(conn_id, {
                        "type": "task_complete",
                        "task_id": task.id,
                        "status": task.status.value,
                        "result": str(task.result_summary)[:2000],
                    })

            elif message.get("type") == "ping":
                await ws_manager.send(conn_id, {"type": "pong"})

    except WebSocketDisconnect:
        ws_manager.disconnect(conn_id)
        LOGGER.info("WebSocket disconnected: %s", conn_id)
    except Exception as exc:
        LOGGER.error("WebSocket error: %s", exc)
        ws_manager.disconnect(conn_id)


# --- Entry Point ---

def run():
    import uvicorn
    s = get_settings()
    uvicorn.run(
        "src.api.server:app",
        host=s.api_host,
        port=s.api_port,
        workers=s.api_workers if s.environment == "production" else 1,
        reload=s.debug,
        log_level=s.log_level.lower(),
    )


if __name__ == "__main__":
    run()