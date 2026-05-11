"""
Multi-tier memory system with vector search (ChromaDB) and RAG retrieval.

Memory tiers:
1. Short-term: In-memory ring buffer for recent context
2. Long-term: ChromaDB vector store with embeddings
3. Episodic: Task execution traces and their outcomes
4. Task memory: Active task state and subtask progress
5. Conversation memory: Agent-to-agent communication history
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import Settings
from src.llm import LLMProvider
from src.models import MemoryEntry, to_dict
from src.utils.helpers import ensure_directory, json_dumps
from src.models import utc_now_iso

LOGGER = logging.getLogger(__name__)


class ShortTermMemory:
    """Ring-buffer for recent observations and context."""

    def __init__(self, max_items: int = 100) -> None:
        self._buffer: deque[dict[str, Any]] = deque(maxlen=max_items)

    def add(self, entry: dict[str, Any]) -> None:
        entry["_stored_at"] = utc_now_iso()
        self._buffer.append(entry)

    def get_all(self) -> list[dict[str, Any]]:
        return list(self._buffer)

    def get_recent(self, n: int = 20) -> list[dict[str, Any]]:
        items = list(self._buffer)
        return items[-n:] if len(items) > n else items

    def clear(self) -> None:
        self._buffer.clear()

    @property
    def size(self) -> int:
        return len(self._buffer)


class VectorMemory:
    """Long-term memory backed by ChromaDB with embedding-based retrieval."""

    def __init__(self, settings: Settings, llm: LLMProvider) -> None:
        self.settings = settings
        self.llm = llm
        persist_dir = ensure_directory(Path(settings.chroma_persist_directory))
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collections: dict[str, Any] = {}

    def _get_collection(self, name: str) -> Any:
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    async def store(
        self, content: str, collection: str = "long_term",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        entry_id = MemoryEntry(content=content, metadata=metadata or {}).id
        embedding = (await self.llm.embed([content]))[0]
        col = self._get_collection(collection)
        col.add(
            ids=[entry_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata or {}],
        )
        return entry_id

    async def query(
        self, query: str, collection: str = "long_term",
        top_k: int | None = None, threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        top_k = top_k or self.settings.vector_search_top_k
        threshold = threshold or self.settings.memory_similarity_threshold
        embedding = (await self.llm.embed([query]))[0]
        col = self._get_collection(collection)
        results = col.query(query_embeddings=[embedding], n_results=top_k)
        items = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results.get("distances") else 0
                similarity = 1 - distance
                if similarity >= threshold:
                    items.append({
                        "id": doc_id,
                        "content": results["documents"][0][i] if results.get("documents") else "",
                        "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                        "similarity": similarity,
                    })
        return items

    def delete_collection(self, name: str) -> None:
        try:
            self._client.delete_collection(name)
            self._collections.pop(name, None)
        except Exception:
            pass

    def collection_count(self, name: str) -> int:
        try:
            return self._get_collection(name).count()
        except Exception:
            return 0


class MemoryManager:
    """Orchestrates all memory tiers with unified store/retrieve API."""

    def __init__(self, settings: Settings, llm: LLMProvider) -> None:
        self.settings = settings
        self.llm = llm
        self.short_term = ShortTermMemory(max_items=settings.short_term_memory_limit)
        self.vector = VectorMemory(settings, llm)
        self.conversation_history: deque[dict[str, Any]] = deque(maxlen=200)
        self.error_log: deque[dict[str, Any]] = deque(maxlen=100)

    async def store_short_term(self, event_type: str, data: dict[str, Any]) -> None:
        self.short_term.add({"type": event_type, "data": data})

    async def store_long_term(
        self, content: str, entry_type: str = "general", metadata: dict[str, Any] | None = None,
    ) -> str:
        meta = metadata or {}
        meta["entry_type"] = entry_type
        return await self.vector.store(content, collection="long_term", metadata=meta)

    async def store_episodic(self, task_id: str, task_data: dict[str, Any]) -> str:
        summary = (
            f"Task {task_id}: {task_data.get('objective', '')} "
            f"Status: {task_data.get('status', '')} "
            f"Result: {str(task_data.get('result_summary', ''))[:500]}"
        )
        return await self.vector.store(
            content=summary,
            collection="episodic",
            metadata={"task_id": task_id, **task_data},
        )

    async def store_conversation(self, from_agent: str, to_agent: str, message: str) -> None:
        entry = {
            "from": from_agent,
            "to": to_agent,
            "message": message,
            "timestamp": utc_now_iso(),
        }
        self.conversation_history.append(entry)
        await self.store_short_term("conversation", entry)

    async def store_error(self, error_type: str, error_message: str, context: dict[str, Any]) -> None:
        entry = {
            "type": error_type,
            "message": error_message,
            "context": context,
            "timestamp": utc_now_iso(),
        }
        self.error_log.append(entry)
        await self.vector.store(
            content=f"Error: {error_type} - {error_message}",
            collection="errors",
            metadata=entry,
        )

    async def retrieve(self, query: str, top_k: int = 5,
                        search_episodic: bool = False) -> list[dict[str, Any]]:
        results = await self.vector.query(query, "long_term", top_k=top_k)
        if search_episodic:
            episodic = await self.vector.query(query, "episodic", top_k=top_k)
            results.extend(episodic)
        results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        return results[:top_k]

    async def retrieve_similar_errors(self, error_message: str, top_k: int = 3) -> list[dict[str, Any]]:
        return await self.vector.query(error_message, "errors", top_k=top_k)

    async def retrieve_context_for_task(self, task: str, top_k: int = 5) -> str:
        results = await self.retrieve(task, top_k=top_k, search_episodic=True)
        recent = self.short_term.get_recent(10)
        parts = []

        if recent:
            parts.append("=== Recent Observations ===")
            for item in recent:
                parts.append(f"- {json_dumps(item)}")

        if results:
            parts.append("\n=== Related Past Knowledge ===")
            for r in results:
                parts.append(f"- [{r['similarity']:.2f}] {r['content']}")

        return "\n".join(parts)

    async def get_conversation_context(self, n: int = 20) -> str:
        items = list(self.conversation_history)[-n:]
        return "\n".join(
            f"[{e['from']} -> {e['to']}]: {e['message']}" for e in items
        )

    def get_short_term_summary(self) -> str:
        recent = self.short_term.get_recent(10)
        return json_dumps(recent)

    async def consolidate(self) -> None:
        """Periodically consolidate short-term into long-term memory."""
        items = self.short_term.get_all()
        if items:
            summary = json_dumps(items)
            await self.store_long_term(
                content=summary, entry_type="consolidated",
                metadata={"item_count": len(items), "consolidated_at": utc_now_iso()},
            )
            self.short_term.clear()
            LOGGER.info("Consolidated %d items from short-term to long-term memory", len(items))