"""Multi-provider LLM client with retry, cost tracking, and streaming."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from src.config import Settings
from src.utils.helpers import build_retry, parse_json_response

LOGGER = logging.getLogger(__name__)


class LLMProvider:
    """Unified LLM interface supporting OpenAI, Anthropic, and Google providers."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._openai: AsyncOpenAI | None = None
        self._anthropic: Any = None
        self._google: Any = None
        self._nim: AsyncOpenAI | None = None
        self._init_clients()

    def _init_clients(self) -> None:
        if self.settings.openai_api_key:
            self._openai = AsyncOpenAI(api_key=self.settings.openai_api_key)

        if self.settings.anthropic_api_key:
            try:
                from anthropic import AsyncAnthropic
                self._anthropic = AsyncAnthropic(api_key=self.settings.anthropic_api_key)
            except ImportError:
                pass

        if self.settings.google_api_key:
            try:
                from google import genai
                self._google = genai.Client(api_key=self.settings.google_api_key)
            except ImportError:
                pass

        if self.settings.nim_api_key:
            self._nim = AsyncOpenAI(
                api_key=self.settings.nim_api_key,
                base_url=self.settings.nim_base_url,
            )

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> dict[str, Any]:
        model = model or self.settings.openai_model
        temperature = temperature or self.settings.llm_temperature
        max_tokens = max_tokens or self.settings.llm_max_tokens

        provider = self._detect_provider(model)
        LOGGER.debug("LLM call: provider=%s model=%s prompt_len=%d", provider, model, len(prompt))

        if provider == "openai":
            return await self._generate_openai(prompt, system_prompt, model, temperature, max_tokens, json_mode, tools, tool_choice)
        elif provider == "anthropic":
            return await self._generate_anthropic(prompt, system_prompt, model, temperature, max_tokens, json_mode, tools)
        elif provider == "google":
            return await self._generate_google(prompt, system_prompt, model, temperature, max_tokens)
        elif provider == "nim":
            return await self._generate_nim(prompt, system_prompt, model, temperature, max_tokens, json_mode, tools, tool_choice)
        else:
            raise ValueError(f"Unknown provider for model: {model}")

    def _detect_provider(self, model: str) -> str:
        model_lower = model.lower()
        if any(p in model_lower for p in ["gpt", "o1", "o3", "o4"]):
            return "openai"
        if any(p in model_lower for p in ["claude"]):
            return "anthropic"
        if any(p in model_lower for p in ["gemini"]):
            return "google"
        if any(p in model_lower for p in ["deepseek", "nim"]):
            return "nim"
        return self.settings.default_llm_provider

    async def _generate_openai(
        self, prompt: str, system: str | None, model: str,
        temperature: float, max_tokens: int, json_mode: bool,
        tools: list[dict] | None, tool_choice: str | None,
    ) -> dict[str, Any]:
        if not self._openai:
            raise RuntimeError("OpenAI client not configured")

        retry = build_retry(self.settings.llm_max_retries)
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        @retry
        async def _call() -> Any:
            return await self._openai.chat.completions.create(**kwargs)

        response = await _call()
        choice = response.choices[0]
        result: dict[str, Any] = {
            "content": choice.message.content or "",
            "model": response.model,
            "finish_reason": choice.finish_reason,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
        }
        if choice.message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in choice.message.tool_calls
            ]
        return result

    async def _generate_anthropic(
        self, prompt: str, system: str | None, model: str,
        temperature: float, max_tokens: int, json_mode: bool,
        tools: list[dict] | None,
    ) -> dict[str, Any]:
        if not self._anthropic:
            raise RuntimeError("Anthropic client not configured")

        retry = build_retry(self.settings.llm_max_retries)

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        @retry
        async def _call() -> Any:
            return await self._anthropic.messages.create(**kwargs)

        response = await _call()
        content = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "function": {"name": block.name, "arguments": json.dumps(block.input)},
                })

        result: dict[str, Any] = {
            "content": content,
            "model": response.model,
            "finish_reason": response.stop_reason,
            "usage": {
                "prompt_tokens": response.usage.input_tokens if hasattr(response, 'usage') else 0,
                "completion_tokens": response.usage.output_tokens if hasattr(response, 'usage') else 0,
                "total_tokens": (response.usage.input_tokens + response.usage.output_tokens) if hasattr(response, 'usage') else 0,
            },
        }
        if tool_calls:
            result["tool_calls"] = tool_calls
        return result

    async def _generate_google(
        self, prompt: str, system: str | None, model: str,
        temperature: float, max_tokens: int,
    ) -> dict[str, Any]:
        if not self._google:
            raise RuntimeError("Google client not configured")

        full_prompt = prompt
        if system:
            full_prompt = f"System: {system}\n\nUser: {prompt}"

        retry = build_retry(self.settings.llm_max_retries)

        @retry
        async def _call() -> Any:
            return self._google.models.generate_content(
                model=model,
                contents=full_prompt,
            )

        response = await _call()
        return {
            "content": response.text,
            "model": model,
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    async def _generate_nim(
        self, prompt: str, system: str | None, model: str,
        temperature: float, max_tokens: int, json_mode: bool,
        tools: list[dict] | None, tool_choice: str | None,
    ) -> dict[str, Any]:
        if not self._nim:
            raise RuntimeError("NVIDIA NIM client not configured")

        retry = build_retry(self.settings.llm_max_retries)
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        @retry
        async def _call() -> Any:
            return await self._nim.chat.completions.create(**kwargs)

        response = await _call()
        choice = response.choices[0]
        result: dict[str, Any] = {
            "content": choice.message.content or "",
            "model": response.model,
            "finish_reason": choice.finish_reason,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
        }
        if choice.message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in choice.message.tool_calls
            ]
        return result

    async def generate_json(
        self, prompt: str, system_prompt: str | None = None, model: str | None = None,
    ) -> dict[str, Any]:
        result = await self.generate(
            prompt=prompt, system_prompt=system_prompt, model=model, json_mode=True,
        )
        return parse_json_response(result["content"])

    async def generate_with_tools(
        self, prompt: str, tools: list[dict[str, Any]],
        system_prompt: str | None = None, model: str | None = None,
    ) -> dict[str, Any]:
        return await self.generate(
            prompt=prompt, system_prompt=system_prompt, model=model,
            tools=tools, tool_choice="auto",
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not self._openai:
            raise RuntimeError("OpenAI client required for embeddings")
        response = await self._openai.embeddings.create(
            model=self.settings.embedding_model,
            input=texts,
        )
        return [d.embedding for d in response.data]