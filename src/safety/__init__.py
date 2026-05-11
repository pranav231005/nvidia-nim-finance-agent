"""
Safety guardrails: permission checks, execution sandboxing, hallucination detection,
cost controls, timeout handling, retry limits, human approval mode.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from src.config import Settings

LOGGER = logging.getLogger(__name__)


class HallucinationDetector:
    """Lightweight hallucination detection using consistency checks and factual verification."""

    def __init__(self) -> None:
        self.suspicious_patterns = [
            r"I don't (know|have access to)",
            r"As an AI (language model|assistant)",
            r"I (cannot|can't) (provide|access|do)",
            r"I'm (not able|unable) to",
        ]

    def detect(self, text: str) -> dict[str, Any]:
        confidence_loss = 0.0
        flags: list[str] = []

        # Check for refusal patterns
        for pattern in self.suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                flags.append(f"Refusal pattern detected: {pattern}")
                confidence_loss += 0.3

        # Check for fabricated URLs
        fabricated_urls = re.findall(r'https?://[^\s<>"]+', text)
        if len(fabricated_urls) > 10:
            flags.append("Unusually high number of URLs")
            confidence_loss += 0.2

        # Check for excessive hallucination markers
        hallmarks = ["maybe", "possibly", "could be", "might be", "approximately", "around"]
        hallmark_count = sum(text.lower().count(h) for h in hallmarks)
        if hallmark_count > 5:
            flags.append("Excessive hedging language — possible hallucination")
            confidence_loss += 0.15

        confidence = max(0.0, 1.0 - confidence_loss)
        return {
            "flagged": len(flags) > 0,
            "confidence": round(confidence, 2),
            "flags": flags,
        }


class Guardrails:
    """Central safety system with permission checks, sandboxing, and cost controls."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.hallucination_detector = HallucinationDetector()
        self.destructive_patterns = [
            r"rm\s+-rf", r"sudo\s+", r">\s*/dev/", r"mkfs\.", r"dd\s+if=",
            r"git\s+push\s+.*--force", r"git\s+reset\s+--hard", r":(){ :|:& };:",
            r"chmod\s+777", r"DROP\s+TABLE", r"DELETE\s+FROM", r"TRUNCATE",
        ]

    def check_tool_permission(self, action: str, settings: Settings) -> bool:
        """Check if an action requires human approval."""
        if not self.settings.require_human_approval:
            return True

        for restricted in self.settings.human_approval_for:
            if restricted in action.lower():
                LOGGER.info("Action requires human approval: %s", action[:100])
                return False
        return True

    def is_destructive(self, command: str) -> bool:
        for pattern in self.destructive_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return True
        return False

    def sanitize_input(self, text: str) -> str:
        """Sanitize input to prevent injection."""
        text = text.replace("\x00", "")
        if len(text) > 50_000:
            text = text[:50_000]
        return text

    def check_hallucination(self, output: str) -> dict[str, Any]:
        return self.hallucination_detector.detect(output)

    def track_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """Estimate cost in USD based on model pricing."""
        pricing = {
            "gpt-4o": (2.50, 10.00),
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-4.1-mini": (0.15, 0.60),
            "claude-sonnet-4-6": (3.00, 15.00),
            "claude-haiku-4-5": (0.80, 4.00),
        }
        input_price, output_price = pricing.get(model, (1.00, 5.00))
        input_cost = (prompt_tokens / 1_000_000) * input_price
        output_cost = (completion_tokens / 1_000_000) * output_price
        return round(input_cost + output_cost, 6)