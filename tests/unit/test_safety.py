"""Unit tests for safety guardrails."""

import pytest
from src.config import Settings
from src.safety import Guardrails, HallucinationDetector


class TestHallucinationDetector:
    @pytest.fixture
    def detector(self) -> HallucinationDetector:
        return HallucinationDetector()

    def test_clean_text(self, detector: HallucinationDetector) -> None:
        result = detector.detect("The stock price increased 5% today following the earnings report.")
        assert result["confidence"] >= 0.8

    def test_refusal_pattern(self, detector: HallucinationDetector) -> None:
        result = detector.detect("I don't know the answer to that question.")
        assert result["flagged"]

    def test_excessive_hedging(self, detector: HallucinationDetector) -> None:
        text = "It could be that maybe the price might be around possibly approximately roughly about nearly $100."
        result = detector.detect(text)
        assert result["confidence"] < 0.9


class TestGuardrails:
    @pytest.fixture
    def guardrails(self) -> Guardrails:
        return Guardrails(Settings())

    def test_detects_destructive_commands(self, guardrails: Guardrails) -> None:
        assert guardrails.is_destructive("rm -rf /")
        assert guardrails.is_destructive("sudo rm something")
        assert guardrails.is_destructive("git push --force origin main")
        assert not guardrails.is_destructive("ls -la")

    def test_sanitize_input(self, guardrails: Guardrails) -> None:
        assert guardrails.sanitize_input("hello\x00world") == "helloworld"
        assert guardrails.sanitize_input("a" * 100_000) == "a" * 50_000

    def test_cost_tracking(self, guardrails: Guardrails) -> None:
        cost = guardrails.track_cost(1000, 500, "gpt-4o")
        assert cost > 0
        assert isinstance(cost, float)